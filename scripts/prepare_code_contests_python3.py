import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HF_DATASET_DIR = DATA_DIR / "code_contests" / "hf_dataset"
OUT_DIR = DATA_DIR / "code_contests_python3"
DATASET_INFO = DATA_DIR / "dataset_info.json"
PYTHON3_PROMPT_SUFFIX = (
    "Write a Python 3 program that reads from standard input and writes to standard output. "
    "Return only the code, without Markdown fences."
)


def label_name(feature: Any, value: Any) -> str:
    if value is None:
        return ""
    try:
        return feature.int2str(int(value))
    except Exception:
        return str(value)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def python3_solution(row: dict[str, Any], language_feature: Any) -> str:
    languages = row["solutions"]["language"]
    solutions = row["solutions"]["solution"]
    for language, solution in zip(languages, solutions):
        if label_name(language_feature, language) == "PYTHON3" and solution:
            return str(solution)
    return ""


def instruction_for(row: dict[str, Any]) -> str:
    return f"{row['description'].strip()}\n\n{PYTHON3_PROMPT_SUFFIX}"


def common_fields(
    row: dict[str, Any],
    split: str,
    index: int,
    source_feature: Any,
    difficulty_feature: Any,
) -> dict[str, Any]:
    return {
        "problem_id": f"{split}:{index}",
        "name": row["name"],
        "source_name": label_name(source_feature, row["source"]),
        "difficulty_name": label_name(difficulty_feature, row["difficulty"]),
        "cf_contest_id": row["cf_contest_id"],
        "cf_index": row["cf_index"],
        "cf_points": row["cf_points"],
        "cf_rating": row["cf_rating"],
        "cf_tags": row["cf_tags"],
        "source": "deepmind/code_contests",
        "split": split,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def dataset_info_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }


def update_dataset_info() -> None:
    info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = {}
    for stage in ("sft", "rl"):
        for split in ("train", "valid", "test"):
            entries[f"code_contests_py3_{stage}_{split}"] = dataset_info_entry(
                f"code_contests_python3/{stage}_{split}.jsonl"
            )

    info.update(entries)
    with DATASET_INFO.open("w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def build() -> dict[str, dict[str, int]]:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(HF_DATASET_DIR))
    source_feature = dataset["train"].features["source"]
    difficulty_feature = dataset["train"].features["difficulty"]
    language_feature = dataset["train"].features["solutions"]["language"].feature
    summary: dict[str, dict[str, int]] = {}

    for split, split_ds in dataset.items():
        sft_rows = []
        rl_rows = []
        for index, row in enumerate(split_ds):
            solution = python3_solution(row, language_feature)
            if not solution:
                continue

            fields = common_fields(row, split, index, source_feature, difficulty_feature)
            sft_rows.append(
                {
                    "instruction": instruction_for(row),
                    "input": "",
                    "output": solution,
                    "solution_language": "PYTHON3",
                    **fields,
                }
            )
            rl_rows.append(
                {
                    "instruction": instruction_for(row),
                    "input": "",
                    "output": "",
                    **fields,
                }
            )

        write_jsonl(OUT_DIR / f"sft_{split}.jsonl", sft_rows)
        write_jsonl(OUT_DIR / f"rl_{split}.jsonl", rl_rows)
        summary[split] = {"sft": len(sft_rows), "rl": len(rl_rows)}

    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Python 3-only CodeContests data.")
    parser.add_argument("--no-update-dataset-info", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build()
    if not args.no_update_dataset_info:
        update_dataset_info()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
