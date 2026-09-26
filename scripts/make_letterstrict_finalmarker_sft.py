import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "mixed_agpo" / "mixed_answer" / "train_finalmarker_nothink.jsonl"
DEFAULT_OUT = ROOT / "data" / "mixed_agpo" / "mixed_answer" / "train_letterstrict_finalmarker_nothink.jsonl"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
SEED = 20260621

LETTER_TARGETS = {
    "gaokao_mathqa": 600,
    "sat_math": 360,
    "mmlu_stem": 240,
    "math": 40,
}

NUMERIC_ANCHOR_TARGETS = {
    "gsm8k": 140,
    "cmath": 100,
    "math": 100,
    "ocw_courses": 60,
    "gaokao_mathcloze": 60,
}

LETTER_SET = {"A", "B", "C", "D", "E"}
LETTER_SUFFIX = (
    "\n\nOnly output the final option letter. Do not include reasoning or option content.\n"
    "Format: Final answer: <option letter>"
)
NUMERIC_SUFFIX = "\n\nOnly output the final answer.\nFormat: Final answer: <answer>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("final_answer") or "").strip()


def is_letter_row(row: dict[str, Any]) -> bool:
    return answer(row).upper() in LETTER_SET


def strip_old_answer_instruction(text: str) -> str:
    text = re.sub(
        r"\n?\s*(?:答案|Answer)\s*[:：]\s*从\s*A\s*到\s*[A-E]\s*,?\s*我们应选择\s*",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\n?\s*A\s*:\s*Among\s+A\s+through\s+[A-E],\s*the\s+answer\s+is\s*",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\n?\s*Answer with the correct option letter only\.?\s*",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_item(row: dict[str, Any], output: str, suffix: str, tag: str, index: int) -> dict[str, Any]:
    item = dict(row)
    item["instruction"] = strip_old_answer_instruction(str(item["instruction"])) + suffix
    item["input"] = str(item.get("input") or "")
    item["output"] = output
    metadata = dict(item.get("metadata") or {})
    metadata["letterstrict_source"] = tag
    metadata["letterstrict_repeat_index"] = index
    metadata["letterstrict_output_style"] = "final_answer_only"
    item["metadata"] = metadata
    return item


def sample_with_replacement(
    rng: random.Random,
    rows: list[dict[str, Any]],
    target: int,
    output_builder: Any,
    suffix: str,
    tag: str,
) -> list[dict[str, Any]]:
    if not rows or target <= 0:
        return []
    rows = rows[:]
    rng.shuffle(rows)
    sampled: list[dict[str, Any]] = []
    for index in range(target):
        source = rows[index % len(rows)]
        sampled.append(make_item(source, output_builder(source), suffix, tag, index // len(rows)))
    return sampled


def update_dataset_info(out_path: Path, dataset_key: str) -> None:
    info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    rel = out_path.resolve().relative_to(ROOT / "data").as_posix()
    info[dataset_key] = {
        "file_name": rel,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    DATASET_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strict final-answer SFT set for letter-source repair.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-key", default="mixed_agpo_mixed_answer_train_letterstrict_finalmarker_nothink")
    parser.add_argument("--no-update-dataset-info", action="store_true")
    args = parser.parse_args()

    rng = random.Random(SEED)
    rows = read_jsonl(args.input)
    letter_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        if is_letter_row(row):
            letter_by_source[source].append(row)
        else:
            numeric_by_source[source].append(row)

    out_rows: list[dict[str, Any]] = []
    for source, target in LETTER_TARGETS.items():
        out_rows.extend(
            sample_with_replacement(
                rng,
                letter_by_source[source],
                target,
                lambda row: f"Final answer: {answer(row).upper()}",
                LETTER_SUFFIX,
                f"letter:{source}",
            )
        )
    for source, target in NUMERIC_ANCHOR_TARGETS.items():
        out_rows.extend(
            sample_with_replacement(
                rng,
                numeric_by_source[source],
                target,
                lambda row: f"Final answer: {answer(row)}",
                NUMERIC_SUFFIX,
                f"numeric_anchor:{source}",
            )
        )

    rng.shuffle(out_rows)
    write_jsonl(args.out, out_rows)
    if not args.no_update_dataset_info:
        update_dataset_info(args.out, args.dataset_key)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "out": str(args.out),
                "dataset_key": args.dataset_key,
                "rows_in": len(rows),
                "rows_out": len(out_rows),
                "source_counts": dict(sorted(Counter(str(row.get("source_dataset") or "unknown") for row in out_rows).items())),
                "letter_rows": sum(is_letter_row(row) for row in out_rows),
                "numeric_anchor_rows": sum(not is_letter_row(row) for row in out_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
