import json
import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "code_contests"
HF_DATASET_DIR = OUT_DIR / "hf_dataset"

LANG_PRIORITY = ["PYTHON3", "PYTHON", "CPP", "JAVA"]


def setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=OUT_DIR / "download.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(console)
    logging.getLogger().setLevel(logging.INFO)


def label_name(feature, value):
    if value is None:
        return ""
    try:
        return feature.int2str(int(value))
    except Exception:
        return str(value)


def choose_solution(row: dict, language_feature) -> tuple[str, str]:
    languages = row["solutions"]["language"]
    solutions = row["solutions"]["solution"]
    pairs = [(label_name(language_feature, lang), sol) for lang, sol in zip(languages, solutions)]
    for preferred in LANG_PRIORITY:
        for lang, sol in pairs:
            if lang == preferred and sol:
                return lang, sol
    for lang, sol in pairs:
        if sol:
            return lang, sol
    return "", ""


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            count += 1
    return count


def main() -> None:
    setup_logging()
    logging.info("Loading deepmind/code_contests")
    dataset: DatasetDict = load_dataset("deepmind/code_contests")
    logging.info("Loaded splits: %s", {split: len(dataset[split]) for split in dataset.keys()})

    logging.info("Saving full Hugging Face dataset to %s", HF_DATASET_DIR)
    dataset.save_to_disk(str(HF_DATASET_DIR))

    source_feature = dataset["train"].features["source"]
    difficulty_feature = dataset["train"].features["difficulty"]
    language_feature = dataset["train"].features["solutions"]["language"].feature

    for split, split_ds in dataset.items():
        logging.info("Exporting split %s (%d rows)", split, len(split_ds))

        def sft_rows():
            for idx, row in enumerate(split_ds):
                lang, solution = choose_solution(row, language_feature)
                yield {
                    "instruction": row["description"],
                    "input": "",
                    "output": solution,
                    "problem_id": f"{split}:{idx}",
                    "name": row["name"],
                    "solution_language": lang,
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

        def rl_rows():
            for idx, row in enumerate(split_ds):
                yield {
                    "instruction": row["description"],
                    "input": "",
                    "output": "",
                    "problem_id": f"{split}:{idx}",
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

        sft_count = write_jsonl(OUT_DIR / f"sft_{split}.jsonl", sft_rows())
        rl_count = write_jsonl(OUT_DIR / f"rl_{split}.jsonl", rl_rows())
        logging.info("Wrote sft_%s.jsonl=%d and rl_%s.jsonl=%d", split, sft_count, split, rl_count)

    logging.info("Done")


if __name__ == "__main__":
    main()
