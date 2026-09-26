import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "mixed_agpo" / "mixed_answer" / "train.jsonl"
DEFAULT_OUT = ROOT / "data" / "mixed_agpo" / "mixed_answer" / "train_letterbalanced.jsonl"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
SEED = 20260612


LETTER_TARGETS = {
    "mmlu_stem": 1000,
    "gaokao_mathqa": 800,
    "sat_math": 500,
    "math": 100,
}

NUMERIC_TARGETS = {
    "gsm8k": 1200,
    "math": 1200,
    "cmath": 450,
    "ocw_courses": 450,
    "gaokao_mathcloze": 300,
}

UNIQUE_NUMERIC_TARGETS = {
    "gsm8k": 450,
    "math": 450,
    "cmath": 200,
    "ocw_courses": 120,
    "gaokao_mathcloze": 100,
}


def is_letter_answer(row: dict[str, Any]) -> bool:
    return str(row.get("answer", "")).strip().upper() in {"A", "B", "C", "D", "E"}


def read_jsonl_lossy(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def sample_with_replacement(
    rng: random.Random,
    rows: list[dict[str, Any]],
    target: int,
    tag: str,
) -> list[dict[str, Any]]:
    if not rows or target <= 0:
        return []

    rows = rows[:]
    rng.shuffle(rows)
    picked: list[dict[str, Any]] = []
    for index in range(target):
        source = rows[index % len(rows)]
        copied = dict(source)
        metadata = dict(copied.get("metadata") or {})
        metadata["letterbalanced_source"] = tag
        metadata["letterbalanced_repeat_index"] = index // len(rows)
        copied["metadata"] = metadata
        picked.append(copied)
    return picked


def sample_without_replacement(
    rng: random.Random,
    rows: list[dict[str, Any]],
    target: int,
    tag: str,
) -> list[dict[str, Any]]:
    if not rows or target <= 0:
        return []

    rows = rows[:]
    rng.shuffle(rows)
    picked: list[dict[str, Any]] = []
    for index, source in enumerate(rows[:target]):
        copied = dict(source)
        metadata = dict(copied.get("metadata") or {})
        metadata["letterbalanced_source"] = tag
        metadata["letterbalanced_repeat_index"] = 0
        copied["metadata"] = metadata
        picked.append(copied)
    return picked


def update_dataset_info(out_path: Path, dataset_key: str) -> None:
    info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    out_path = out_path.resolve()
    rel = out_path.relative_to(ROOT / "data").as_posix()
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
    parser = argparse.ArgumentParser(description="Build a letter-balanced mixed_answer train split for AGPO.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-key", default="mixed_agpo_mixed_answer_train_letterbalanced")
    parser.add_argument(
        "--profile",
        choices=("oversampled", "unique"),
        default="oversampled",
        help="oversampled repeats small sources; unique uses each letter row once and samples numeric rows without repeats.",
    )
    parser.add_argument("--no-update-dataset-info", action="store_true")
    args = parser.parse_args()

    rng = random.Random(SEED)
    rows, bad = read_jsonl_lossy(args.input)

    letter_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        if is_letter_answer(row):
            letter_by_source[source].append(row)
        else:
            numeric_by_source[source].append(row)

    balanced: list[dict[str, Any]] = []
    if args.profile == "unique":
        for source, rows_for_source in sorted(letter_by_source.items()):
            balanced.extend(
                sample_without_replacement(
                    rng,
                    rows_for_source,
                    len(rows_for_source),
                    f"letter_unique:{source}",
                )
            )
        for source, target in UNIQUE_NUMERIC_TARGETS.items():
            balanced.extend(
                sample_without_replacement(
                    rng,
                    numeric_by_source[source],
                    target,
                    f"numeric_unique:{source}",
                )
            )
    else:
        for source, target in LETTER_TARGETS.items():
            balanced.extend(sample_with_replacement(rng, letter_by_source[source], target, f"letter:{source}"))
        for source, target in NUMERIC_TARGETS.items():
            balanced.extend(sample_with_replacement(rng, numeric_by_source[source], target, f"numeric:{source}"))

    rng.shuffle(balanced)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in balanced:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not args.no_update_dataset_info:
        update_dataset_info(args.out, args.dataset_key)

    counts: dict[str, int] = defaultdict(int)
    letter_count = 0
    for row in balanced:
        source = str(row.get("source_dataset") or "unknown")
        counts[source] += 1
        letter_count += int(is_letter_answer(row))

    print(
        json.dumps(
            {
                "input": str(args.input),
                "out": str(args.out),
                "dataset_key": args.dataset_key,
                "rows_in": len(rows),
                "bad_input_lines_skipped": bad,
                "rows_out": len(balanced),
                "letter_rows_out": letter_count,
                "numeric_rows_out": len(balanced) - letter_count,
                "source_counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
