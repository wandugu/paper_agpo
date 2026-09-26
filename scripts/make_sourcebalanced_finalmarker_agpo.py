import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "mixed_agpo" / "mixed_answer" / "train_finalmarker_nothink.jsonl"
DEFAULT_OUT = ROOT / "data" / "mixed_agpo" / "mixed_answer" / "train_sourcebalanced_finalmarker_nothink.jsonl"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
SEED = 20260623


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_key(row: dict[str, Any]) -> str:
    return str(row.get("source_dataset") or row.get("source") or "unknown")


def sample_source_rows(
    rng: random.Random,
    rows: list[dict[str, Any]],
    target: int,
    source: str,
) -> list[dict[str, Any]]:
    rows = rows[:]
    rng.shuffle(rows)

    picked: list[dict[str, Any]] = []
    for index in range(target):
        source_row = rows[index % len(rows)]
        copied = dict(source_row)
        metadata = dict(copied.get("metadata") or {})
        metadata["sourcebalanced_source"] = source
        metadata["sourcebalanced_repeat_index"] = index // len(rows)
        copied["metadata"] = metadata
        picked.append(copied)
    return picked


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
    parser = argparse.ArgumentParser(description="Build a source-balanced final-marker no-think AGPO train split.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-key", default="mixed_agpo_mixed_answer_train_sourcebalanced_finalmarker_nothink")
    parser.add_argument("--target-per-source", type=int, default=240)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-update-dataset-info", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        by_source[source_key(row)].append(row)

    balanced: list[dict[str, Any]] = []
    for source, rows in sorted(by_source.items()):
        balanced.extend(sample_source_rows(rng, rows, args.target_per_source, source))

    rng.shuffle(balanced)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in balanced:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not args.no_update_dataset_info:
        update_dataset_info(args.out, args.dataset_key)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "out": str(args.out),
                "dataset_key": args.dataset_key,
                "target_per_source": args.target_per_source,
                "rows_out": len(balanced),
                "source_counts": {source: len(rows) for source, rows in sorted(by_source.items())},
                "balanced_source_counts": {
                    source: args.target_per_source for source in sorted(by_source)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
