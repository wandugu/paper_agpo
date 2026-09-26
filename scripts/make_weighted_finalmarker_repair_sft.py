import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def clone_row(row: dict[str, Any], source_path: Path, role: str, repeat_index: int) -> dict[str, Any]:
    item = dict(row)
    metadata = dict(item.get("metadata") or {})
    metadata["repair_sft_source"] = str(source_path)
    metadata["repair_sft_role"] = role
    metadata["repair_sft_repeat_index"] = repeat_index
    item["metadata"] = metadata
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a weak-source weighted final-marker SFT repair dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weak-source",
        action="append",
        default=["gaokao_mathqa", "gaokao_mathcloze", "ocw_courses", "cmath"],
    )
    parser.add_argument("--weak-repeat", type=int, default=2)
    parser.add_argument("--anchor-per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    weak_sources = set(args.weak_source)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        if row.get("answer") in (None, ""):
            continue
        by_source[str(row.get("source_dataset") or "unknown")].append(row)

    out_rows: list[dict[str, Any]] = []
    for source, source_rows in sorted(by_source.items()):
        if source in weak_sources:
            for repeat_index in range(args.weak_repeat):
                out_rows.extend(clone_row(row, args.input, "weak", repeat_index) for row in source_rows)
            continue

        anchors = source_rows[:]
        rng.shuffle(anchors)
        for row in anchors[: args.anchor_per_source]:
            out_rows.append(clone_row(row, args.input, "anchor", 0))

    rng.shuffle(out_rows)
    write_jsonl(args.output, out_rows)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "written": len(out_rows),
                "weak_sources": sorted(weak_sources),
                "weak_repeat": args.weak_repeat,
                "anchor_per_source": args.anchor_per_source,
                "by_source": dict(sorted(Counter(row.get("source_dataset") for row in out_rows).items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
