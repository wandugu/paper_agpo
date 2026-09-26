import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build answer-only SFT rows from mixed_answer data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="Answer:")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    out_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        answer = row.get("answer") or row.get("final_answer")
        if answer in (None, ""):
            skipped += 1
            continue

        item = dict(row)
        item["output"] = f"{args.prefix} {str(answer).strip()}"
        metadata = dict(item.get("metadata") or {})
        metadata["answer_only_sft_source"] = str(args.input)
        metadata["answer_only_sft_prefix"] = args.prefix
        item["metadata"] = metadata
        out_rows.append(item)

    write_jsonl(args.output, out_rows)

    by_source: dict[str, int] = {}
    by_answer_type = {"letter": 0, "numeric_or_expr": 0}
    for row in out_rows:
        source = str(row.get("source_dataset") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
        answer = str(row.get("answer") or row.get("final_answer") or "").strip()
        if len(answer) == 1 and answer.upper() in {"A", "B", "C", "D", "E"}:
            by_answer_type["letter"] += 1
        else:
            by_answer_type["numeric_or_expr"] += 1

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "written": len(out_rows),
                "skipped": skipped,
                "by_answer_type": by_answer_type,
                "by_source": dict(sorted(by_source.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
