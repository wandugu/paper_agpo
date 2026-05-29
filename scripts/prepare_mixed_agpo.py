import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "mixed_agpo"
DATASET_INFO = DATA_DIR / "dataset_info.json"
SEED = "mixed-agpo-v1"


@dataclass(frozen=True)
class FileRef:
    split: str
    path: Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    train: tuple[FileRef, ...] = ()
    validation: tuple[FileRef, ...] = ()
    test: tuple[FileRef, ...] = ()
    train_pool: tuple[FileRef, ...] = ()
    pool: tuple[FileRef, ...] = ()
    validation_ratio: float = 0.05


SPECS = (
    DatasetSpec(
        name="gsm8k",
        display_name="GSM8K",
        train=(FileRef("rl_train", DATA_DIR / "gsm8k" / "rl_train.json"),),
        test=(FileRef("rl_test", DATA_DIR / "gsm8k" / "rl_test.json"),),
    ),
    DatasetSpec(
        name="math",
        display_name="MATH",
        train=(FileRef("train", DATA_DIR / "math" / "train.json"),),
        test=(FileRef("test", DATA_DIR / "math" / "test.json"),),
    ),
    DatasetSpec(
        name="mmlu_stem",
        display_name="MMLU_STEM",
        train=(FileRef("dev", DATA_DIR / "mmlu_stem" / "dev.json"),),
        validation=(FileRef("val", DATA_DIR / "mmlu_stem" / "val.json"),),
        test=(FileRef("test", DATA_DIR / "mmlu_stem" / "test.json"),),
    ),
    DatasetSpec(
        name="cmath",
        display_name="CMATH",
        train_pool=(FileRef("validation", DATA_DIR / "cmath" / "validation.json"),),
        test=(FileRef("test", DATA_DIR / "cmath" / "test.json"),),
        validation_ratio=0.10,
    ),
    DatasetSpec(
        name="sat_math",
        display_name="SAT-Math",
        pool=(FileRef("test", DATA_DIR / "agieval" / "sat_math_test.json"),),
    ),
    DatasetSpec(
        name="gaokao_mathqa",
        display_name="GaokaoMath-QA",
        pool=(FileRef("test", DATA_DIR / "agieval" / "gaokao_mathqa_test.json"),),
    ),
    DatasetSpec(
        name="gaokao_mathcloze",
        display_name="GaokaoMath-Cloze",
        pool=(FileRef("test", DATA_DIR / "agieval" / "gaokao_mathcloze_test.json"),),
    ),
    DatasetSpec(
        name="ocw_courses",
        display_name="OCW-Courses",
        pool=(FileRef("test", DATA_DIR / "ocw_courses" / "test.json"),),
    ),
    DatasetSpec(
        name="code_contests",
        display_name="CodeContests",
        train=(FileRef("rl_train", DATA_DIR / "code_contests" / "rl_train.jsonl"),),
        validation=(FileRef("rl_valid", DATA_DIR / "code_contests" / "rl_valid.jsonl"),),
        test=(FileRef("rl_test", DATA_DIR / "code_contests" / "rl_test.jsonl"),),
    ),
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}")
    return rows


def stable_key(*parts: str) -> str:
    text = "\n".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_instruction(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("instruction", "")).split())


def shuffle_groups(rows: list[dict[str, Any]], dataset_name: str) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(normalized_instruction(row), []).append(row)

    grouped = list(groups.values())
    grouped.sort(key=lambda items: stable_key(SEED, dataset_name, normalized_instruction(items[0])))
    return grouped


def split_grouped(
    rows: list[dict[str, Any]],
    dataset_name: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"train": [], "validation": [], "test": []}

    total = len(rows)
    validation_target = max(1, round(total * validation_ratio)) if validation_ratio else 0
    test_target = max(1, round(total * test_ratio)) if test_ratio else 0
    train_target = max(0, total - validation_target - test_target)

    splits = {"train": [], "validation": [], "test": []}
    targets = {"train": train_target, "validation": validation_target, "test": test_target}
    order = ("train", "validation", "test")

    for group in shuffle_groups(rows, dataset_name):
        split = min(order, key=lambda name: len(splits[name]) / max(1, targets[name]))
        splits[split].extend(group)

    return splits


def split_train_validation(rows: list[dict[str, Any]], dataset_name: str, validation_ratio: float) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"train": [], "validation": []}

    validation_target = max(1, round(len(rows) * validation_ratio))
    splits = {"train": [], "validation": []}
    for group in shuffle_groups(rows, dataset_name):
        split = "validation" if len(splits["validation"]) < validation_target else "train"
        splits[split].extend(group)
    return splits


def collect(refs: tuple[FileRef, ...], spec: DatasetSpec) -> list[dict[str, Any]]:
    rows = []
    for ref in refs:
        if not ref.path.exists():
            raise FileNotFoundError(ref.path)

        for index, row in enumerate(load_rows(ref.path)):
            copied = dict(row)
            copied["_source_file"] = ref.path.relative_to(ROOT).as_posix()
            copied["_source_split"] = ref.split
            copied["_source_index"] = index
            copied["_dataset_name"] = spec.name
            copied["_dataset_display_name"] = spec.display_name
            rows.append(copied)
    return rows


def make_splits(spec: DatasetSpec) -> dict[str, list[dict[str, Any]]]:
    if spec.pool:
        return split_grouped(collect(spec.pool, spec), spec.name, train_ratio=0.80, validation_ratio=0.10, test_ratio=0.10)

    splits = {
        "train": collect(spec.train, spec),
        "validation": collect(spec.validation, spec),
        "test": collect(spec.test, spec),
    }

    if spec.train_pool:
        pool_splits = split_train_validation(collect(spec.train_pool, spec), spec.name, spec.validation_ratio)
        splits["train"].extend(pool_splits["train"])
        splits["validation"].extend(pool_splits["validation"])

    if splits["train"] and not splits["validation"]:
        train_validation = split_train_validation(splits["train"], spec.name, spec.validation_ratio)
        splits["train"] = train_validation["train"]
        splits["validation"] = train_validation["validation"]

    return splits


def normalize_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    answer = row.get("answer") or row.get("final_answer") or ""
    reference_output = row.get("output") or ""

    ignored = {
        "instruction",
        "input",
        "output",
        "answer",
        "final_answer",
        "source",
        "split",
    }
    metadata = {key: value for key, value in row.items() if key not in ignored and not key.startswith("_")}

    return {
        "instruction": str(row.get("instruction", "")).strip(),
        "input": str(row.get("input", "")),
        "output": "",
        "answer": str(answer),
        "reference_output": str(reference_output),
        "source_dataset": row["_dataset_name"],
        "source_dataset_name": row["_dataset_display_name"],
        "source": row.get("source", row["_dataset_name"]),
        "mixed_split": split,
        "original_split": row.get("split", row["_source_split"]),
        "original_file": row["_source_file"],
        "original_index": row["_source_index"],
        "sample_id": stable_key(row["_dataset_name"], row["_source_file"], str(row["_source_index"])),
        "has_answer": bool(str(answer).strip()),
        "metadata": metadata,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(manifest: dict[str, Any]) -> None:
    lines = [
        "# Mixed AGPO datasets",
        "",
        "Generated by `scripts/prepare_mixed_agpo.py`.",
        "",
        "- `mixed_all/`: all 9 datasets, including CodeContests.",
        "- `mixed_answer/`: answer-bearing datasets only; CodeContests is excluded because it needs execution-based reward.",
        "- Per-dataset folders contain the same normalized `train.jsonl`, `validation.jsonl`, and `test.jsonl` files.",
        "",
        "All rows use Alpaca-style columns: `instruction`, `input`, `output`. For AGPO, `output` is intentionally empty and the gold target is stored in `answer`.",
        "",
        "## Counts",
        "",
        "| dataset | train | validation | test | answer-bearing |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, counts in manifest["datasets"].items():
        lines.append(
            f"| {name} | {counts['train']} | {counts['validation']} | {counts['test']} | {counts['has_answer']} |"
        )

    lines.extend(
        [
            f"| mixed_all | {manifest['mixed_all']['train']} | {manifest['mixed_all']['validation']} | {manifest['mixed_all']['test']} | {manifest['mixed_all']['has_answer']} |",
            f"| mixed_answer | {manifest['mixed_answer']['train']} | {manifest['mixed_answer']['validation']} | {manifest['mixed_answer']['test']} | {manifest['mixed_answer']['has_answer']} |",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


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
    original_text = DATASET_INFO.read_text(encoding="utf-8")
    line_ending = "\n"
    info = json.loads(original_text)
    entries: dict[str, dict[str, Any]] = {}

    for spec in SPECS:
        for split in ("train", "validation", "test"):
            entries[f"mixed_agpo_{spec.name}_{split}"] = dataset_info_entry(
                f"mixed_agpo/{spec.name}/{split}.jsonl"
            )

    for name in ("mixed_all", "mixed_answer"):
        for split in ("train", "validation", "test"):
            entries[f"mixed_agpo_{name}_{split}"] = dataset_info_entry(
                f"mixed_agpo/{name}/{split}.jsonl"
            )

    info.update(entries)
    text = json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    with DATASET_INFO.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", line_ending))


def build_dataset() -> dict[str, Any]:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    manifest: dict[str, Any] = {
        "version": 1,
        "seed": SEED,
        "datasets": {},
        "mixed_all": {},
        "mixed_answer": {},
    }
    mixed_all = {"train": [], "validation": [], "test": []}
    mixed_answer = {"train": [], "validation": [], "test": []}

    for spec in SPECS:
        split_rows = make_splits(spec)
        normalized = {
            split: [normalize_row(row, split) for row in rows]
            for split, rows in split_rows.items()
        }
        for split in ("train", "validation", "test"):
            rows = normalized.get(split, [])
            rows.sort(key=lambda row: stable_key(SEED, row["source_dataset"], split, row["sample_id"]))
            write_jsonl(OUT_DIR / spec.name / f"{split}.jsonl", rows)
            mixed_all[split].extend(rows)
            mixed_answer[split].extend(row for row in rows if row["has_answer"])

        all_rows = [row for rows in normalized.values() for row in rows]
        manifest["datasets"][spec.name] = {
            "display_name": spec.display_name,
            "train": len(normalized.get("train", [])),
            "validation": len(normalized.get("validation", [])),
            "test": len(normalized.get("test", [])),
            "has_answer": sum(1 for row in all_rows if row["has_answer"]),
        }

    for mixed_name, splits in (("mixed_all", mixed_all), ("mixed_answer", mixed_answer)):
        for split, rows in splits.items():
            rows.sort(key=lambda row: stable_key(SEED, mixed_name, split, row["source_dataset"], row["sample_id"]))
            write_jsonl(OUT_DIR / mixed_name / f"{split}.jsonl", rows)
        all_rows = [row for rows in splits.values() for row in rows]
        manifest[mixed_name] = {
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
            "has_answer": sum(1 for row in all_rows if row["has_answer"]),
        }

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized mixed AGPO datasets.")
    parser.add_argument("--no-update-dataset-info", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_dataset()
    if not args.no_update_dataset_info:
        update_dataset_info()

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
