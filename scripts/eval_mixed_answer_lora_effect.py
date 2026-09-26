import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from answer_reward_server import AnswerScorer  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Adapters must use LABEL=PATH, for example rewardfix_100=saves/...")

    label, path = value.split("=", 1)
    label = label.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"Adapter label must be simple ASCII, got {label!r}")

    adapter_path = Path(path)
    if not adapter_path.exists():
        raise argparse.ArgumentTypeError(f"Adapter path does not exist: {adapter_path}")

    return label, adapter_path


def user_text(row: dict[str, Any]) -> str:
    text = str(row.get("instruction") or "")
    extra = str(row.get("input") or "").strip()
    if extra:
        text = f"{text}\n\n{extra}"
    return text


def is_letter_answer(row: dict[str, Any]) -> bool:
    answer = str(row.get("answer") or row.get("final_answer") or "").strip()
    return bool(re.fullmatch(r"[A-E]", answer, re.IGNORECASE))


def row_instruction_suffix(row: dict[str, Any], args: argparse.Namespace) -> str:
    suffixes = []
    if is_letter_answer(row) and args.letter_answer_instruction:
        suffixes.append(args.letter_answer_instruction.strip())
    elif not is_letter_answer(row) and args.numeric_answer_instruction:
        suffixes.append(args.numeric_answer_instruction.strip())

    if args.generation_instruction_suffix:
        suffixes.append(args.generation_instruction_suffix.strip())

    return "\n".join(suffix for suffix in suffixes if suffix)


def format_prompt(tokenizer: AutoTokenizer, row: dict[str, Any], enable_thinking: bool, instruction_suffix: str = "") -> str:
    content = user_text(row)
    if instruction_suffix:
        content = f"{content.rstrip()}\n\n{instruction_suffix.strip()}"

    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def reward_message(row: dict[str, Any], response: str) -> str:
    return f"<|im_start|>user\n{user_text(row)}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


def select_balanced_samples(
    rows: list[dict[str, Any]],
    samples_per_source: int,
    max_total: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("has_answer") and row.get("answer") not in (None, ""):
            by_source[str(row.get("source_dataset") or "unknown")].append(row)

    selected = []
    for source in sorted(by_source):
        source_rows = by_source[source][:]
        rng.shuffle(source_rows)
        selected.extend(source_rows[:samples_per_source])

    rng.shuffle(selected)
    if max_total is not None:
        selected = selected[:max_total]

    return selected


def result_key(item: dict[str, Any]) -> tuple[int | None, str | None]:
    index = item.get("index")
    return (int(index) if isinstance(index, int) else None, item.get("sample_id"))


def score_detail(scorer: AnswerScorer, row: dict[str, Any], response: str) -> dict[str, Any]:
    message = reward_message(row, response)
    question = scorer._extract_question(message)
    gold = None
    prediction = None
    if question is not None and question in scorer.answers:
        gold = scorer._norm_answer(scorer.answers[question])
        prediction = scorer._extract_answer(
            scorer._extract_response(message),
            expect_letter=scorer._canonical_letter_answer(gold) is not None,
            question=question,
        )

    matched = bool(prediction is not None and gold is not None and scorer._answers_match(question, gold, prediction))
    option_prediction = None
    if prediction is not None and gold is not None and scorer._canonical_letter_answer(gold) is not None:
        option_prediction = scorer._option_letter_for_prediction(question, prediction)

    score = 1.0 if matched else 0.0
    return {
        "score": score,
        "dense_score": float(scorer.score(message)),
        "gold": gold,
        "prediction": prediction,
        "option_prediction": option_prediction,
        "matched": matched,
    }


def generation_kwargs(args: argparse.Namespace, tokenizer: AutoTokenizer) -> dict[str, Any]:
    eos_ids = []
    for token in (tokenizer.eos_token, "<|im_end|>"):
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
            eos_ids.append(token_id)

    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": eos_ids or tokenizer.eos_token_id,
    }
    if args.do_sample:
        kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        kwargs.update({"do_sample": False})
    if args.max_generation_seconds is not None:
        kwargs["max_time"] = args.max_generation_seconds

    return kwargs


def generate_one(
    model: Any,
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    label: str,
    args: argparse.Namespace,
    gen_kwargs: dict[str, Any],
) -> str:
    prompt = format_prompt(
        tokenizer,
        row,
        enable_thinking=args.enable_thinking,
        instruction_suffix=row_instruction_suffix(row, args),
    )
    assistant_prefix = args.assistant_prefix
    if assistant_prefix.strip():
        prompt = f"{prompt}{assistant_prefix}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(model.device)

    if isinstance(model, PeftModel) and label != "base":
        model.set_adapter(label)
        context = nullcontext()
    elif isinstance(model, PeftModel):
        context = model.disable_adapter()
    else:
        context = nullcontext()

    with context, torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)

    generated = output[0, inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if assistant_prefix.strip():
        return f"{assistant_prefix}{response}".strip()
    return response


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[dict[str, Any]], model_labels: list[str], compare_from: str, compare_to: str) -> dict[str, Any]:
    by_model = {}
    by_model_dense = {}
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_source_dense: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in results:
        source = str(item["source_dataset"])
        for label in model_labels:
            score = float(item["outputs"][label]["score"])
            dense_score = float(item["outputs"][label].get("dense_score", score))
            by_source[source][label].append(score)
            by_source_dense[source][label].append(dense_score)

    for label in model_labels:
        scores = [float(item["outputs"][label]["score"]) for item in results]
        dense_scores = [float(item["outputs"][label].get("dense_score", item["outputs"][label]["score"])) for item in results]
        by_model[label] = {
            "mean": mean(scores),
            "sum": sum(scores),
            "nonzero": sum(score > 0 for score in scores),
            "score_counts": dict(sorted(Counter(scores).items())),
        }
        by_model_dense[label] = {
            "mean": mean(dense_scores),
            "sum": sum(dense_scores),
            "nonzero": sum(score > 0 for score in dense_scores),
            "score_counts": dict(sorted(Counter(dense_scores).items())),
        }

    comparison = None
    comparison_dense = None
    if compare_from in model_labels and compare_to in model_labels:
        from_scores = [float(item["outputs"][compare_from]["score"]) for item in results]
        to_scores = [float(item["outputs"][compare_to]["score"]) for item in results]
        from_dense_scores = [
            float(item["outputs"][compare_from].get("dense_score", item["outputs"][compare_from]["score"]))
            for item in results
        ]
        to_dense_scores = [
            float(item["outputs"][compare_to].get("dense_score", item["outputs"][compare_to]["score"]))
            for item in results
        ]
        comparison = {
            "from": compare_from,
            "to": compare_to,
            "delta_mean": mean(to_scores) - mean(from_scores),
            "wins": sum(to > old for old, to in zip(from_scores, to_scores)),
            "ties": sum(to == old for old, to in zip(from_scores, to_scores)),
            "losses": sum(to < old for old, to in zip(from_scores, to_scores)),
        }
        comparison_dense = {
            "from": compare_from,
            "to": compare_to,
            "delta_mean": mean(to_dense_scores) - mean(from_dense_scores),
            "wins": sum(to > old for old, to in zip(from_dense_scores, to_dense_scores)),
            "ties": sum(to == old for old, to in zip(from_dense_scores, to_dense_scores)),
            "losses": sum(to < old for old, to in zip(from_dense_scores, to_dense_scores)),
        }

    return {
        "n": len(results),
        "models": model_labels,
        "by_model": by_model,
        "by_model_dense": by_model_dense,
        "by_source": {
            source: {
                label: {"n": len(scores), "mean": mean(scores), "sum": sum(scores)}
                for label, scores in sorted(model_scores.items())
            }
            for source, model_scores in sorted(by_source.items())
        },
        "by_source_dense": {
            source: {
                label: {"n": len(scores), "mean": mean(scores), "sum": sum(scores)}
                for label, scores in sorted(model_scores.items())
            }
            for source, model_scores in sorted(by_source_dense.items())
        },
        "comparison": comparison,
        "comparison_dense": comparison_dense,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer-only LoRA adapters on a held-out mixed_answer split.")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--data", default="data/mixed_agpo/mixed_answer/validation.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--samples-per-source", type=int, default=4)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-prompt-chars", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-generation-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-instruction-suffix", default="")
    parser.add_argument("--letter-answer-instruction", default="")
    parser.add_argument("--numeric-answer-instruction", default="")
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--compare-from", default=None)
    parser.add_argument("--compare-to", default=None)
    parser.add_argument("--resume", action="store_true", help="Append to an existing JSONL and skip completed samples.")
    parser.add_argument("--summarize-existing", action="store_true", help="Only summarize an existing JSONL output file.")
    args = parser.parse_args()

    rows = select_balanced_samples(
        read_jsonl(Path(args.data)),
        samples_per_source=args.samples_per_source,
        max_total=args.max_total,
        seed=args.seed,
    )
    if not rows:
        raise SystemExit("No held-out samples selected.")
    if args.max_prompt_chars is not None:
        rows = [row for row in rows if len(user_text(row)) <= args.max_prompt_chars]
        if not rows:
            raise SystemExit(f"No held-out samples remain after --max-prompt-chars={args.max_prompt_chars}.")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_items: list[tuple[str, Path]] = list(args.adapter)
    model_labels = []
    if not args.no_base:
        model_labels.append("base")
    model_labels.extend(label for label, _ in adapter_items)

    compare_from = args.compare_from or (model_labels[-2] if len(model_labels) >= 2 else model_labels[0])
    compare_to = args.compare_to or model_labels[-1]

    if args.summarize_existing:
        if not output_path.exists():
            raise SystemExit(f"Cannot summarize missing output file: {output_path}")
        existing_results = read_jsonl(output_path)
        if not existing_results:
            raise SystemExit(f"Cannot summarize empty output file: {output_path}")
        summary = summarize(existing_results, model_labels, compare_from=compare_from, compare_to=compare_to)
        summary.update(
            {
                "data": args.data,
                "output_path": str(output_path),
                "seed": args.seed,
                "samples_per_source": args.samples_per_source,
                "max_total": args.max_total,
                "max_prompt_chars": args.max_prompt_chars,
                "max_new_tokens": args.max_new_tokens,
                "max_generation_seconds": args.max_generation_seconds,
                "enable_thinking": args.enable_thinking,
                "generation_instruction_suffix": args.generation_instruction_suffix,
                "letter_answer_instruction": args.letter_answer_instruction,
                "numeric_answer_instruction": args.numeric_answer_instruction,
                "assistant_prefix": args.assistant_prefix,
                "do_sample": args.do_sample,
                "partial": True,
            }
        )
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved summary to {summary_path}", flush=True)
        return

    results: list[dict[str, Any]] = []
    completed_keys: set[tuple[int | None, str | None]] = set()
    completed_sample_ids: set[str] = set()
    if args.resume and output_path.exists():
        results = read_jsonl(output_path)
        completed_keys = {result_key(item) for item in results}
        completed_sample_ids = {
            str(item["sample_id"])
            for item in results
            if item.get("sample_id") not in (None, "")
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)

    if adapter_items:
        first_label, first_path = adapter_items[0]
        model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first_label)
        for label, adapter_path in adapter_items[1:]:
            model.load_adapter(adapter_path, adapter_name=label)
    else:
        model = base_model

    model.eval()
    scorer = AnswerScorer(Path(args.data))
    gen_kwargs = generation_kwargs(args, tokenizer)

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as f:
        for index, row in enumerate(rows, 1):
            sample_id = row.get("sample_id")
            key = (index, sample_id)
            if args.resume and (key in completed_keys or (sample_id not in (None, "") and str(sample_id) in completed_sample_ids)):
                print(f"[{index}/{len(rows)}] skip completed {row.get('source_dataset')} sample_id={sample_id}", flush=True)
                continue

            item = {
                "index": index,
                "sample_id": sample_id,
                "source_dataset": row.get("source_dataset"),
                "answer": row.get("answer"),
                "instruction": row.get("instruction"),
                "outputs": {},
            }
            for label in model_labels:
                response = generate_one(model, tokenizer, row, label, args, gen_kwargs)
                detail = score_detail(scorer, row, response)
                item["outputs"][label] = {"response": response, **detail}
                print(
                    f"[{index}/{len(rows)}] {row.get('source_dataset')} {label}={detail['score']:.0f} "
                    f"dense={detail['dense_score']:.3g} "
                    f"pred={detail['prediction']!r} gold={detail['gold']!r}",
                    flush=True,
                )

            results.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()

    summary = summarize(results, model_labels, compare_from=compare_from, compare_to=compare_to)
    summary.update(
        {
            "data": args.data,
            "output_path": str(output_path),
            "seed": args.seed,
            "samples_per_source": args.samples_per_source,
            "max_total": args.max_total,
            "max_prompt_chars": args.max_prompt_chars,
            "max_new_tokens": args.max_new_tokens,
            "max_generation_seconds": args.max_generation_seconds,
            "enable_thinking": args.enable_thinking,
            "generation_instruction_suffix": args.generation_instruction_suffix,
            "letter_answer_instruction": args.letter_answer_instruction,
            "numeric_answer_instruction": args.numeric_answer_instruction,
            "assistant_prefix": args.assistant_prefix,
            "do_sample": args.do_sample,
        }
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved results to {output_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
