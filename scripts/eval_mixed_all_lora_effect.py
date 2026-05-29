import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hybrid_reward_server import HybridRewardScorer  # noqa: E402
from code_reward_server import parse_test_suites  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_samples(rows: list[dict[str, Any]], answer_count: int, code_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    answers = [row for row in rows if row.get("has_answer") and row.get("source_dataset") != "code_contests"]
    code = [row for row in rows if row.get("source_dataset") == "code_contests"]
    rng.shuffle(answers)
    rng.shuffle(code)
    selected = answers[:answer_count] + code[:code_count]
    rng.shuffle(selected)
    return selected


def user_text(row: dict[str, Any]) -> str:
    text = str(row.get("instruction") or "")
    extra = str(row.get("input") or "").strip()
    if extra:
        text = f"{text}\n\n{extra}"
    return text


def format_prompt(tokenizer: AutoTokenizer, row: dict[str, Any], enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_text(row)}]
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


def generate_one(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    adapter_enabled: bool,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    prompt = format_prompt(tokenizer, row, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and LoRA rewards on mixed_all validation samples.")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--adapter", default="saves/qwen3-0.6b/lora/mixed_all_agpo_200")
    parser.add_argument("--data", default="data/mixed_agpo/mixed_all/validation.jsonl")
    parser.add_argument("--code-dataset-dir", default="data/code_contests/hf_dataset")
    parser.add_argument("--out", default="saves/qwen3-0.6b/lora/mixed_all_agpo_200/eval_mixed_all_validation.jsonl")
    parser.add_argument("--answer-count", type=int, default=12)
    parser.add_argument("--code-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--test-suites", default="public,generated")
    parser.add_argument("--max-tests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    rows = select_samples(read_jsonl(Path(args.data)), args.answer_count, args.code_count, args.seed)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    scorer = HybridRewardScorer(
        answer_data=Path(args.data),
        code_dataset_dir=Path(args.code_dataset_dir),
        test_suites=parse_test_suites(args.test_suites),
        max_tests=args.max_tests,
        timeout=args.timeout,
        language="python",
        prefix_chars=300,
    )

    results = []
    with output_path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows, 1):
            base_response = generate_one(
                model, tokenizer, row, False, args.max_new_tokens, enable_thinking=args.enable_thinking
            )
            lora_response = generate_one(
                model, tokenizer, row, True, args.max_new_tokens, enable_thinking=args.enable_thinking
            )
            base_reward = scorer.score_detailed(reward_message(row, base_response))
            lora_reward = scorer.score_detailed(reward_message(row, lora_response))
            item = {
                "index": index,
                "sample_id": row.get("sample_id"),
                "source_dataset": row.get("source_dataset"),
                "instruction": row.get("instruction"),
                "answer": row.get("answer"),
                "base_score": base_reward.score,
                "base_source": base_reward.source,
                "base_detail": base_reward.detail,
                "base_response": base_response,
                "lora_score": lora_reward.score,
                "lora_source": lora_reward.source,
                "lora_detail": lora_reward.detail,
                "lora_response": lora_response,
            }
            results.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[{index}/{len(rows)}] {row.get('source_dataset')} "
                f"base={base_reward.score:.3f} lora={lora_reward.score:.3f}",
                flush=True,
            )

    by_source: dict[str, dict[str, list[float]]] = {}
    for item in results:
        source = str(item["source_dataset"])
        by_source.setdefault(source, {"base": [], "lora": []})
        by_source[source]["base"].append(float(item["base_score"]))
        by_source[source]["lora"].append(float(item["lora_score"]))

    summary = {
        "n": len(results),
        "base_mean": mean([float(item["base_score"]) for item in results]),
        "lora_mean": mean([float(item["lora_score"]) for item in results]),
        "wins": sum(1 for item in results if float(item["lora_score"]) > float(item["base_score"])),
        "ties": sum(1 for item in results if float(item["lora_score"]) == float(item["base_score"])),
        "losses": sum(1 for item in results if float(item["lora_score"]) < float(item["base_score"])),
        "by_source": {
            source: {
                "n": len(values["base"]),
                "base_mean": mean(values["base"]),
                "lora_mean": mean(values["lora"]),
            }
            for source, values in sorted(by_source.items())
        },
        "output_path": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
