import json
import math
import tempfile
import types

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    DataCollatorWithPadding,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
)

from llamafactory.hparams import FinetuningArguments, GeneratingArguments, ModelArguments
from llamafactory.train.agpo.trainer import CustomAGPOTrainer, load_agpo_config


def _make_tiny_tokenizer():
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<pad>": 1, "x": 2, "y": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<pad>",
        unk_token="<unk>",
    )


def _make_tiny_model():
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=16,
            pad_token_id=1,
            eos_token_id=1,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
    )


def test_load_agpo_config():
    config = load_agpo_config()
    assert config['controller']['ema_alpha'] == 0.1
    assert config['controller']['entropy_h_min'] == 0.5
    assert config['controller']['advantage_floor'] == 1e-8
    assert config['logging']['debug_reward_samples'] is True
    assert config['logging']['controller_trace_jsonl'] is True


def test_compute_centered_uncertainty_and_temperature():
    centered, updated_ema = CustomAGPOTrainer.compute_centered_uncertainty(2.0, 1.0, 0.25)
    assert math.isclose(updated_ema, 1.25)
    assert math.isclose(centered, 1.0, rel_tol=1e-6)

    tau = CustomAGPOTrainer.compute_adaptive_temperature(
        tau_base=1.0,
        lambda_temp=0.2,
        centered_uncertainty=centered,
        tau_min=0.5,
        tau_max=1.5,
    )
    assert math.isclose(tau, 1.2, rel_tol=1e-6)


def test_compute_adaptive_clip_uses_entropy_and_skew():
    eps = CustomAGPOTrainer.compute_adaptive_clip(
        eps_base=0.2,
        eps_min=0.05,
        eps_max=0.4,
        reward_dispersion=0.4,
        abs_skew=0.3,
        policy_entropy=1.5,
        entropy_ref=1.0,
        vote_entropy=0.7,
        step_kl=0.2,
        alpha_var=1.0,
        gamma_stepkl=0.5,
        delta_probe_entropy=0.1,
        zeta_skew=0.2,
        entropy_h_min=0.5,
        entropy_h_max=2.0,
        entropy_floor=1e-8,
    )
    expected = 0.2 * 1.5 * (1 + 0.07) / (1 + 0.4 + 0.1 + 0.06)
    assert math.isclose(eps, expected, rel_tol=1e-6)


def test_vote_key_prefers_extracted_math_answers():
    assert CustomAGPOTrainer._extract_vote_key(r"therefore \boxed{42}") == "42"
    assert CustomAGPOTrainer._extract_vote_key("reasoning #### 42") == "42"
    assert CustomAGPOTrainer._extract_vote_key("final answer is 1,024.") == "1024"


def test_batch_ema_and_controls_are_prompt_order_invariant():
    records = [
        ("a", 0.4, 1.1),
        ("b", 1.2, 0.8),
        ("c", 0.7, 1.4),
    ]
    uncertainty_ema = 0.8
    entropy_ref = 1.0

    def controls(items):
        result = {}
        for key, raw_uncertainty, policy_entropy in items:
            centered, _ = CustomAGPOTrainer.compute_centered_uncertainty(
                raw_uncertainty,
                uncertainty_ema,
                0.1,
            )
            tau = CustomAGPOTrainer.compute_adaptive_temperature(1.0, 0.15, centered, 0.5, 1.5)
            eps = CustomAGPOTrainer.compute_adaptive_clip(
                eps_base=0.2,
                eps_min=0.05,
                eps_max=0.4,
                reward_dispersion=raw_uncertainty,
                abs_skew=0.0,
                policy_entropy=policy_entropy,
                entropy_ref=entropy_ref,
                vote_entropy=0.3,
                step_kl=0.1,
                alpha_var=1.0,
                gamma_stepkl=0.5,
                delta_probe_entropy=0.1,
                zeta_skew=0.0,
                entropy_h_min=0.5,
                entropy_h_max=1.5,
                entropy_floor=1e-8,
            )
            result[key] = (centered, tau, eps)
        return result

    reversed_records = list(reversed(records))
    assert controls(records) == controls(reversed_records)
    assert math.isclose(
        CustomAGPOTrainer.update_ema_from_batch(
            [record[1] for record in records],
            uncertainty_ema,
            0.1,
        ),
        CustomAGPOTrainer.update_ema_from_batch(
            [record[1] for record in reversed_records],
            uncertainty_ema,
            0.1,
        ),
    )
    assert math.isclose(
        CustomAGPOTrainer.update_ema_from_batch(
            [record[2] for record in records],
            entropy_ref,
            0.01,
        ),
        CustomAGPOTrainer.update_ema_from_batch(
            [record[2] for record in reversed_records],
            entropy_ref,
            0.01,
        ),
    )


def test_budget_cap_reserves_probe_and_train_groups_without_overshoot():
    cap = CustomAGPOTrainer.compute_budgeted_max_new_tokens(
        remaining_tokens=100,
        group_size=4,
        world_size=2,
        groups_to_reserve=2,
        configured_max_new_tokens=10,
    )
    assert cap == 6
    assert cap * 4 * 2 * 2 <= 100

    assert (
        CustomAGPOTrainer.compute_budgeted_max_new_tokens(
            remaining_tokens=15,
            group_size=4,
            world_size=2,
            groups_to_reserve=2,
            configured_max_new_tokens=10,
        )
        == 0
    )


def test_plain_policy_end_to_end_budget_smoke():
    tokenizer = _make_tiny_tokenizer()
    model = _make_tiny_model()
    dataset = [
        {"input_ids": torch.tensor([2, 2]), "attention_mask": torch.tensor([1, 1])},
        {"input_ids": torch.tensor([2, 3]), "attention_mask": torch.tensor([1, 1])},
    ]

    with tempfile.TemporaryDirectory() as output_dir:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=4,
            report_to=[],
            save_steps=100,
            logging_steps=1,
            learning_rate=1e-4,
            disable_tqdm=True,
        )
        finetuning_args = FinetuningArguments(
            stage="ppo",
            rl_algo="agpo",
            finetuning_type="full",
            reward_model="http://127.0.0.1:9",
            reward_model_type="api",
            agpo_group_size=2,
            agpo_update_epochs=1,
            agpo_beta_ref_kl=0.0,
            agpo_max_generated_tokens=8,
            agpo_count_probe_tokens_in_budget=True,
        )
        trainer = CustomAGPOTrainer(
            model_args=ModelArguments(model_name_or_path="tiny-local"),
            training_args=training_args,
            finetuning_args=finetuning_args,
            generating_args=GeneratingArguments(max_new_tokens=2),
            callbacks=[],
            model=model,
            reward_model="unused",
            ref_model=None,
            tokenizer=tokenizer,
            processor=None,
            data_collator=DataCollatorWithPadding(tokenizer),
            train_dataset=dataset,
        )

        def fake_rewards(self, queries, responses):
            return [torch.tensor(float(index % 2), device=self.current_device) for index in range(len(responses))]

        def fake_generate_group(self, prompt, temperature, max_new_tokens=None):
            response_length = max_new_tokens or 2
            return [torch.full((response_length,), 2, dtype=torch.long) for _ in range(self.group_size)]

        trainer.get_rewards = types.MethodType(fake_rewards, trainer)
        trainer._generate_group = types.MethodType(fake_generate_group, trainer)
        trainer.ppo_train()

        unwrapped_policy = trainer.accelerator.unwrap_model(trainer.model)
        assert not hasattr(unwrapped_policy, "v_head")
        assert trainer.accelerator.gradient_accumulation_steps == 1
        assert trainer.budget_generated_tokens <= finetuning_args.agpo_max_generated_tokens
        assert trainer.total_generated_tokens == trainer.probe_generated_tokens + trainer.train_generated_tokens
        assert (trainer.state.global_step, trainer.token_budget_exhausted) == (1, True)
        with open(f"{output_dir}/agpo_controller_trace.rank0.jsonl", encoding="utf-8") as trace_file:
            trace_rows = [json.loads(line) for line in trace_file if line.strip()]
        assert len(trace_rows) == 1
        assert trace_rows[0]["prompt_sha256"]
        assert trace_rows[0]["budget_generated_tokens"] == trainer.budget_generated_tokens
        trainer.accelerator.end_training()


def test_peft_reference_kl_uses_base_policy_without_value_head():
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = _make_tiny_tokenizer()
    model = get_peft_model(
        _make_tiny_model(),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
        ),
    )
    dataset = [{"input_ids": torch.tensor([2, 2]), "attention_mask": torch.tensor([1, 1])}]

    with tempfile.TemporaryDirectory() as output_dir:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            max_steps=1,
            report_to=[],
            save_steps=10,
            logging_steps=1,
        )
        finetuning_args = FinetuningArguments(
            stage="ppo",
            rl_algo="agpo",
            finetuning_type="lora",
            reward_model="http://127.0.0.1:9",
            reward_model_type="api",
            agpo_group_size=2,
            agpo_update_epochs=1,
            agpo_beta_ref_kl=0.03,
        )
        trainer = CustomAGPOTrainer(
            model_args=ModelArguments(model_name_or_path="tiny-local"),
            training_args=training_args,
            finetuning_args=finetuning_args,
            generating_args=GeneratingArguments(max_new_tokens=2),
            callbacks=[],
            model=model,
            reward_model="unused",
            ref_model=None,
            tokenizer=tokenizer,
            processor=None,
            data_collator=DataCollatorWithPadding(tokenizer),
            train_dataset=dataset,
        )
        input_ids = torch.tensor([[2, 2, 3], [2, 2, 2]], device=trainer.current_device)
        attention_mask = torch.ones_like(input_ids)
        response_mask = torch.tensor(
            [[0, 0, 1], [0, 0, 1]],
            dtype=torch.float32,
            device=trainer.current_device,
        )
        token_logprobs, mask, _ = trainer._sequence_logprobs(
            trainer.model,
            input_ids,
            attention_mask,
            response_mask,
            requires_grad=True,
        )
        kl_ref = trainer._compute_ref_kl(
            input_ids,
            attention_mask,
            mask,
            token_logprobs,
        )

        assert not hasattr(trainer.accelerator.unwrap_model(trainer.model), "v_head")
        assert math.isclose(float(kl_ref.detach()), 0.0, abs_tol=1e-6)
        trainer.accelerator.end_training()
