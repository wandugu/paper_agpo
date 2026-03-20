# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This implementation extends the PPO trainer to support AGPO training.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from accelerate.utils import DistributedDataParallelKwargs
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override
from yaml import safe_load

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)
_AGPO_CONFIG_PATH = Path(__file__).with_name('config.yaml')


def load_agpo_config() -> dict[str, Any]:
    with _AGPO_CONFIG_PATH.open('r', encoding='utf-8') as f:
        config = safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f'Invalid AGPO config format in {_AGPO_CONFIG_PATH}.')
    return config


class CustomAGPOTrainer(PPOTrainer, Trainer):
    r"""AGPO trainer built upon PPOTrainer utilities."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("AGPO trainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=1,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("AGPO trainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(len(train_dataset) / total_train_batch_size)

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()
        self.agpo_runtime_config = load_agpo_config()
        self.group_size = finetuning_args.agpo_group_size
        self.beta_ref_kl = finetuning_args.agpo_beta_ref_kl
        self.uncertainty_ema = float(self.agpo_runtime_config.get('controller', {}).get('uncertainty_ema_init', 0.0))
        self.step_kl_ema = float(self.agpo_runtime_config.get('controller', {}).get('step_kl_ema_init', 0.0))
        self.clip_entropy_floor = float(self.agpo_runtime_config.get('controller', {}).get('clip_entropy_floor', 1e-8))
        self.debug_reward_samples = bool(self.agpo_runtime_config.get('logging', {}).get('debug_reward_samples', True))
        self.max_debug_reward_items = int(self.agpo_runtime_config.get('logging', {}).get('max_debug_reward_items', 3))
        self.old_policy = self._clone_model()

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )

        self.amp_context = torch.autocast(self.current_device.type)

        self.add_callback(FixValueHeadModelCallback)
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        self.tokenizer = tokenizer
        self.processor = processor

        logger.debug(
            'Loaded AGPO runtime config from %s: %s',
            _AGPO_CONFIG_PATH,
            self.agpo_runtime_config,
        )

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _clone_model(self) -> "AutoModelForCausalLMWithValueHead":
        unwrapped = self.accelerator.unwrap_model(self.model)
        cloned = copy.deepcopy(unwrapped)
        cloned.to(self.current_device)
        cloned.eval()
        return cloned

    def _sync_old_policy(self) -> None:
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.old_policy.load_state_dict(unwrapped.state_dict())
        self.old_policy.eval()

    @staticmethod
    def _trim_response(sequence: "torch.Tensor", pad_token_id: int) -> "torch.Tensor":
        sequence = sequence.detach().clone()
        if pad_token_id is not None:
            non_pad = (sequence != pad_token_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0:
                return sequence.new_empty(0)
            last_index = non_pad[-1].item() + 1
            sequence = sequence[:last_index]
        return sequence

    def _generate_group(self, prompt: "torch.Tensor", temperature: float) -> list["torch.Tensor"]:
        input_ids = prompt.unsqueeze(0).to(self.current_device)
        attention_mask = torch.ones_like(input_ids)
        gen_config = GenerationConfig.from_dict(self.generation_config.to_dict())
        gen_config.temperature = temperature
        gen_config.do_sample = True
        gen_config.num_return_sequences = self.group_size

        logger.debug('Generating group with temperature=%.4f, prompt_tokens=%d', temperature, input_ids.size(-1))
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped = self.accelerator.unwrap_model(unwrapped_model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped)

            outputs: torch.Tensor = unwrapped.generate(
                generation_config=gen_config,
                logits_processor=get_logits_processor(),
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped, layernorm_params)

        generated = outputs[:, input_ids.size(-1) :]
        responses = []
        for seq in generated:
            responses.append(self._trim_response(seq.cpu(), self.tokenizer.pad_token_id))
        logger.debug('Generated %d responses, response_lengths=%s', len(responses), [len(r) for r in responses])
        return responses

    @staticmethod
    def _pad_sequences(
        sequences: list["torch.Tensor"],
        pad_token_id: int,
        dtype: torch.dtype = torch.long,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        padded = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id).to(dtype)
        attention_mask = pad_sequence(
            [torch.ones_like(seq, dtype=torch.long) for seq in sequences], batch_first=True, padding_value=0
        )
        return padded, attention_mask

    def _build_sequence_batch(
        self,
        prompt: "torch.Tensor",
        responses: list["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        prompt = prompt.to(self.current_device)
        all_sequences = []
        response_masks = []
        for response in responses:
            response = response.to(self.current_device)
            sequence = torch.cat((prompt, response), dim=-1)
            all_sequences.append(sequence)
            response_mask = torch.cat((torch.zeros_like(prompt, dtype=torch.long), torch.ones_like(response, dtype=torch.long)))
            response_masks.append(response_mask)

        input_ids, attention_mask = self._pad_sequences(all_sequences, self.tokenizer.pad_token_id)
        response_masks = pad_sequence(response_masks, batch_first=True, padding_value=0).to(torch.float32)
        return input_ids, attention_mask, response_masks

    def _sequence_logprobs(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        response_mask: "torch.Tensor",
        requires_grad: bool,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context, self.amp_context:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)
        target_tokens = input_ids[:, 1:]
        token_logprobs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        mask = response_mask[:, 1:]
        seq_logprobs = (token_logprobs * mask).sum(dim=-1)
        masked_token_count = mask.sum().clamp_min(1.0)
        policy_entropy = (token_entropy * mask).sum() / masked_token_count
        return seq_logprobs, token_logprobs, mask, policy_entropy

    def _compute_ref_kl(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        mask: "torch.Tensor",
        token_logprobs_new: "torch.Tensor",
    ) -> torch.Tensor:
        if self.ref_model is None or self.beta_ref_kl == 0:
            return torch.zeros((), device=self.current_device)

        with torch.no_grad(), self.amp_context:
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        ref_logits = ref_outputs[0] if isinstance(ref_outputs, tuple) else ref_outputs.logits
        ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        target_tokens = input_ids[:, 1:]
        token_logprobs_ref = ref_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        kl_tokens = (token_logprobs_new.detach() - token_logprobs_ref) * mask
        return kl_tokens.sum(dim=-1).mean()

    @staticmethod
    def _dispersion(values: torch.Tensor, mode: str) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros((), device=values.device)
        if mode == 'mad':
            median = values.median()
            return 1.4826 * (values - median).abs().median()
        if mode == 'iqr':
            q75 = values.quantile(0.75)
            q25 = values.quantile(0.25)
            return (q75 - q25) / 1.349
        return values.std(unbiased=False)

    @staticmethod
    def _vote_entropy(texts: list[str]) -> float:
        counter = Counter(texts)
        probs = torch.tensor(list(counter.values()), dtype=torch.float32)
        probs = probs / probs.sum()
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
        return float(entropy)

    @staticmethod
    def _skewness(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        mean = values.mean()
        std = values.std(unbiased=False)
        if std <= 0:
            return 0.0
        skew = (((values - mean) / std) ** 3).mean().item()
        return float(skew)

    @staticmethod
    def _aggregate_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
        if not stats_list:
            return {}
        merged = defaultdict(list)
        for stats in stats_list:
            for key, value in stats.items():
                merged[key].append(value)
        return {key: float(sum(values) / len(values)) for key, values in merged.items()}

    @staticmethod
    def compute_centered_uncertainty(raw_uncertainty: float, uncertainty_ema: float, ema_alpha: float) -> tuple[float, float]:
        updated_ema = ema_alpha * raw_uncertainty + (1.0 - ema_alpha) * uncertainty_ema
        return raw_uncertainty - updated_ema, updated_ema

    @staticmethod
    def compute_adaptive_temperature(
        tau_base: float,
        lambda_temp: float,
        centered_uncertainty: float,
        tau_min: float,
        tau_max: float,
    ) -> float:
        return float(torch.clamp(torch.tensor(tau_base * (1.0 + lambda_temp * centered_uncertainty)), min=tau_min, max=tau_max).item())

    @staticmethod
    def compute_adaptive_clip(
        eps_base: float,
        eps_min: float,
        eps_max: float,
        reward_dispersion: float,
        abs_skew: float,
        policy_entropy: float,
        step_kl: float,
        alpha_var: float,
        gamma_stepkl: float,
        zeta_skew: float,
        entropy_scale: float,
        entropy_floor: float,
    ) -> float:
        denom = 1.0 + alpha_var * reward_dispersion + gamma_stepkl * step_kl + zeta_skew * abs_skew
        entropy_bonus = max(policy_entropy, entropy_floor) * entropy_scale
        eps = eps_base * (1.0 + entropy_bonus) / denom
        return float(torch.clamp(torch.tensor(eps), min=eps_min, max=eps_max).item())

    def _log_prompt_debug(self, prefix: str, rewards: torch.Tensor, decoded_responses: list[str]) -> None:
        if not self.debug_reward_samples:
            return
        preview = []
        for idx, (reward, text) in enumerate(zip(rewards.tolist(), decoded_responses)):
            if idx >= self.max_debug_reward_items:
                break
            preview.append({'rank': idx, 'reward': reward, 'text_preview': text[:120]})
        logger.debug('%s reward preview: %s', prefix, preview)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:  # type: ignore[override]
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = math.ceil(self.args.max_steps / len(self.dataloader))
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running AGPO training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0("  Total train batch size (parallel, distributed & accumulation) = " f"{total_train_batch_size:,}")
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            self._sync_old_policy()
            self.model.train()
            self.optimizer.zero_grad()

            accumulated_loss = 0.0
            accumulated_stats: list[dict[str, float]] = []
            accumulated_rewards: list[float] = []

            for _ in range(self.args.gradient_accumulation_steps):
                try:
                    batch = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    batch = next(dataiter)

                loss, stats, rewards = self._agpo_step(batch)
                loss = loss / self.args.gradient_accumulation_steps
                accumulated_loss += float(loss.detach().cpu())
                accumulated_stats.append(stats)
                accumulated_rewards.extend(rewards)
                self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            aggregated_stats = self._aggregate_stats(accumulated_stats)
            aggregated_stats['agpo/learning_rate'] = self.lr_scheduler.get_last_lr()[0]
            aggregated_stats['agpo/loss/policy'] = accumulated_loss
            reward_mean = sum(accumulated_rewards) / len(accumulated_rewards) if accumulated_rewards else 0.0
            aggregated_stats['agpo/reward/mean'] = reward_mean
            aggregated_stats['agpo/controller/uncertainty_ema'] = self.uncertainty_ema
            aggregated_stats['agpo/controller/step_kl_ema'] = self.step_kl_ema

            loss_meter.update(aggregated_stats['agpo/loss/policy'], n=len(accumulated_rewards) or 1)
            reward_meter.update(reward_mean, n=len(accumulated_rewards) or 1)

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=aggregated_stats['agpo/learning_rate'],
                    epoch=round(step / steps_in_epoch, 2),
                    uncertainty_ema=round(self.uncertainty_ema, 4),
                    step_kl_ema=round(self.step_kl_ema, 4),
                )
                tqdm.write(str(logs))
                logs['step'] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:
                self.save_model(os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"))
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    def _agpo_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], list[float]]:
        self.model.eval()
        self.tokenizer.padding_side = 'right'

        controller_cfg = self.agpo_runtime_config.get('controller', {})
        ema_alpha = float(controller_cfg.get('ema_alpha', 0.1))
        entropy_scale = float(controller_cfg.get('entropy_scale', 0.1))

        input_ids: torch.Tensor = batch['input_ids'].to(self.current_device)
        attention_mask: torch.Tensor = batch['attention_mask'].to(self.current_device)
        prompts: list[torch.Tensor] = []
        for i in range(len(input_ids)):
            start_index = (attention_mask[i] != 0).nonzero(as_tuple=False)[0].item()
            prompts.append(input_ids[i, start_index:].detach().cpu())

        stats_per_prompt: list[dict[str, float]] = []
        losses: list[torch.Tensor] = []
        reward_collection: list[float] = []

        for prompt_index, prompt in enumerate(prompts):
            logger.debug('Starting AGPO prompt step idx=%d, prompt_len=%d', prompt_index, len(prompt))
            probe_responses = self._generate_group(prompt, self.finetuning_args.agpo_tau_base)
            probe_queries = [prompt for _ in range(self.group_size)]
            probe_rewards = self.get_rewards(probe_queries, probe_responses)
            probe_rewards_tensor = torch.tensor([reward.item() for reward in probe_rewards], dtype=torch.float32)
            probe_token_count = sum(len(response) for response in probe_responses)
            probe_texts = self.tokenizer.batch_decode(probe_responses, skip_special_tokens=True)

            dispersion = self._dispersion(probe_rewards_tensor, self.finetuning_args.agpo_use_robust_dispersion).item()
            vote_entropy = self._vote_entropy(probe_texts)
            skew = self._skewness(probe_rewards_tensor) if self.finetuning_args.agpo_w_k > 0 and probe_rewards_tensor.numel() > 0 else 0.0
            raw_uncertainty = (
                self.finetuning_args.agpo_w_r * dispersion
                + self.finetuning_args.agpo_w_e * vote_entropy
                + self.finetuning_args.agpo_w_k * abs(skew)
            )
            centered_uncertainty, self.uncertainty_ema = self.compute_centered_uncertainty(raw_uncertainty, self.uncertainty_ema, ema_alpha)
            tau_t = self.compute_adaptive_temperature(
                tau_base=self.finetuning_args.agpo_tau_base,
                lambda_temp=self.finetuning_args.agpo_lambda_temp,
                centered_uncertainty=centered_uncertainty,
                tau_min=self.finetuning_args.agpo_tau_min,
                tau_max=self.finetuning_args.agpo_tau_max,
            )
            self._log_prompt_debug('probe', probe_rewards_tensor, probe_texts)

            train_responses = self._generate_group(prompt, tau_t)
            train_queries = [prompt for _ in range(self.group_size)]
            train_rewards = self.get_rewards(train_queries, train_responses)
            train_rewards_tensor = torch.tensor([reward.item() for reward in train_rewards], dtype=torch.float32)
            reward_collection.extend(train_rewards_tensor.tolist())
            train_texts = self.tokenizer.batch_decode(train_responses, skip_special_tokens=True)
            self._log_prompt_debug('train', train_rewards_tensor, train_texts)

            reward_mean = train_rewards_tensor.mean()
            reward_dispersion = self._dispersion(train_rewards_tensor, self.finetuning_args.agpo_use_robust_dispersion)
            reward_dispersion = torch.clamp(reward_dispersion, min=1e-8)
            advantages = ((train_rewards_tensor - reward_mean) / reward_dispersion).to(self.current_device)

            input_ids_full, attention_mask_full, response_mask = self._build_sequence_batch(prompt, train_responses)
            logp_old, _, _, _ = self._sequence_logprobs(self.old_policy, input_ids_full, attention_mask_full, response_mask, requires_grad=False)
            logp_new, token_logprobs_new, mask, policy_entropy = self._sequence_logprobs(self.model, input_ids_full, attention_mask_full, response_mask, requires_grad=True)

            step_kl = torch.clamp((logp_old - logp_new.detach()).mean(), min=0.0).item()
            _, self.step_kl_ema = self.compute_centered_uncertainty(step_kl, self.step_kl_ema, ema_alpha)
            eps_adapt = self.compute_adaptive_clip(
                eps_base=self.finetuning_args.agpo_eps_base,
                eps_min=self.finetuning_args.agpo_eps_min,
                eps_max=self.finetuning_args.agpo_eps_max,
                reward_dispersion=reward_dispersion.item(),
                abs_skew=abs(skew),
                policy_entropy=float(policy_entropy.detach().cpu().item()),
                step_kl=self.step_kl_ema,
                alpha_var=self.finetuning_args.agpo_alpha_var,
                gamma_stepkl=self.finetuning_args.agpo_gamma_stepkl,
                zeta_skew=self.finetuning_args.agpo_zeta_skew,
                entropy_scale=entropy_scale,
                entropy_floor=self.clip_entropy_floor,
            )

            ratios = torch.exp(logp_new - logp_old.detach())
            clipped = torch.clamp(ratios, 1 - eps_adapt, 1 + eps_adapt)
            surrogate = torch.minimum(ratios * advantages, clipped * advantages)
            policy_loss = -surrogate.mean()
            kl_ref = self._compute_ref_kl(input_ids_full, attention_mask_full, mask, token_logprobs_new)
            loss = policy_loss + self.beta_ref_kl * kl_ref
            losses.append(loss)

            clip_sat = ((ratios <= 1 - eps_adapt) | (ratios >= 1 + eps_adapt)).float().mean().item()
            logger.debug(
                'Prompt idx=%d stats: raw_uncertainty=%.6f centered_uncertainty=%.6f tau=%.6f reward_mean=%.6f reward_dispersion=%.6f skew=%.6f policy_entropy=%.6f step_kl=%.6f step_kl_ema=%.6f eps=%.6f clip_sat=%.6f',
                prompt_index,
                raw_uncertainty,
                centered_uncertainty,
                tau_t,
                reward_mean.item(),
                reward_dispersion.item(),
                skew,
                float(policy_entropy.detach().cpu().item()),
                step_kl,
                self.step_kl_ema,
                eps_adapt,
                clip_sat,
            )

            stats = {
                'agpo/tau': tau_t,
                'agpo/sigma': reward_dispersion.item(),
                'agpo/step_kl': step_kl,
                'agpo/step_kl_ema': self.step_kl_ema,
                'agpo/eps': eps_adapt,
                'agpo/kl_ref': float(kl_ref.detach().cpu().item()),
                'agpo/clip_saturation': clip_sat,
                'agpo/vote_entropy': vote_entropy,
                'agpo/policy_entropy': float(policy_entropy.detach().cpu().item()),
                'agpo/reward/mean_group': reward_mean.item(),
                'agpo/uncertainty/raw': raw_uncertainty,
                'agpo/uncertainty/centered': centered_uncertainty,
                'agpo/uncertainty/ema': self.uncertainty_ema,
            }
            if self.finetuning_args.agpo_log_probe_metrics:
                stats['agpo/probe_dispersion'] = dispersion
                stats['agpo/probe_skew'] = skew
            if self.finetuning_args.agpo_count_probe_tokens_in_budget:
                stats['agpo/probe_tokens'] = float(probe_token_count)
            stats_per_prompt.append(stats)

        loss_total = torch.stack(losses).mean() if losses else torch.zeros((), device=self.current_device)

        self.model.train()
        self.tokenizer.padding_side = 'left'
        return loss_total, self._aggregate_stats(stats_per_prompt), reward_collection

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [dict(params=nodecay_params), dict(params=decay_params, weight_decay=training_args.weight_decay)]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer") -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_rewards(self, queries: list["torch.Tensor"], responses: list["torch.Tensor"]) -> list["torch.Tensor"]:
        if self.finetuning_args.reward_model_type == 'api':
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type in ['lora', 'oft']:
            replace_model(unwrapped_model, target='reward')
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type in ['lora', 'oft']:
            replace_model(unwrapped_model, target='default')

        rewards = values.gather(dim=-1, index=(batch['attention_mask'].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    ' stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,'
                    ' use zero_to_fp32.py to recover weights'
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)
        elif self.args.should_save:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
