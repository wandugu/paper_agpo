# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Optional

from ...data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_ref_model, create_reward_model
from .trainer import CustomAGPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_agpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
) -> None:
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = MultiModalDataCollatorForSeq2Seq(template=template, model=model, **tokenizer_module)

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)

    agpo_trainer = CustomAGPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )

    if training_args.do_train:
        agpo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        agpo_trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        agpo_trainer.save_state()  # must be called after save_model to have a folder
        if agpo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
