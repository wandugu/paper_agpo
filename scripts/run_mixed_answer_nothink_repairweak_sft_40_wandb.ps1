$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$config = "examples\train_lora\qwen3_0_6b_mixed_answer_nothink_repairweak_sft_40_wandb.yaml"
$runName = "mixed_answer_repairweak_sft_40"

Set-Location $repoRoot

$env:WANDB_PROJECT = "agpo-mixed-answer"
$env:WANDB_RUN_GROUP = "qwen3-0.6b-mixed-answer"
$env:WANDB_MODE = "online"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

..\uv_agpo\Scripts\python.exe scripts\launch_lf_train.py --config $config --name $runName
