$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$llamafactory = Join-Path $repoRoot "..\uv_agpo\Scripts\llamafactory-cli.exe"
$sftConfig = "examples\train_lora\qwen3_0_6b_code_contests_sft.yaml"

Set-Location $repoRoot

$env:WANDB_PROJECT = "agpo-code-contests"
$env:WANDB_RUN_GROUP = "qwen3-0.6b-code-contests"
$env:WANDB_MODE = "online"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"

& $llamafactory train $sftConfig
