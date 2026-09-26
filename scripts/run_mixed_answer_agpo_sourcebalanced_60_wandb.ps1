$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "..\uv_agpo\Scripts\python.exe"
$config = "examples\train_lora\qwen3_0_6b_mixed_answer_agpo_sourcebalanced_60_wandb.yaml"
$runId = Get-Date -Format "yyyyMMdd_HHmmss"

Set-Location $repoRoot
& $python scripts\launch_answer_agpo_run.py `
  --config $config `
  --name mixed_answer_sourcebalanced_60 `
  --run-id $runId
