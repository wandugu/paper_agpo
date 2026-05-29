$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "..\uv_agpo\Scripts\python.exe"
$llamafactory = Join-Path $repoRoot "..\uv_agpo\Scripts\llamafactory-cli.exe"
$agpoConfig = "examples\train_lora\qwen3_0_6b_mixed_answer_agpo_wandb.yaml"
$rewardStdoutLog = Join-Path $repoRoot "saves\qwen3-0.6b\lora\answer_reward_server_mixed_answer.out.log"
$rewardStderrLog = Join-Path $repoRoot "saves\qwen3-0.6b\lora\answer_reward_server_mixed_answer.err.log"

Set-Location $repoRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $rewardStdoutLog) | Out-Null

$env:WANDB_PROJECT = "agpo-mixed-answer"
$env:WANDB_RUN_GROUP = "qwen3-0.6b-mixed-answer"
$env:WANDB_MODE = "online"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$rewardConfig = & $python -c @'
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
reward_model = config.get('reward_model')
if not reward_model:
    raise SystemExit(f'Missing reward_model in {config_path}')

answer_data = config.get('reward_data')
if not answer_data:
    raise SystemExit(f'Missing reward_data in {config_path}')

parsed = urlparse(reward_model)
if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
    raise SystemExit(f'reward_model must be a full HTTP URL, got: {reward_model}')

port = parsed.port or (443 if parsed.scheme == 'https' else 80)
print(json.dumps({
    'url': reward_model.rstrip('/'),
    'answer_data': str(answer_data),
    'host': parsed.hostname,
    'port': port,
    'health_url': reward_model.rstrip('/') + '/health',
}))
'@ $agpoConfig | ConvertFrom-Json

$env:ANSWER_REWARD_DATA = $rewardConfig.answer_data

$server = Start-Process `
    -FilePath $python `
    -ArgumentList @(
        "scripts\answer_reward_server.py",
        "--data", $rewardConfig.answer_data,
        "--host", $rewardConfig.host,
        "--port", "$($rewardConfig.port)"
    ) `
    -RedirectStandardOutput $rewardStdoutLog `
    -RedirectStandardError $rewardStderrLog `
    -WindowStyle Hidden `
    -PassThru

try {
    for ($i = 0; $i -lt 90; $i++) {
        try {
            Invoke-RestMethod -Uri $rewardConfig.health_url -TimeoutSec 2 | Out-Null
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    Invoke-RestMethod -Uri $rewardConfig.health_url -TimeoutSec 5 | ConvertTo-Json -Depth 4
    & $llamafactory train $agpoConfig
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
