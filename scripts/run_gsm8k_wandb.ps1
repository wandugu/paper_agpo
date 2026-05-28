$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "..\uv_agpo\Scripts\python.exe"
$llamafactory = Join-Path $repoRoot "..\uv_agpo\Scripts\llamafactory-cli.exe"
$agpoConfig = "examples\train_lora\qwen3_0_6b_gsm8k_agpo_wandb.yaml"
$rewardStdoutLog = Join-Path $repoRoot "saves\qwen3-0.6b\lora\gsm8k_reward_server.out.log"
$rewardStderrLog = Join-Path $repoRoot "saves\qwen3-0.6b\lora\gsm8k_reward_server.err.log"

Set-Location $repoRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $rewardStdoutLog) | Out-Null

$env:WANDB_PROJECT = "agpo-gsm8k"
$env:WANDB_RUN_GROUP = "qwen3-0.6b-gsm8k"
$env:WANDB_MODE = "online"
$env:GSM8K_REWARD_DATA = "data\gsm8k\rl_train.json"

$rewardConfig = & $python -c @"
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml

config_path = Path(r"$agpoConfig")
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
reward_model = config.get("reward_model")
if not reward_model:
    raise SystemExit(f"Missing reward_model in {config_path}")

parsed = urlparse(reward_model)
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    raise SystemExit(f"reward_model must be a full HTTP URL, got: {reward_model}")

port = parsed.port or (443 if parsed.scheme == "https" else 80)
print(json.dumps({
    "url": reward_model.rstrip("/"),
    "host": parsed.hostname,
    "port": port,
    "health_url": reward_model.rstrip("/") + "/health",
}))
"@ | ConvertFrom-Json

$server = Start-Process `
    -FilePath $python `
    -ArgumentList @("scripts\gsm8k_reward_server.py", "--data", "data\gsm8k\rl_train.json", "--host", $rewardConfig.host, "--port", "$($rewardConfig.port)") `
    -RedirectStandardOutput $rewardStdoutLog `
    -RedirectStandardError $rewardStderrLog `
    -WindowStyle Hidden `
    -PassThru

try {
    for ($i = 0; $i -lt 30; $i++) {
        try {
            Invoke-RestMethod -Uri $rewardConfig.health_url -TimeoutSec 2 | Out-Null
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    & $llamafactory train $agpoConfig
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
