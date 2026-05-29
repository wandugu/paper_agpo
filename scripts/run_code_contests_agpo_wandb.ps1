$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "..\uv_agpo\Scripts\python.exe"
$llamafactory = Join-Path $repoRoot "..\uv_agpo\Scripts\llamafactory-cli.exe"
$agpoConfig = "examples\train_lora\qwen3_0_6b_code_contests_agpo_wandb.yaml"
$rewardStdoutLog = Join-Path $repoRoot "saves\qwen3-0.6b\lora\code_reward_server.out.log"
$rewardStderrLog = Join-Path $repoRoot "saves\qwen3-0.6b\lora\code_reward_server.err.log"

Set-Location $repoRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $rewardStdoutLog) | Out-Null

$env:WANDB_PROJECT = "agpo-code-contests"
$env:WANDB_RUN_GROUP = "qwen3-0.6b-code-contests"
$env:WANDB_MODE = "online"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$gccBin = "C:\soft\msys64\ucrt64\bin"
if ((Test-Path $gccBin) -and (($env:Path -split ";") -notcontains $gccBin)) {
    $env:Path = "$gccBin;$env:Path"
}

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

reward_data = config.get("reward_data")
if not reward_data:
    raise SystemExit(f"Missing reward_data in {config_path}")

parsed = urlparse(reward_model)
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    raise SystemExit(f"reward_model must be a full HTTP URL, got: {reward_model}")

port = parsed.port or (443 if parsed.scheme == "https" else 80)
print(json.dumps({
    "url": reward_model.rstrip("/"),
    "data": str(reward_data),
    "host": parsed.hostname,
    "port": port,
    "health_url": reward_model.rstrip("/") + "/health",
    "test_suites": str(config.get("code_reward_test_suites", "public,generated")),
    "max_tests": int(config.get("code_reward_max_tests", 8)),
    "timeout": float(config.get("code_reward_timeout", 2.0)),
    "language": str(config.get("code_reward_language", "python")),
}))
"@ | ConvertFrom-Json

$env:CODE_REWARD_DATA = $rewardConfig.data
$env:CODE_REWARD_TEST_SUITES = $rewardConfig.test_suites
$env:CODE_REWARD_MAX_TESTS = "$($rewardConfig.max_tests)"
$env:CODE_REWARD_TIMEOUT = "$($rewardConfig.timeout)"
$env:CODE_REWARD_LANGUAGE = $rewardConfig.language

$server = Start-Process `
    -FilePath $python `
    -ArgumentList @(
        "scripts\code_reward_server.py",
        "--dataset-dir", $rewardConfig.data,
        "--test-suites", $rewardConfig.test_suites,
        "--max-tests", "$($rewardConfig.max_tests)",
        "--timeout", "$($rewardConfig.timeout)",
        "--language", $rewardConfig.language,
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
