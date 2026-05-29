import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INFO_PATH = DATA_DIR / "dataset_info.json"
LOG_DIR = ROOT / "saves" / "smoke_logs"
CONFIG_DIR = ROOT / "saves" / "smoke_configs"
OUTPUT_DIR = ROOT / "saves" / "smoke"


def load_dataset_info() -> dict[str, Any]:
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


def local_dataset_keys(info: dict[str, Any]) -> list[str]:
    keys = []
    for key, value in info.items():
        file_name = value.get("file_name")
        if file_name and (DATA_DIR / file_name).exists():
            keys.append(key)
    return keys


def llama_factory_cli() -> str:
    exe_name = "llamafactory-cli.exe" if os.name == "nt" else "llamafactory-cli"
    sibling = Path(sys.executable).with_name(exe_name)
    return str(sibling if sibling.exists() else exe_name)


def to_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_config(args: argparse.Namespace, dataset_key: str, reward_url: str) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{dataset_key}.yaml"
    output_path = OUTPUT_DIR / dataset_key
    text = f"""### model
model_name_or_path: {args.model_path}
reward_model: {reward_url}
reward_model_type: api
trust_remote_code: true

### method
stage: ppo
rl_algo: agpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all
agpo_group_size: 2
agpo_update_epochs: 1
agpo_tau_base: 1.0
agpo_tau_min: 0.6
agpo_tau_max: 1.3
agpo_lambda_temp: 0.15
agpo_eps_base: 0.2
agpo_eps_min: 0.05
agpo_eps_max: 0.4
agpo_beta_ref_kl: 0.03
agpo_use_robust_dispersion: std

### dataset
dataset: {dataset_key}
template: qwen3
cutoff_len: {args.cutoff_len}
max_samples: {args.max_samples}
overwrite_cache: true
preprocessing_num_workers: 1
dataloader_num_workers: 0

### output
output_dir: {to_posix(output_path)}
logging_steps: 1
save_steps: 9999
plot_loss: false
overwrite_output_dir: true
report_to: none
run_name: smoke-{dataset_key}
disable_tqdm: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
max_steps: {args.max_steps}
lr_scheduler_type: cosine
warmup_ratio: 0.0
bf16: true
ddp_timeout: 180000000

### generate
max_new_tokens: {args.max_new_tokens}
top_k: 0
top_p: 0.9
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path


def wait_for_health(url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.5)

    raise RuntimeError(f"reward server did not become healthy: {last_error}")


def terminate(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_one(args: argparse.Namespace, info: dict[str, Any], dataset_key: str, port: int) -> dict[str, Any]:
    file_name = info[dataset_key]["file_name"]
    data_path = DATA_DIR / file_name
    reward_url = f"http://127.0.0.1:{port}"
    health_url = f"{reward_url}/health"
    config_path = write_config(args, dataset_key, reward_url)

    result: dict[str, Any] = {
        "dataset": dataset_key,
        "file": to_posix(data_path),
        "config": to_posix(config_path),
        "reward_url": reward_url,
        "ok": False,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    reward_out = LOG_DIR / f"{dataset_key}.reward.out.log"
    reward_err = LOG_DIR / f"{dataset_key}.reward.err.log"
    train_out = LOG_DIR / f"{dataset_key}.train.out.log"
    train_err = LOG_DIR / f"{dataset_key}.train.err.log"
    result.update(
        {
            "reward_stdout": to_posix(reward_out),
            "reward_stderr": to_posix(reward_err),
            "train_stdout": to_posix(train_out),
            "train_stderr": to_posix(train_err),
        }
    )

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"

    start = time.time()
    reward_proc = None
    with reward_out.open("w", encoding="utf-8") as rout, reward_err.open("w", encoding="utf-8") as rerr:
        try:
            reward_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "answer_reward_server.py"),
                    "--data",
                    str(data_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=env,
                stdout=rout,
                stderr=rerr,
            )
            result["health"] = wait_for_health(health_url, args.health_timeout)

            with train_out.open("w", encoding="utf-8") as tout, train_err.open("w", encoding="utf-8") as terr:
                completed = subprocess.run(
                    [llama_factory_cli(), "train", str(config_path)],
                    cwd=ROOT,
                    env=env,
                    stdout=tout,
                    stderr=terr,
                    timeout=args.train_timeout,
                )

            result["returncode"] = completed.returncode
            result["ok"] = completed.returncode == 0
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            if reward_proc is not None:
                terminate(reward_proc)

    result["elapsed_s"] = round(time.time() - start, 2)
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


def write_summary(results: list[dict[str, Any]]) -> None:
    summary_path = LOG_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-step AGPO smoke tests for local datasets.")
    parser.add_argument("--only", nargs="*", help="Dataset keys to test. Defaults to every local dataset in dataset_info.json.")
    parser.add_argument("--start-port", type=int, default=8100)
    parser.add_argument("--model-path", default="models/Qwen3-0.6B")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--health-timeout", type=float, default=60)
    parser.add_argument("--train-timeout", type=float, default=900)
    parser.add_argument("--stop-on-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = load_dataset_info()
    dataset_keys = args.only or local_dataset_keys(info)
    unknown = [key for key in dataset_keys if key not in info]
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}")

    missing = [key for key in dataset_keys if not (DATA_DIR / info[key].get("file_name", "")).exists()]
    if missing:
        raise FileNotFoundError(f"Missing local dataset files for keys: {missing}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Testing {len(dataset_keys)} dataset entries with {args.max_steps} train step each.", flush=True)

    results: list[dict[str, Any]] = []
    for index, dataset_key in enumerate(dataset_keys):
        port = args.start_port + index
        print(f"[{index + 1}/{len(dataset_keys)}] {dataset_key} on port {port}", flush=True)
        result = run_one(args, info, dataset_key, port)
        results.append(result)
        write_summary(results)

        status = "OK" if result["ok"] else "FAIL"
        detail = f"returncode={result.get('returncode')}" if "returncode" in result else result.get("error", "")
        print(f"[{status}] {dataset_key} ({result['elapsed_s']}s) {detail}", flush=True)
        if args.stop_on_fail and not result["ok"]:
            break

    ok_count = sum(1 for item in results if item["ok"])
    print(f"Finished: {ok_count}/{len(results)} passed. Summary: {to_posix(LOG_DIR / 'summary.json')}", flush=True)
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
