import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def clean_env() -> dict[str, str]:
    env: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in os.environ.items():
        lowered = key.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        env[key] = value

    env.update(
        {
            "WANDB_PROJECT": "agpo-mixed-answer",
            "WANDB_RUN_GROUP": "qwen3-0.6b-mixed-answer",
            "WANDB_MODE": "online",
            "PYTHONUTF8": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def load_reward_config(config_path: Path) -> dict[str, str | int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    reward_model = config.get("reward_model")
    if not reward_model:
        raise SystemExit(f"Missing reward_model in {config_path}")

    answer_data = config.get("reward_data")
    if not answer_data:
        raise SystemExit(f"Missing reward_data in {config_path}")

    parsed = urlparse(str(reward_model))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"reward_model must be a full HTTP URL, got: {reward_model}")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return {
        "url": str(reward_model).rstrip("/"),
        "answer_data": str(answer_data),
        "host": parsed.hostname,
        "port": port,
        "health_url": str(reward_model).rstrip("/") + "/health",
    }


def wait_for_health(url: str, seconds: int = 120) -> dict[str, object]:
    last_error = None
    for _ in range(seconds):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - report last health failure in supervisor log.
            last_error = exc
            time.sleep(1)

    raise RuntimeError(f"Reward server did not become healthy at {url}: {last_error}")


def terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def supervise(args: argparse.Namespace) -> int:
    config_path = (REPO_ROOT / args.config).resolve()
    reward = load_reward_config(config_path)
    env = clean_env()
    env["ANSWER_REWARD_DATA"] = str(reward["answer_data"])

    reward_stdout = REPO_ROOT / "saves" / "qwen3-0.6b" / "lora" / f"answer_reward_server_{args.name}.out.log"
    reward_stderr = REPO_ROOT / "saves" / "qwen3-0.6b" / "lora" / f"answer_reward_server_{args.name}.err.log"
    reward_stdout.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "supervisor_start", "config": str(config_path), "reward": reward}, ensure_ascii=False), flush=True)
    with reward_stdout.open("ab") as server_out, reward_stderr.open("ab") as server_err:
        server = subprocess.Popen(
            [
                sys.executable,
                "scripts/answer_reward_server.py",
                "--data",
                str(reward["answer_data"]),
                "--host",
                str(reward["host"]),
                "--port",
                str(reward["port"]),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=server_out,
            stderr=server_err,
        )

    try:
        health = wait_for_health(str(reward["health_url"]))
        print(json.dumps({"event": "reward_healthy", "health": health}, ensure_ascii=False), flush=True)

        command = [sys.executable, "-m", "llamafactory.cli", "train", str(config_path)]
        print(json.dumps({"event": "train_start", "command": command}, ensure_ascii=False), flush=True)
        train = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
        rc = train.wait()
        print(json.dumps({"event": "train_exit", "returncode": rc}, ensure_ascii=False), flush=True)
        return rc
    finally:
        terminate(server)
        print(json.dumps({"event": "reward_server_stopped"}, ensure_ascii=False), flush=True)


def detach(args: argparse.Namespace) -> None:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "saves" / "qwen3-0.6b" / "lora" / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    outer_stdout = log_dir / f"{args.name}_{run_id}.outer.out.log"
    outer_stderr = log_dir / f"{args.name}_{run_id}.outer.err.log"
    pid_file = log_dir / f"{args.name}_{run_id}.pid.txt"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        args.config,
        "--name",
        args.name,
        "--run-id",
        run_id,
        "--supervise",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = clean_env()
    with outer_stdout.open("ab") as out, outer_stderr.open("ab") as err:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=out,
            stderr=err,
            creationflags=creationflags,
        )

    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(
        json.dumps(
            {
                "pid": process.pid,
                "run_id": run_id,
                "outer_stdout": str(outer_stdout),
                "outer_stderr": str(outer_stderr),
                "pid_file": str(pid_file),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an answer-reward AGPO run with a supervised reward server.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()

    if args.supervise:
        raise SystemExit(supervise(args))
    detach(args)


if __name__ == "__main__":
    main()
