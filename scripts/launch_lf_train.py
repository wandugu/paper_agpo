import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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


def supervise(args: argparse.Namespace) -> int:
    config_path = (REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Missing config: {config_path}")

    command = [sys.executable, "-m", "llamafactory.cli", "train", str(config_path)]
    print(json.dumps({"event": "train_start", "command": command}, ensure_ascii=False), flush=True)
    train = subprocess.Popen(command, cwd=REPO_ROOT, env=clean_env())
    rc = train.wait()
    print(json.dumps({"event": "train_exit", "returncode": rc}, ensure_ascii=False), flush=True)
    return rc


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
    with outer_stdout.open("ab") as out, outer_stderr.open("ab") as err:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=clean_env(),
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
    parser = argparse.ArgumentParser(description="Launch a detached LLaMA-Factory train run with logs.")
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
