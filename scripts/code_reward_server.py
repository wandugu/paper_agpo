import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from datasets import load_from_disk
from fastapi import FastAPI
from pydantic import BaseModel


USER_BLOCK_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
ASSISTANT_BLOCK_RE = re.compile(r"<\|im_start\|>assistant\n?(.*)", re.DOTALL)
FENCED_CODE_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)\s*\n(.*?)```", re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
PYTHON3_PROMPT_SUFFIX = (
    "Write a Python 3 program that reads from standard input and writes to standard output. "
    "Return only the code, without Markdown fences."
)
LANG_ALIASES = {
    "py": "python",
    "python": "python",
    "python3": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
}


class RewardRequest(BaseModel):
    model: str | None = None
    messages: list[str]


@dataclass(frozen=True)
class TestCase:
    input: str
    output: str


@dataclass(frozen=True)
class Problem:
    problem_id: str
    name: str
    description: str
    tests: tuple[TestCase, ...]
    input_file: str
    output_file: str


@dataclass(frozen=True)
class ExecutionResult:
    score: float
    passed: int
    total: int
    language: str
    error: str = ""


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalise_output(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def label_name(feature: Any, value: Any) -> str:
    if value is None:
        return ""
    try:
        return feature.int2str(int(value))
    except Exception:
        return str(value)


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8", errors="ignore")).hexdigest()


class CodeContestScorer:
    def __init__(
        self,
        dataset_dir: Path,
        test_suites: tuple[str, ...],
        max_tests: int,
        timeout: float,
        language: str,
        prefix_chars: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.test_suites = test_suites
        self.max_tests = max_tests
        self.timeout = timeout
        self.language = language
        self.prefix_chars = prefix_chars
        self.gpp_path = self._find_gpp()
        self.cache: dict[str, ExecutionResult] = {}

        dataset = load_from_disk(str(dataset_dir))
        self.by_prompt: dict[str, Problem] = {}
        self.by_prefix: dict[str, list[Problem]] = {}
        self.problems: list[Problem] = []

        for split_name, split_ds in dataset.items():
            for index, row in enumerate(split_ds):
                tests = self._select_tests(row)
                problem = Problem(
                    problem_id=f"{split_name}:{index}",
                    name=row.get("name") or "",
                    description=row.get("description") or "",
                    tests=tuple(tests),
                    input_file=row.get("input_file") or "",
                    output_file=row.get("output_file") or "",
                )
                self.problems.append(problem)
                self._index_problem(problem)

    @staticmethod
    def _find_gpp() -> str | None:
        configured = os.environ.get("CODE_REWARD_GPP")
        if configured and Path(configured).exists():
            return configured

        discovered = shutil.which("g++")
        if discovered:
            return discovered

        for candidate in (
            Path("C:/soft/msys64/ucrt64/bin/g++.exe"),
            Path("C:/msys64/ucrt64/bin/g++.exe"),
        ):
            if candidate.exists():
                return str(candidate)

        return None

    def _index_problem(self, problem: Problem) -> None:
        prompts = [
            problem.description,
            f"{problem.description}\n\n{PYTHON3_PROMPT_SUFFIX}",
        ]
        for prompt in prompts:
            norm = normalise_text(prompt)
            if not norm:
                continue

            self.by_prompt.setdefault(norm, problem)
            self.by_prefix.setdefault(norm[: self.prefix_chars], []).append(problem)

    def _select_tests(self, row: dict[str, Any]) -> list[TestCase]:
        tests: list[TestCase] = []
        for suite in self.test_suites:
            suite_data = row.get(f"{suite}_tests") or {}
            inputs = suite_data.get("input") or []
            outputs = suite_data.get("output") or []
            tests.extend(TestCase(str(inp), str(out)) for inp, out in zip(inputs, outputs))

        if self.max_tests > 0 and len(tests) > self.max_tests:
            public_tests = []
            other_tests = []
            public_data = row.get("public_tests") or {}
            public_pairs = {
                (str(inp), str(out))
                for inp, out in zip(public_data.get("input") or [], public_data.get("output") or [])
            }
            for test in tests:
                if (test.input, test.output) in public_pairs:
                    public_tests.append(test)
                else:
                    other_tests.append(test)

            slots = max(0, self.max_tests - len(public_tests))
            other_tests.sort(key=lambda test: stable_hash(row.get("name") or "", test.input, test.output))
            tests = (public_tests + other_tests[:slots])[: self.max_tests]

        return tests

    @staticmethod
    def _extract_question(message: str) -> str | None:
        matches = USER_BLOCK_RE.findall(message)
        if matches:
            return normalise_text(matches[-1])
        return normalise_text(message)

    @staticmethod
    def _strip_prompt_suffix(question: str) -> str:
        suffix = normalise_text(PYTHON3_PROMPT_SUFFIX)
        if question.endswith(suffix):
            return question[: -len(suffix)].strip()
        return question

    def _find_problem(self, message: str) -> Problem | None:
        question = self._extract_question(message)
        if not question:
            return None

        candidates = [question, self._strip_prompt_suffix(question)]
        for candidate in candidates:
            if candidate in self.by_prompt:
                return self.by_prompt[candidate]

        for candidate in candidates:
            prefix = candidate[: self.prefix_chars]
            for problem in self.by_prefix.get(prefix, []):
                norm_desc = normalise_text(problem.description)
                norm_py3 = normalise_text(f"{problem.description}\n\n{PYTHON3_PROMPT_SUFFIX}")
                if candidate.startswith(norm_desc[: min(len(candidate), len(norm_desc), self.prefix_chars)]):
                    return problem
                if candidate.startswith(norm_py3[: min(len(candidate), len(norm_py3), self.prefix_chars)]):
                    return problem

        return None

    @staticmethod
    def _extract_response(message: str) -> str:
        match = ASSISTANT_BLOCK_RE.search(message)
        if match:
            response = match.group(1)
        else:
            response = message

        if "<|im_end|>" in response:
            response = response.split("<|im_end|>", 1)[0]
        return response.strip()

    def _extract_code(self, response: str) -> tuple[str, str]:
        response = THINK_BLOCK_RE.sub("", response).strip()
        fenced = FENCED_CODE_RE.findall(response)
        if fenced:
            preferred = []
            fallback = []
            for language, code in fenced:
                normalised_lang = LANG_ALIASES.get(language.lower().strip(), "")
                if normalised_lang == self.language:
                    preferred.append((normalised_lang, code))
                elif normalised_lang:
                    fallback.append((normalised_lang, code))
                else:
                    fallback.append(("", code))

            selected_lang, code = (preferred or fallback)[0]
            return code.strip(), selected_lang or self._detect_language(code)

        code = response.strip()
        return code, self._detect_language(code)

    def _detect_language(self, code: str) -> str:
        if self.language in {"python", "cpp"}:
            return self.language

        lowered = code.lower()
        if "#include" in lowered or "using namespace std" in lowered or "int main(" in lowered:
            return "cpp"
        return "python"

    def score(self, message: str) -> float:
        result = self.score_detailed(message)
        return result.score

    def score_detailed(self, message: str) -> ExecutionResult:
        problem = self._find_problem(message)
        if problem is None:
            return ExecutionResult(score=0.0, passed=0, total=0, language=self.language, error="problem_not_found")

        response = self._extract_response(message)
        code, language = self._extract_code(response)
        if not code:
            return ExecutionResult(score=0.0, passed=0, total=len(problem.tests), language=language, error="empty_code")

        cache_key = stable_hash(problem.problem_id, language, code)
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self._execute(problem, code, language)
        if len(self.cache) > 4096:
            self.cache.clear()
        self.cache[cache_key] = result
        return result

    def _execute(self, problem: Problem, code: str, language: str) -> ExecutionResult:
        if not problem.tests:
            return ExecutionResult(score=0.0, passed=0, total=0, language=language, error="no_tests")

        with tempfile.TemporaryDirectory(prefix="code_reward_") as tmp:
            tmp_path = Path(tmp)
            if language == "cpp":
                runner, compile_error = self._compile_cpp(code, tmp_path)
                if compile_error:
                    return ExecutionResult(
                        score=0.0, passed=0, total=len(problem.tests), language=language, error=compile_error
                    )
            elif language == "python":
                runner = [sys.executable, "-I", str(tmp_path / "main.py")]
                (tmp_path / "main.py").write_text(code, encoding="utf-8", newline="\n")
            else:
                return ExecutionResult(
                    score=0.0, passed=0, total=len(problem.tests), language=language, error=f"unsupported_language:{language}"
                )

            passed = 0
            last_error = ""
            for test in problem.tests:
                ok, error = self._run_test(runner, tmp_path, problem, test)
                if ok:
                    passed += 1
                elif error and not last_error:
                    last_error = error

            total = len(problem.tests)
            return ExecutionResult(
                score=passed / total if total else 0.0,
                passed=passed,
                total=total,
                language=language,
                error=last_error,
            )

    def _compile_cpp(self, code: str, tmp_path: Path) -> tuple[list[str], str]:
        if not self.gpp_path:
            return [], "g++_not_found"

        source = tmp_path / "main.cpp"
        binary = tmp_path / ("main.exe" if os.name == "nt" else "main")
        source.write_text(code, encoding="utf-8", newline="\n")
        try:
            completed = subprocess.run(
                [self.gpp_path, "-std=c++17", "-O2", str(source), "-o", str(binary)],
                cwd=tmp_path,
                text=True,
                capture_output=True,
                timeout=max(10.0, self.timeout),
            )
        except subprocess.TimeoutExpired:
            return [], "compile_timeout"

        if completed.returncode != 0:
            return [], f"compile_error:{completed.stderr[:240]}"
        return [str(binary)], ""

    def _run_test(self, runner: list[str], tmp_path: Path, problem: Problem, test: TestCase) -> tuple[bool, str]:
        stdin = test.input
        expected = normalise_output(test.output)

        input_path = tmp_path / problem.input_file if problem.input_file else None
        output_path = tmp_path / problem.output_file if problem.output_file else None
        if input_path:
            input_path.write_text(test.input, encoding="utf-8", newline="\n")
            stdin = ""
        if output_path and output_path.exists():
            output_path.unlink()

        try:
            completed = subprocess.run(
                runner,
                cwd=tmp_path,
                input=stdin,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as exc:
            return False, f"runtime_exception:{exc!r}"

        if completed.returncode != 0:
            return False, f"runtime_error:{completed.stderr[:240]}"

        if output_path and output_path.exists():
            actual = normalise_output(output_path.read_text(encoding="utf-8", errors="replace"))
        else:
            actual = normalise_output(completed.stdout)

        return actual == expected, ""


def parse_test_suites(value: str) -> tuple[str, ...]:
    suites = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"public", "generated", "private"}
    unknown = set(suites) - allowed
    if unknown:
        raise ValueError(f"Unknown test suites: {sorted(unknown)}")
    return suites or ("public", "generated")


def create_app(
    dataset_dir: Path,
    test_suites: tuple[str, ...],
    max_tests: int,
    timeout: float,
    language: str,
    prefix_chars: int,
) -> FastAPI:
    scorer = CodeContestScorer(
        dataset_dir=dataset_dir,
        test_suites=test_suites,
        max_tests=max_tests,
        timeout=timeout,
        language=language,
        prefix_chars=prefix_chars,
    )
    app = FastAPI(title="CodeContests execution reward server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "dataset": str(dataset_dir),
            "problems": len(scorer.problems),
            "test_suites": list(test_suites),
            "max_tests": max_tests,
            "timeout": timeout,
            "language": language,
            "python": sys.executable,
            "g++": scorer.gpp_path,
            "warning": "Executes generated code locally. Use a stronger sandbox for untrusted or large runs.",
        }

    @app.post("/")
    def reward(payload: RewardRequest) -> dict[str, list[float]]:
        return {"scores": [scorer.score(message) for message in payload.messages]}

    @app.post("/debug")
    def reward_debug(payload: RewardRequest) -> dict[str, list[dict[str, Any]]]:
        return {"results": [scorer.score_detailed(message).__dict__ for message in payload.messages]}

    return app


default_dataset_dir = Path(os.environ.get("CODE_REWARD_DATA", "data/code_contests/hf_dataset"))
default_test_suites = parse_test_suites(os.environ.get("CODE_REWARD_TEST_SUITES", "public,generated"))
default_max_tests = int(os.environ.get("CODE_REWARD_MAX_TESTS", "8"))
default_timeout = float(os.environ.get("CODE_REWARD_TIMEOUT", "2.0"))
default_language = os.environ.get("CODE_REWARD_LANGUAGE", "python").lower()
app = FastAPI(title="CodeContests execution reward server")


@app.get("/health")
def unconfigured_health() -> dict[str, str]:
    return {"status": "not_configured", "hint": "Start this script directly so it can load the dataset."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--test-suites", default=",".join(default_test_suites))
    parser.add_argument("--max-tests", type=int, default=default_max_tests)
    parser.add_argument("--timeout", type=float, default=default_timeout)
    parser.add_argument("--language", choices=["auto", "python", "cpp"], default=default_language)
    parser.add_argument("--prefix-chars", type=int, default=512)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    global app
    app = create_app(
        dataset_dir=args.dataset_dir,
        test_suites=parse_test_suites(args.test_suites),
        max_tests=args.max_tests,
        timeout=args.timeout,
        language=args.language,
        prefix_chars=args.prefix_chars,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
