import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from code_reward_server import CodeContestScorer, ExecutionResult, parse_test_suites


USER_BLOCK_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
ASSISTANT_BLOCK_RE = re.compile(r"<\|im_start\|>assistant\n?(.*)", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
LETTER_RE = re.compile(
    r"(?:answer|choice|option)\s*(?:is|:)?\s*[\(\[]?([A-E])[\)\].]?\b",
    re.IGNORECASE,
)


class RewardRequest(BaseModel):
    model: str | None = None
    messages: list[str]


@dataclass(frozen=True)
class HybridResult:
    source: str
    score: float
    detail: dict[str, Any]


class AnswerExactMatchScorer:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        rows = self._load_rows(data_path)
        self.answers = {
            self._norm_question(row["instruction"]): str(answer)
            for row in rows
            if (answer := row.get("answer") or row.get("final_answer")) not in (None, "")
        }
        self.fallback_items = list(self.answers.items())

    @staticmethod
    def _load_rows(data_path: Path) -> list[dict[str, Any]]:
        if data_path.suffix == ".jsonl":
            rows = []
            with data_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows

        rows = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of examples in {data_path}")
        return rows

    @staticmethod
    def _norm_question(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _norm_answer(text: str) -> str:
        text = text.strip()
        boxed = BOXED_RE.fullmatch(text)
        if boxed:
            text = boxed.group(1).strip()

        if re.fullmatch(r"[A-Ea-e]", text):
            return text.upper()

        text = text.replace("\\$", "")
        text = text.replace("$", "")
        text = text.replace(",", "")
        text = text.rstrip(".")
        try:
            return str(Decimal(text).normalize())
        except InvalidOperation:
            return re.sub(r"\s+", "", text).lower()

    def extract_question(self, message: str) -> str | None:
        matches = USER_BLOCK_RE.findall(message)
        if matches:
            return self._norm_question(matches[-1])

        normalised_message = self._norm_question(message)
        for question, _ in self.fallback_items:
            if question in normalised_message:
                return question
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
        return response

    def _extract_answer(self, response: str, expect_letter: bool) -> str | None:
        if expect_letter:
            letter_matches = LETTER_RE.findall(response)
            if letter_matches:
                return self._norm_answer(letter_matches[-1])

        if "####" in response:
            tail = response.rsplit("####", 1)[-1]
            match = NUMBER_RE.search(tail)
            if match:
                return self._norm_answer(match.group(0))

        boxed = BOXED_RE.findall(response)
        if boxed:
            match = NUMBER_RE.search(boxed[-1])
            if match:
                return self._norm_answer(match.group(0))
            return self._norm_answer(boxed[-1])

        numbers = NUMBER_RE.findall(response)
        if numbers:
            return self._norm_answer(numbers[-1])

        if expect_letter:
            bare = re.findall(r"\b([A-E])\b", response, flags=re.IGNORECASE)
            if bare:
                return self._norm_answer(bare[-1])

        return None

    def score_detailed(self, message: str) -> HybridResult | None:
        question = self.extract_question(message)
        if question is None or question not in self.answers:
            return None

        gold = self._norm_answer(self.answers[question])
        prediction = self._extract_answer(self._extract_response(message), expect_letter=bool(re.fullmatch(r"[A-E]", gold)))
        score = 1.0 if prediction is not None and prediction == gold else 0.0
        return HybridResult(
            source="answer",
            score=score,
            detail={
                "gold": gold,
                "prediction": prediction,
                "matched": prediction == gold if prediction is not None else False,
            },
        )


class HybridRewardScorer:
    def __init__(
        self,
        answer_data: Path,
        code_dataset_dir: Path,
        test_suites: tuple[str, ...],
        max_tests: int,
        timeout: float,
        language: str,
        prefix_chars: int,
    ) -> None:
        self.answer_scorer = AnswerExactMatchScorer(answer_data)
        self.code_scorer = CodeContestScorer(
            dataset_dir=code_dataset_dir,
            test_suites=test_suites,
            max_tests=max_tests,
            timeout=timeout,
            language=language,
            prefix_chars=prefix_chars,
        )

    def score(self, message: str) -> float:
        return self.score_detailed(message).score

    def score_detailed(self, message: str) -> HybridResult:
        answer_result = self.answer_scorer.score_detailed(message)
        if answer_result is not None:
            return answer_result

        code_result = self.code_scorer.score_detailed(message)
        if code_result.error == "problem_not_found":
            return HybridResult(
                source="unmatched",
                score=0.0,
                detail={"error": code_result.error},
            )

        return HybridResult(
            source="code",
            score=code_result.score,
            detail=execution_result_to_dict(code_result),
        )


def execution_result_to_dict(result: ExecutionResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "total": result.total,
        "language": result.language,
        "error": result.error,
    }


def create_app(
    answer_data: Path,
    code_dataset_dir: Path,
    test_suites: tuple[str, ...],
    max_tests: int,
    timeout: float,
    language: str,
    prefix_chars: int,
) -> FastAPI:
    scorer = HybridRewardScorer(
        answer_data=answer_data,
        code_dataset_dir=code_dataset_dir,
        test_suites=test_suites,
        max_tests=max_tests,
        timeout=timeout,
        language=language,
        prefix_chars=prefix_chars,
    )
    app = FastAPI(title="Hybrid answer and code reward server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "answer_data": str(answer_data),
            "answer_examples": len(scorer.answer_scorer.answers),
            "code_dataset": str(code_dataset_dir),
            "code_problems": len(scorer.code_scorer.problems),
            "test_suites": list(test_suites),
            "max_tests": max_tests,
            "timeout": timeout,
            "language": language,
            "python": sys.executable,
            "g++": scorer.code_scorer.gpp_path,
            "warning": "Executes generated code locally. Use a stronger sandbox for untrusted or large runs.",
        }

    @app.post("/")
    def reward(payload: RewardRequest) -> dict[str, list[float]]:
        return {"scores": [scorer.score(message) for message in payload.messages]}

    @app.post("/debug")
    def reward_debug(payload: RewardRequest) -> dict[str, list[dict[str, Any]]]:
        results = []
        for message in payload.messages:
            result = scorer.score_detailed(message)
            results.append({"source": result.source, "score": result.score, **result.detail})
        return {"results": results}

    return app


app = FastAPI(title="Hybrid answer and code reward server")


@app.get("/health")
def unconfigured_health() -> dict[str, str]:
    return {"status": "not_configured", "hint": "Start this script directly so it can load reward data."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one reward endpoint for mixed answer and CodeContests data.")
    parser.add_argument(
        "--answer-data",
        type=Path,
        default=Path(os.environ.get("HYBRID_ANSWER_DATA", "data/mixed_agpo/mixed_all/train.jsonl")),
        help="JSON/JSONL data containing answer-bearing rows.",
    )
    parser.add_argument(
        "--code-dataset-dir",
        type=Path,
        default=Path(os.environ.get("CODE_REWARD_DATA", "data/code_contests/hf_dataset")),
        help="Hugging Face dataset directory with CodeContests tests.",
    )
    parser.add_argument("--test-suites", default=os.environ.get("CODE_REWARD_TEST_SUITES", "public,generated"))
    parser.add_argument("--max-tests", type=int, default=int(os.environ.get("CODE_REWARD_MAX_TESTS", "8")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("CODE_REWARD_TIMEOUT", "2.0")))
    parser.add_argument(
        "--language",
        choices=["auto", "python", "cpp"],
        default=os.environ.get("CODE_REWARD_LANGUAGE", "python").lower(),
    )
    parser.add_argument("--prefix-chars", type=int, default=512)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    global app
    app = create_app(
        answer_data=args.answer_data,
        code_dataset_dir=args.code_dataset_dir,
        test_suites=parse_test_suites(args.test_suites),
        max_tests=args.max_tests,
        timeout=args.timeout,
        language=args.language,
        prefix_chars=args.prefix_chars,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
