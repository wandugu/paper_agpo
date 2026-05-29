import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


USER_BLOCK_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
ASSISTANT_BLOCK_RE = re.compile(r"<\|im_start\|>assistant\n?(.*)", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
LETTER_RE = re.compile(
    r"(?:answer\s*(?:is|:)|choice\s*(?:is|:)|option\s*(?:is|:)|therefore|so)?\s*[\(\[]?([A-E])[\)\].]?",
    re.IGNORECASE,
)


class RewardRequest(BaseModel):
    model: str | None = None
    messages: list[str]


class AnswerScorer:
    def __init__(self, data_path: Path) -> None:
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
        if re.fullmatch(r"[A-Ea-e]", text):
            return text.upper()

        text = text.replace(",", "")
        text = text.replace("$", "")
        text = text.rstrip(".")
        try:
            return str(Decimal(text).normalize())
        except InvalidOperation:
            return re.sub(r"\s+", "", text).lower()

    def _extract_question(self, message: str) -> str | None:
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

    def _extract_answer(self, response: str) -> str | None:
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

        numbers = NUMBER_RE.findall(response)
        if numbers:
            return self._norm_answer(numbers[-1])
        return None

    def score(self, message: str) -> float:
        question = self._extract_question(message)
        if question is None or question not in self.answers:
            return 0.0

        prediction = self._extract_answer(self._extract_response(message))
        if prediction is None:
            return 0.0

        gold = self._norm_answer(self.answers[question])
        return 1.0 if prediction == gold else 0.0


def create_app(data_path: Path) -> FastAPI:
    scorer = AnswerScorer(data_path)
    app = FastAPI(title="Answer reward server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "examples": len(scorer.answers), "data": str(data_path)}

    @app.post("/")
    def reward(payload: RewardRequest) -> dict[str, list[float]]:
        return {"scores": [scorer.score(message) for message in payload.messages]}

    return app


default_data_path = Path(os.environ.get("ANSWER_REWARD_DATA", "data/gsm8k/rl_train.json"))
app = create_app(default_data_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_data_path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    global app
    app = create_app(args.data)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
