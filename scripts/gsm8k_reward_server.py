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


class RewardRequest(BaseModel):
    model: str | None = None
    messages: list[str]


class GSM8KScorer:
    def __init__(self, data_path: Path) -> None:
        rows = json.loads(data_path.read_text(encoding="utf-8"))
        self.answers = {self._norm_question(row["instruction"]): str(row["answer"]) for row in rows}
        self.fallback_items = list(self.answers.items())

    @staticmethod
    def _norm_question(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _norm_answer(text: str) -> str:
        text = text.strip()
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
    scorer = GSM8KScorer(data_path)
    app = FastAPI(title="GSM8K reward server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "examples": len(scorer.answers)}

    @app.post("/")
    def reward(payload: RewardRequest) -> dict[str, list[float]]:
        return {"scores": [scorer.score(message) for message in payload.messages]}

    return app


default_data_path = Path(os.environ.get("GSM8K_REWARD_DATA", "data/gsm8k/rl_train.json"))
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
