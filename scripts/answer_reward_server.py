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

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:  # pragma: no cover - scorer still works without symbolic parsing.
    sp = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None


USER_BLOCK_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
ASSISTANT_BLOCK_RE = re.compile(r"<\|im_start\|>assistant\n?(.*)", re.DOTALL)
FRACTION_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*/\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
LATEX_FRACTION_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}")
NUMBER_RE = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
SYMBOLIC_HINT_RE = re.compile(r"(\\(?:sqrt|frac|dfrac|tfrac|pi)|\b(?:sqrt|pi)\b|\^|π)")
FINAL_ANSWER_PREFIX_RE = re.compile(
    r"(?:####|final\s+answer\s*(?:is|=|:)?|the\s+answer\s*(?:is|=|:)?|answer\s*(?:is|=|:)?|"
    r"\u7b54\u6848\s*(?:\u662f|=|:|\uff1a)?)",
    re.IGNORECASE,
)
BARE_LETTER_RE = re.compile(r"\b([A-E])\b")
LETTER_ANSWER_RE = re.compile(
    r"(?:answer\s*(?:is|=|:)?|choice\s*(?:is|=|:)?|choose|select|"
    r"therefore(?:\s+(?:the\s+)?(?:answer|choice|option)\s*(?:is|=|:)?)?|"
    r"so(?:\s+that(?:'s|\s+is))?(?:\s+(?:the\s+)?(?:answer|choice|option)\s*(?:is|=|:)?)?|"
    r"\u7b54\u6848\s*(?:\u662f|=|:|\uff1a)?|\u9009\u9879\s*(?:\u662f|=|:|\uff1a)?|"
    r"\u6545\u9009|\u9009\u62e9)\s*[\(\[]?([A-E])[\)\].]?",
    re.IGNORECASE,
)
BARE_LETTER_LINE_RE = re.compile(r"^\s*(?:[\(\[]?([A-E])[\)\].]?|[A-E]\s*[、.．])\s*$", re.IGNORECASE)
EXPANDED_OPTION_RE = re.compile(
    r"(?:^|\s)([A-E])\.\s*(.*?)(?=\s+[A-E]\.\s*|\s+(?:Answer\b|\u7b54\u6848)|\Z)",
    re.DOTALL,
)
INLINE_OPTION_RE = re.compile(
    r"[\(（]([A-E])[\)）]\s*(.*?)(?=[\(（][A-E][\)）]|\s+(?:Answer\b|\u7b54\u6848)|\Z)",
    re.DOTALL,
)
OPTION_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:[\(\[\{\uff08]\s*)?[A-Ea-e]\s*(?:[\)\]\}\uff09]|[.:：．、])?\s*"
)


class RewardRequest(BaseModel):
    model: str | None = None
    messages: list[str]


class AnswerScorer:
    dense_numeric_near_rel_error = Decimal("0.05")
    dense_numeric_mid_rel_error = Decimal("0.10")
    dense_numeric_far_rel_error = Decimal("0.20")
    dense_numeric_near_score = Decimal("0.8")
    dense_numeric_mid_score = Decimal("0.5")
    dense_numeric_far_score = Decimal("0.25")
    dense_numeric_format_score = Decimal("0.0")
    dense_letter_format_score = Decimal("0.05")
    symbolic_transformations = (
        standard_transformations + (implicit_multiplication_application, convert_xor)
        if parse_expr is not None and implicit_multiplication_application is not None and convert_xor is not None
        else ()
    )

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
    def _canonical_letter_answer(text: str) -> str | None:
        raw = text.strip()
        raw = re.sub(
            r"^(?:answer|option|choice|final\s+answer)\s*(?:is|=|:)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        raw = raw.strip().strip("()[]{}")
        has_separator = bool(re.search(r"[\s,;/|&+\-.:]|[\u3001\uff0c\uff1b\uff1a\u548c\u53ca\u4e0e]", raw))
        compact = re.sub(r"[\s,;/|&+\-.:]|[\u3001\uff0c\uff1b\uff1a\u548c\u53ca\u4e0e]", "", raw)
        if not compact:
            return None
        if len(compact) == 1 or has_separator or raw == raw.upper():
            if re.fullmatch(r"[A-Ea-e]{1,5}", compact):
                return "".join(sorted(set(compact.upper())))
        return None

    @staticmethod
    def _norm_answer(text: str) -> str:
        text = text.strip()
        boxed = AnswerScorer._boxed_values(text)
        if len(boxed) == 1 and text.startswith("\\boxed"):
            text = boxed[0].strip()

        latex_fraction = LATEX_FRACTION_RE.fullmatch(text)
        if latex_fraction:
            text = f"{latex_fraction.group(1)}/{latex_fraction.group(2)}"

        if re.fullmatch(r"[A-Ea-e]", text):
            return text.upper()

        letter_answer = AnswerScorer._canonical_letter_answer(text)
        if letter_answer is not None:
            return letter_answer

        text = text.replace(",", "")
        text = text.replace("$", "")
        text = text.rstrip(".")
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*/\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text):
            numerator, denominator = re.split(r"\s*/\s*", text, maxsplit=1)
            try:
                denominator_value = Decimal(denominator)
                if denominator_value != 0:
                    return str((Decimal(numerator) / denominator_value).normalize())
            except InvalidOperation:
                pass

        try:
            return str(Decimal(text).normalize())
        except InvalidOperation:
            return re.sub(r"\s+", "", text).lower()

    @staticmethod
    def _boxed_values(text: str) -> list[str]:
        values = []
        marker = "\\boxed{"
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                return values

            pos = idx + len(marker)
            depth = 1
            end = pos
            while end < len(text) and depth:
                char = text[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                end += 1

            if depth == 0:
                values.append(text[pos : end - 1].strip())
            start = idx + len(marker)

    @staticmethod
    def _read_braced(text: str, open_index: int) -> tuple[str, int] | None:
        if open_index >= len(text) or text[open_index] != "{":
            return None

        pos = open_index + 1
        depth = 1
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1

        if depth != 0:
            return None
        return text[open_index + 1 : pos - 1], pos

    @classmethod
    def _replace_latex_fractions(cls, text: str) -> str:
        pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\{")
        start = 0
        pieces: list[str] = []
        while True:
            match = pattern.search(text, start)
            if match is None:
                pieces.append(text[start:])
                return "".join(pieces)

            first_open = match.end() - 1
            first = cls._read_braced(text, first_open)
            if first is None:
                pieces.append(text[start : match.end()])
                start = match.end()
                continue

            second_open = first[1]
            while second_open < len(text) and text[second_open].isspace():
                second_open += 1
            second = cls._read_braced(text, second_open)
            if second is None:
                pieces.append(text[start : first[1]])
                start = first[1]
                continue

            numerator = cls._latex_to_sympy_text(first[0])
            denominator = cls._latex_to_sympy_text(second[0])
            pieces.append(text[start : match.start()])
            pieces.append(f"(({numerator})/({denominator}))")
            start = second[1]

    @classmethod
    def _replace_latex_sqrts(cls, text: str) -> str:
        marker = "\\sqrt{"
        start = 0
        pieces: list[str] = []
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                pieces.append(text[start:])
                return "".join(pieces)

            open_index = idx + len("\\sqrt")
            braced = cls._read_braced(text, open_index)
            if braced is None:
                pieces.append(text[start : idx + len(marker)])
                start = idx + len(marker)
                continue

            radicand = cls._latex_to_sympy_text(braced[0])
            pieces.append(text[start:idx])
            pieces.append(f"sqrt({radicand})")
            start = braced[1]

    @classmethod
    def _latex_to_sympy_text(cls, text: str) -> str:
        text = text.strip()
        for boxed in cls._boxed_values(text):
            text = text.replace(f"\\boxed{{{boxed}}}", boxed)
        text = cls._replace_latex_fractions(text)
        text = cls._replace_latex_sqrts(text)
        text = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", text)
        text = text.replace("\\left", "").replace("\\right", "")
        text = text.replace("\\cdot", "*").replace("\\times", "*")
        text = text.replace("\\pi", "pi").replace("π", "pi")
        text = text.replace("^", "**")
        text = text.replace("{", "(").replace("}", ")")
        text = text.replace("$", "").replace(",", "")
        text = re.sub(r"\\[,;:! ]", "", text)
        return text.strip()

    @classmethod
    def _symbolic_decimal(cls, text: str) -> Decimal | None:
        if parse_expr is None or sp is None:
            return None
        if not SYMBOLIC_HINT_RE.search(text):
            return None

        expr_text = cls._latex_to_sympy_text(text)
        if not expr_text or len(expr_text) > 160:
            return None

        try:
            expr = parse_expr(
                expr_text,
                local_dict={"sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E, "e": sp.E},
                transformations=cls.symbolic_transformations,
                evaluate=True,
            )
        except Exception:
            return None

        if getattr(expr, "free_symbols", None):
            return None
        if not bool(expr.is_real):
            return None

        try:
            return Decimal(str(sp.N(expr, 40)))
        except (InvalidOperation, ValueError):
            return None

    @staticmethod
    def _to_decimal(text: str) -> Decimal | None:
        if re.fullmatch(r"[A-E]", text):
            return None

        try:
            return Decimal(text)
        except InvalidOperation:
            return AnswerScorer._symbolic_decimal(text)

    @classmethod
    def _numeric_similarity(cls, gold: str, prediction: str) -> float:
        gold_value = cls._to_decimal(gold)
        prediction_value = cls._to_decimal(prediction)
        if gold_value is None or prediction_value is None:
            return 0.0

        absolute_error = abs(prediction_value - gold_value)
        if absolute_error == 0:
            return 1.0
        if gold_value == 0:
            return float(cls.dense_numeric_format_score)

        relative_error = absolute_error / abs(gold_value)
        if relative_error <= cls.dense_numeric_near_rel_error:
            return float(cls.dense_numeric_near_score)
        if relative_error <= cls.dense_numeric_mid_rel_error:
            return float(cls.dense_numeric_mid_score)
        if relative_error <= cls.dense_numeric_far_rel_error:
            return float(cls.dense_numeric_far_score)

        return float(cls.dense_numeric_format_score)

    @staticmethod
    def _split_answer_components(text: str) -> list[str]:
        parts = [part.strip() for part in re.split(r"[;；]", text) if part.strip()]
        return parts if len(parts) > 1 else []

    @classmethod
    def _component_matches(cls, gold: str, prediction: str) -> bool:
        if prediction == gold:
            return True
        return cls._numeric_similarity(gold, prediction) >= float(cls.dense_numeric_near_score)

    @classmethod
    def _multi_answer_similarity(cls, gold: str, prediction: str) -> float:
        gold_parts = cls._split_answer_components(gold)
        if not gold_parts:
            return 0.0

        prediction_parts = cls._split_answer_components(prediction) or [prediction]
        unused_predictions = prediction_parts[:]
        matches = 0
        for gold_part in gold_parts:
            for index, prediction_part in enumerate(unused_predictions):
                if cls._component_matches(gold_part, prediction_part):
                    matches += 1
                    unused_predictions.pop(index)
                    break

        return matches / len(gold_parts) if matches else 0.0

    @classmethod
    def _extract_symbolic_candidate(cls, text: str, prefer_first: bool) -> str | None:
        candidates: list[str] = []
        candidates.extend(match.group(1) for match in re.finditer(r"\$([^$\n]{1,160})\$", text))
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if lines:
            line = lines[0] if prefer_first else lines[-1]
            fragment = re.split(r"(?:[\u3002\uff1b;]|\s+because\b|\s+since\b|\s+therefore\b|\s+so\b)", line, 1)[0]
            candidates.extend([fragment, line])

        ordered = candidates if prefer_first else list(reversed(candidates))
        for candidate in ordered:
            candidate = candidate.strip().strip(" .,:;\u3002\uff0c\uff1b\uff1a")
            if not candidate or not SYMBOLIC_HINT_RE.search(candidate):
                continue
            normalised = cls._norm_answer(candidate)
            if cls._to_decimal(normalised) is not None:
                return normalised
        return None

    @classmethod
    def _extract_multi_answer_candidate(cls, text: str) -> str | None:
        for line in text.strip().splitlines()[:3]:
            candidate = cls._norm_answer(line.strip().strip("。"))
            if cls._split_answer_components(candidate):
                return candidate
        return None

    def _extract_question(self, message: str) -> str | None:
        matches = USER_BLOCK_RE.findall(message)
        if matches:
            question = self._norm_question(matches[-1])
            if question in self.answers:
                return question

            for known_question, _ in self.fallback_items:
                if known_question in question:
                    return known_question
            return question

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

    def _extract_candidate(self, text: str, expect_letter: bool, prefer_first: bool = False) -> str | None:
        boxed = self._boxed_values(text)
        if boxed:
            boxed_value = boxed[0] if prefer_first else boxed[-1]
            candidate = self._extract_candidate(boxed_value, expect_letter=expect_letter, prefer_first=prefer_first)
            if candidate is not None:
                return candidate
            return self._norm_answer(boxed_value)

        latex_fractions = LATEX_FRACTION_RE.findall(text)
        if latex_fractions:
            numerator, denominator = latex_fractions[0] if prefer_first else latex_fractions[-1]
            return self._norm_answer(f"{numerator}/{denominator}")

        fractions = FRACTION_RE.findall(text)
        if fractions:
            return self._norm_answer(fractions[0] if prefer_first else fractions[-1])

        symbolic_candidate = self._extract_symbolic_candidate(text, prefer_first=prefer_first)
        if symbolic_candidate is not None:
            return symbolic_candidate

        numbers = NUMBER_RE.findall(text)
        if numbers:
            return self._norm_answer(numbers[0] if prefer_first else numbers[-1])

        if expect_letter:
            lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
            if lines:
                line = lines[0] if prefer_first else lines[-1]
                letter_answer = self._canonical_letter_answer(line)
                if letter_answer is not None:
                    return letter_answer

            bare = BARE_LETTER_RE.findall(text)
            if bare:
                return self._norm_answer(bare[0] if prefer_first else bare[-1])

        return None

    def _extract_scalar_option_value(self, text: str) -> str | None:
        text = re.sub(r"^\s*[\(（]?[A-E][\)）.]?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = OPTION_LABEL_PREFIX_RE.sub("", text, count=1)
        boxed = self._boxed_values(text)
        if len(boxed) == 1:
            candidate = self._extract_scalar_option_value(boxed[0])
            if candidate is not None:
                return candidate

        latex_fractions = LATEX_FRACTION_RE.findall(text)
        if len(latex_fractions) == 1:
            numerator, denominator = latex_fractions[0]
            return self._norm_answer(f"{numerator}/{denominator}")

        text_without_latex_fractions = LATEX_FRACTION_RE.sub(" ", text)
        fractions = FRACTION_RE.findall(text_without_latex_fractions)
        if len(fractions) == 1:
            return self._norm_answer(fractions[0])

        text_without_fractions = FRACTION_RE.sub(" ", text_without_latex_fractions)
        numbers = NUMBER_RE.findall(text_without_fractions)
        if len(numbers) == 1:
            return self._norm_answer(numbers[0])

        return None

    def _option_values(self, question: str) -> dict[str, str]:
        for pattern in (EXPANDED_OPTION_RE, INLINE_OPTION_RE):
            values: dict[str, str] = {}
            for letter, option_text in pattern.findall(question):
                value = self._extract_scalar_option_value(option_text)
                if value is not None:
                    values[self._norm_answer(letter)] = value
            if values:
                return values

        return {}

    @staticmethod
    def _normalise_option_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^\s*[\(锛圿?[A-E][\)锛?]?\s*", "", text, flags=re.IGNORECASE)
        text = OPTION_LABEL_PREFIX_RE.sub("", text, count=1)
        for boxed in AnswerScorer._boxed_values(text):
            text = text.replace(f"\\boxed{{{boxed}}}", boxed)
        text = re.sub(r"\\textcircled\{([^{}]+)\}", r"\1", text)
        text = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", text)
        text = text.translate({ord(chr(0x2460 + index)): str(index + 1) for index in range(20)})
        text = text.replace("\\$", "").replace("$", "")
        text = re.sub(r"\\(?:left|right|,|;|:|!|\s)+", "", text)
        text = text.replace("\\", "")
        text = re.sub(r"\band\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[\s,;/|&+\-.:`'\"$]|[\u3001\u3002\uff0c\uff1b\uff1a\uff08\uff09\u548c\u53ca\u4e0e]", "", text)
        text = re.sub(r"[\(\)\[\]\{\}]", "", text)
        return text.lower()

    def _option_text_values(self, question: str) -> dict[str, str]:
        for pattern in (EXPANDED_OPTION_RE, INLINE_OPTION_RE):
            values: dict[str, str] = {}
            for letter, option_text in pattern.findall(question):
                value = self._normalise_option_text(option_text)
                if value:
                    values[self._norm_answer(letter)] = value
            if values:
                return values
        return {}

    def _option_letter_for_prediction(self, question: str, prediction: str) -> str | None:
        raw_prediction = prediction
        prediction = self._norm_answer(prediction)
        for letter, option_value in self._option_values(question).items():
            if prediction == option_value:
                return letter

        text_prediction = self._normalise_option_text(raw_prediction)
        if text_prediction:
            for letter, option_value in self._option_text_values(question).items():
                if text_prediction == option_value:
                    return letter
        return None

    def _answers_match(self, question: str, gold: str, prediction: str) -> bool:
        if prediction == gold:
            return True
        gold_letters = self._canonical_letter_answer(gold)
        if gold_letters is not None:
            prediction_letters = self._canonical_letter_answer(prediction)
            if prediction_letters is not None:
                return prediction_letters == gold_letters
            return self._option_letter_for_prediction(question, prediction) == gold_letters
        return self._numeric_similarity(gold, prediction) == 1.0

    def _letter_similarity(self, question: str, gold: str, prediction: str) -> float:
        gold_letters = self._canonical_letter_answer(gold)
        if gold_letters is None:
            return 0.0

        prediction_letters = self._canonical_letter_answer(prediction)
        if prediction_letters is None:
            return 0.0
        if prediction_letters == gold_letters:
            return 1.0

        gold_set = set(gold_letters)
        prediction_set = set(prediction_letters)
        overlap = gold_set & prediction_set
        if not overlap:
            return float(self.dense_letter_format_score)
        return max(float(self.dense_letter_format_score), len(overlap) / len(gold_set | prediction_set))

    def _extract_option_text_candidate(self, text: str, question: str | None) -> str | None:
        if question is None:
            return None

        snippets: list[str] = []
        for line in [line.strip() for line in text.strip().splitlines() if line.strip()][:3]:
            fragment = re.split(
                r"(?:\s+because\b|\s+since\b|\s+therefore\b|\s+so\b|[\u3002\uff1b;])",
                line,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            snippets.extend([line, fragment])

        for snippet in snippets:
            snippet = snippet.strip().strip(" .;:\u3002\uff1b\uff1a")
            if not snippet or len(snippet) > 160:
                continue
            letter_answer = self._canonical_letter_answer(snippet)
            if letter_answer is not None:
                return letter_answer
            option_letter = self._option_letter_for_prediction(question, snippet)
            if option_letter is not None:
                return snippet
        return None

    def _extract_final_numeric_candidate(self, text: str) -> str | None:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) <= 1:
            return None

        answer_markers = re.compile(
            r"(^[-*•]\s*)|(\banswer\b|\bso\b|\btherefore\b|\bby\b|=|\u7b54\u6848|\u6240\u4ee5|\u56e0\u6b64)",
            re.IGNORECASE,
        )
        for line in reversed(lines[-6:]):
            if len(line) > 240 or not answer_markers.search(line):
                continue
            candidate = self._extract_candidate(line, expect_letter=False, prefer_first=False)
            if candidate is not None:
                return candidate
        return None

    def _extract_answer(self, response: str, expect_letter: bool, question: str | None = None) -> str | None:
        prefix_matches = list(FINAL_ANSWER_PREFIX_RE.finditer(response))
        for match in reversed(prefix_matches):
            if not expect_letter:
                multi_candidate = self._extract_multi_answer_candidate(response[match.end() :])
                if multi_candidate is not None:
                    return multi_candidate
                final_candidate = self._extract_final_numeric_candidate(response[match.end() :])
                if final_candidate is not None:
                    return final_candidate
            else:
                option_text_candidate = self._extract_option_text_candidate(response[match.end() :], question)
                if option_text_candidate is not None:
                    return option_text_candidate

            candidate = self._extract_candidate(response[match.end() :], expect_letter=expect_letter, prefer_first=True)
            if candidate is not None:
                return candidate

        if expect_letter:
            letter_matches = LETTER_ANSWER_RE.findall(response)
            if letter_matches:
                return self._norm_answer(letter_matches[-1])

            lines = [line.strip() for line in response[-400:].splitlines() if line.strip()]
            for line in reversed(lines[-5:]):
                bare_line = BARE_LETTER_LINE_RE.fullmatch(line)
                if bare_line:
                    return self._norm_answer(bare_line.group(1) or line[0])

            compact_tail = response.strip()
            if len(compact_tail) <= 80:
                tail_letters = BARE_LETTER_RE.findall(compact_tail)
                if tail_letters:
                    return self._norm_answer(tail_letters[-1])

            option_text_candidate = self._extract_option_text_candidate(response[-400:], question)
            if option_text_candidate is not None:
                return option_text_candidate

        if not expect_letter:
            multi_candidate = self._extract_multi_answer_candidate(response)
            if multi_candidate is not None:
                return multi_candidate

        return self._extract_candidate(response, expect_letter=expect_letter)

    def score(self, message: str) -> float:
        question = self._extract_question(message)
        if question is None or question not in self.answers:
            return 0.0

        gold = self._norm_answer(self.answers[question])
        expect_letter = self._canonical_letter_answer(gold) is not None
        prediction = self._extract_answer(self._extract_response(message), expect_letter=expect_letter, question=question)
        if prediction is None:
            return 0.0

        if self._answers_match(question, gold, prediction):
            return 1.0

        if expect_letter:
            return self._letter_similarity(question, gold, prediction)

        multi_score = self._multi_answer_similarity(gold, prediction)
        if multi_score > 0.0:
            return multi_score

        return self._numeric_similarity(gold, prediction)


def create_app(data_path: Path) -> FastAPI:
    scorer = AnswerScorer(data_path)
    app = FastAPI(title="Answer reward server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "examples": len(scorer.answers),
            "data": str(data_path),
            "dense_numeric_near_rel_error": str(scorer.dense_numeric_near_rel_error),
            "dense_numeric_mid_rel_error": str(scorer.dense_numeric_mid_rel_error),
            "dense_numeric_far_rel_error": str(scorer.dense_numeric_far_rel_error),
            "dense_numeric_near_score": str(scorer.dense_numeric_near_score),
            "dense_numeric_mid_score": str(scorer.dense_numeric_mid_score),
            "dense_numeric_far_score": str(scorer.dense_numeric_far_score),
            "dense_numeric_format_score": str(scorer.dense_numeric_format_score),
            "dense_letter_format_score": str(scorer.dense_letter_format_score),
        }

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
