import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from answer_reward_server import AnswerScorer  # noqa: E402
from hybrid_reward_server import AnswerExactMatchScorer  # noqa: E402


CHOICE_SCALAR_Q = (
    "choice scalar q\n\n"
    "A. 1\n"
    "B. 2\n"
    "C. 3\n"
    "D. 4\n\n"
    "Answer with the correct option letter only."
)
CHOICE_NONSCALAR_Q = (
    "choice nonscalar q\n\n"
    "A. x_1 < x_5\n"
    "B. x_2 < x_8\n"
    "C. x_6 < x_2\n"
    "D. x_4 < x_7\n\n"
    "Answer with the correct option letter only."
)
CHOICE_COMPOUND_Q = (
    "choice compound q\n\n"
    "A. \\textcircled{1}\\textcircled{2}\\textcircled{4}\n"
    "B. \\textcircled{2}\\textcircled{4}\n"
    "C. \\textcircled{1}\\textcircled{4}\n"
    "D. \\textcircled{1}\\textcircled{3}\n\n"
    "Answer with the correct option letter only."
)
CHOICE_INLINE_Q = (
    "Q: Which equation matches the line? "
    "Answer Choices: (A)$y=2x+3$ (B)$y=2x+4$ (C)$y=\\frac{3}{2}x+3$ (D)$y=\\frac{3}{2}x+1$ "
    "Answer with the correct option letter only."
)


def _message(question: str, response: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


def _answer_scorer() -> AnswerScorer:
    scorer = AnswerScorer.__new__(AnswerScorer)
    scorer.answers = {
        "numeric q": "5",
        "fraction q": "9/2",
        "scientific q": "600",
        "sqrt q": "\\sqrt{3}",
        "sqrt fraction q": "\\frac{2\\sqrt{3}}{3}",
        "pi q": "164\\pi",
        "small q": "\\dfrac{1}{221}",
        "zero q": "0",
        "letter q": "B",
        "multi letter q": "ACD",
        "multi q": "5;720",
        "symbolic multi q": "\\frac{1}{3};\\sqrt{3}",
        AnswerScorer._norm_question(CHOICE_SCALAR_Q): "B",
        AnswerScorer._norm_question(CHOICE_NONSCALAR_Q): "D",
        AnswerScorer._norm_question(CHOICE_COMPOUND_Q): "C",
        AnswerScorer._norm_question(CHOICE_INLINE_Q): "D",
    }
    scorer.fallback_items = list(scorer.answers.items())
    return scorer


def _hybrid_answer_scorer() -> AnswerExactMatchScorer:
    scorer = AnswerExactMatchScorer.__new__(AnswerExactMatchScorer)
    scorer.answers = {
        "numeric q": "5",
        "fraction q": "9/2",
        "scientific q": "600",
        "sqrt q": "\\sqrt{3}",
        "sqrt fraction q": "\\frac{2\\sqrt{3}}{3}",
        "pi q": "164\\pi",
        "small q": "\\dfrac{1}{221}",
        "zero q": "0",
        "letter q": "B",
        "multi letter q": "ACD",
        "multi q": "5;720",
        "symbolic multi q": "\\frac{1}{3};\\sqrt{3}",
        AnswerExactMatchScorer._norm_question(CHOICE_SCALAR_Q): "B",
        AnswerExactMatchScorer._norm_question(CHOICE_NONSCALAR_Q): "D",
        AnswerExactMatchScorer._norm_question(CHOICE_COMPOUND_Q): "C",
        AnswerExactMatchScorer._norm_question(CHOICE_INLINE_Q): "D",
    }
    scorer.fallback_items = list(scorer.answers.items())
    return scorer


def test_answer_scorer_ignores_letters_for_numeric_answers() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("numeric q", "I choose A while computing, but the final answer is 5.")) == 1.0
    assert scorer.score(_message("numeric q", "The final answer is 4.99.")) == 0.8
    assert scorer.score(_message("numeric q", "The final answer is 4.6.")) == 0.5
    assert scorer.score(_message("numeric q", "The final answer is 4.")) == 0.25
    assert scorer.score(_message("numeric q", "The final answer is 1.")) == 0.0


def test_answer_scorer_prioritizes_final_answer_markers() -> None:
    scorer = _answer_scorer()

    response = "We tried 3 and 4. The final answer is 5. Earlier scratch value 100 is irrelevant."
    assert scorer.score(_message("numeric q", response)) == 1.0


def test_answer_scorer_handles_multiline_final_answer_sections() -> None:
    scorer = _answer_scorer()

    response = (
        "Work shows William has 22 and Brad has 26.\n"
        "### Final Answer:\n"
        "- William has read 22 books across the two months.\n"
        "- Brad has read 26 books across the two months.\n"
        "So, Brad has read more by 4 books."
    )
    assert scorer.score(_message("numeric q", response)) == 0.25

    scorer.answers["numeric q"] = "4"
    assert scorer.score(_message("numeric q", response)) == 1.0
    scorer.answers["numeric q"] = "5"


def test_answer_scorer_matches_questions_with_generation_suffix() -> None:
    scorer = _answer_scorer()

    question = "numeric q\n\nReturn only the final answer in the form `Answer: <answer>`."
    assert scorer.score(_message(question, "Answer: 5")) == 1.0


def test_answer_scorer_handles_fraction_formats() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("fraction q", "So the answer is 9/2.")) == 1.0
    assert scorer.score(_message("fraction q", "So the answer is 4.5.")) == 1.0
    assert scorer.score(_message("fraction q", "Therefore, \\boxed{\\frac{9}{2}}.")) == 1.0
    assert scorer.score(_message("fraction q", "Wrong trail 10. Final answer: \\boxed{\\frac{9}{2}}.")) == 1.0


def test_answer_scorer_handles_scientific_notation() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("scientific q", "Final answer: 6E+2.")) == 1.0
    assert scorer.score(_message("scientific q", "Final answer: 6.1e2.")) == 0.8


def test_answer_scorer_handles_symbolic_numeric_answers() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("sqrt q", "Final answer: \\sqrt{3}.")) == 1.0
    assert scorer.score(_message("sqrt q", "Final answer: 1.7320508075688772.")) == 0.8
    assert scorer.score(_message("sqrt fraction q", "Final answer: \\frac{2\\sqrt{3}}{3}.")) == 1.0
    assert scorer.score(_message("sqrt fraction q", "Final answer: 1.1547005383792515.")) == 0.8
    assert scorer.score(_message("pi q", "Final answer: 164\\pi.")) == 1.0
    assert scorer.score(_message("symbolic multi q", "Final answer: 0.333333333333;1.7320508075688772")) == 1.0


def test_answer_scorer_uses_relative_error_for_small_numeric_answers() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("small q", "Final answer: 0.0046.")) == 0.8
    assert scorer.score(_message("small q", "Final answer: \\frac{1}{26!}.")) == 0.0
    assert scorer.score(_message("zero q", "Final answer: 0.")) == 1.0
    assert scorer.score(_message("zero q", "Final answer: 0.01.")) == 0.0


def test_answer_scorer_uses_letters_only_for_letter_answers() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("letter q", "After checking, the answer is B.")) == 1.0
    assert scorer.score(_message("letter q", "15*12*5*21 = 18900, so that's option B.")) == 1.0
    assert scorer.score(_message("letter q", "The computation gives 18900. B")) == 1.0
    assert scorer.score(_message("letter q", "After checking, the answer is C.")) == 0.05
    assert scorer.score(_message("letter q", "The final answer is 18.")) == 0.0


def test_answer_scorer_does_not_treat_option_explanations_as_final_letters() -> None:
    scorer = _answer_scorer()

    response = "Option D: plausible but wrong.\nOption B: correct.\n\nFinal Answer: B"
    assert scorer.score(_message("letter q", response)) == 1.0

    scorer.answers["letter q"] = "D"
    assert scorer.score(_message("letter q", response)) == 0.05


def test_answer_scorer_accepts_correct_scalar_option_content() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message(CHOICE_SCALAR_Q, "The final answer is 2.")) == 1.0
    assert scorer.score(_message(CHOICE_SCALAR_Q, "The final answer is 3.")) == 0.0
    assert scorer.score(_message(CHOICE_NONSCALAR_Q, "The final answer is 7.")) == 0.0


def test_answer_scorer_accepts_compound_option_content() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message(CHOICE_COMPOUND_Q, "Final answer: \\textcircled{1}\\textcircled{4}")) == 1.0
    assert scorer.score(_message(CHOICE_COMPOUND_Q, "Final answer: 1和4")) == 1.0
    assert scorer.score(_message(CHOICE_COMPOUND_Q, "Final answer: 1 and 4")) == 1.0
    assert scorer.score(_message(CHOICE_COMPOUND_Q, "Final answer: 1和3")) == 0.0


def test_answer_scorer_accepts_inline_option_content() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message(CHOICE_INLINE_Q, "Final answer: $y=\\frac{3}{2}x+1$")) == 1.0
    assert scorer.score(_message(CHOICE_INLINE_Q, "Final answer: y=2x+3")) == 0.0


def test_answer_scorer_handles_multi_letter_answers() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("multi letter q", "Final answer: ACD")) == 1.0
    assert scorer.score(_message("multi letter q", "Final answer: A, C, D")) == 1.0
    partial = scorer.score(_message("multi letter q", "Final answer: A, C"))
    assert 0.05 < partial < 1.0
    assert scorer.score(_message("multi letter q", "Final answer: B")) == 0.05


def test_answer_scorer_gives_partial_reward_for_semicolon_answers() -> None:
    scorer = _answer_scorer()

    assert scorer.score(_message("multi q", "Final answer: 5;720")) == 1.0
    assert scorer.score(_message("multi q", "Final answer: 5")) == 0.5
    assert scorer.score(_message("multi q", "Final answer: 720")) == 0.5
    assert scorer.score(_message("multi q", "Final answer: 4")) == 0.0


def test_hybrid_answer_scorer_matches_answer_server_numeric_logic() -> None:
    scorer = _hybrid_answer_scorer()

    numeric = scorer.score_detailed(_message("numeric q", "I choose A while computing, but the final answer is 5."))
    near_numeric = scorer.score_detailed(_message("numeric q", "The final answer is 4.99."))
    fraction = scorer.score_detailed(_message("fraction q", "Therefore, \\boxed{\\frac{9}{2}}."))
    scientific = scorer.score_detailed(_message("scientific q", "Final answer: 6E+2."))
    letter = scorer.score_detailed(_message("letter q", "15*12*5*21 = 18900, so that's option B."))
    wrong_letter = scorer.score_detailed(_message("letter q", "After checking, the answer is C."))
    invalid_letter = scorer.score_detailed(_message("letter q", "The final answer is 18."))

    assert numeric is not None and numeric.score == 1.0
    assert near_numeric is not None and near_numeric.score == 0.8
    assert fraction is not None and fraction.score == 1.0
    assert scientific is not None and scientific.score == 1.0
    assert letter is not None and letter.score == 1.0
    assert wrong_letter is not None and wrong_letter.score == 0.05
    assert invalid_letter is not None and invalid_letter.score == 0.0


def test_hybrid_answer_scorer_handles_multiline_final_answer_sections() -> None:
    scorer = _hybrid_answer_scorer()
    scorer.answers["numeric q"] = "4"

    response = (
        "Work shows William has 22 and Brad has 26.\n"
        "### Final Answer:\n"
        "- William has read 22 books across the two months.\n"
        "- Brad has read 26 books across the two months.\n"
        "So, Brad has read more by 4 books."
    )
    detail = scorer.score_detailed(_message("numeric q", response))
    assert detail is not None and detail.score == 1.0


def test_hybrid_answer_scorer_accepts_correct_scalar_option_content() -> None:
    scorer = _hybrid_answer_scorer()

    scalar = scorer.score_detailed(_message(CHOICE_SCALAR_Q, "The final answer is 2."))
    wrong_scalar = scorer.score_detailed(_message(CHOICE_SCALAR_Q, "The final answer is 3."))
    nonscalar = scorer.score_detailed(_message(CHOICE_NONSCALAR_Q, "The final answer is 7."))

    assert scalar is not None and scalar.score == 1.0
    assert wrong_scalar is not None and wrong_scalar.score == 0.0
    assert nonscalar is not None and nonscalar.score == 0.0


def test_hybrid_answer_scorer_accepts_compound_option_content() -> None:
    scorer = _hybrid_answer_scorer()

    latex = scorer.score_detailed(_message(CHOICE_COMPOUND_Q, "Final answer: \\textcircled{1}\\textcircled{4}"))
    chinese = scorer.score_detailed(_message(CHOICE_COMPOUND_Q, "Final answer: 1和4"))
    english = scorer.score_detailed(_message(CHOICE_COMPOUND_Q, "Final answer: 1 and 4"))
    wrong = scorer.score_detailed(_message(CHOICE_COMPOUND_Q, "Final answer: 1和3"))

    assert latex is not None and latex.score == 1.0
    assert chinese is not None and chinese.score == 1.0
    assert english is not None and english.score == 1.0
    assert wrong is not None and wrong.score == 0.0


def test_hybrid_answer_scorer_accepts_inline_option_content() -> None:
    scorer = _hybrid_answer_scorer()

    correct = scorer.score_detailed(_message(CHOICE_INLINE_Q, "Final answer: $y=\\frac{3}{2}x+1$"))
    wrong = scorer.score_detailed(_message(CHOICE_INLINE_Q, "Final answer: y=2x+3"))

    assert correct is not None and correct.score == 1.0
    assert wrong is not None and wrong.score == 0.0


def test_hybrid_answer_scorer_handles_multi_letter_answers() -> None:
    scorer = _hybrid_answer_scorer()

    compact = scorer.score_detailed(_message("multi letter q", "Final answer: ACD"))
    spaced = scorer.score_detailed(_message("multi letter q", "Final answer: A, C, D"))
    partial = scorer.score_detailed(_message("multi letter q", "Final answer: A, C"))
    wrong = scorer.score_detailed(_message("multi letter q", "Final answer: B"))

    assert compact is not None and compact.score == 1.0
    assert spaced is not None and spaced.score == 1.0
    assert partial is not None and 0.05 < partial.score < 1.0
    assert wrong is not None and wrong.score == 0.05


def test_hybrid_answer_scorer_gives_partial_reward_for_semicolon_answers() -> None:
    scorer = _hybrid_answer_scorer()

    full = scorer.score_detailed(_message("multi q", "Final answer: 5;720"))
    first = scorer.score_detailed(_message("multi q", "Final answer: 5"))
    second = scorer.score_detailed(_message("multi q", "Final answer: 720"))
    wrong = scorer.score_detailed(_message("multi q", "Final answer: 4"))

    assert full is not None and full.score == 1.0
    assert first is not None and first.score == 0.5
    assert second is not None and second.score == 0.5
    assert wrong is not None and wrong.score == 0.0


def test_hybrid_answer_scorer_handles_symbolic_numeric_answers() -> None:
    scorer = _hybrid_answer_scorer()

    exact = scorer.score_detailed(_message("sqrt q", "Final answer: \\sqrt{3}."))
    decimal = scorer.score_detailed(_message("sqrt q", "Final answer: 1.7320508075688772."))
    sqrt_fraction = scorer.score_detailed(_message("sqrt fraction q", "Final answer: \\frac{2\\sqrt{3}}{3}."))
    sqrt_fraction_decimal = scorer.score_detailed(_message("sqrt fraction q", "Final answer: 1.1547005383792515."))
    pi_value = scorer.score_detailed(_message("pi q", "Final answer: 164\\pi."))
    multi = scorer.score_detailed(_message("symbolic multi q", "Final answer: 0.333333333333;1.7320508075688772"))

    assert exact is not None and exact.score == 1.0
    assert decimal is not None and decimal.score == 0.8
    assert sqrt_fraction is not None and sqrt_fraction.score == 1.0
    assert sqrt_fraction_decimal is not None and sqrt_fraction_decimal.score == 0.8
    assert pi_value is not None and pi_value.score == 1.0
    assert multi is not None and multi.score == 1.0


def test_hybrid_answer_scorer_uses_relative_error_for_small_numeric_answers() -> None:
    scorer = _hybrid_answer_scorer()

    near = scorer.score_detailed(_message("small q", "Final answer: 0.0046."))
    factorial = scorer.score_detailed(_message("small q", "Final answer: \\frac{1}{26!}."))
    zero_exact = scorer.score_detailed(_message("zero q", "Final answer: 0."))
    zero_wrong = scorer.score_detailed(_message("zero q", "Final answer: 0.01."))

    assert near is not None and near.score == 0.8
    assert factorial is not None and factorial.score == 0.0
    assert zero_exact is not None and zero_exact.score == 1.0
    assert zero_wrong is not None and zero_wrong.score == 0.0


def test_hybrid_answer_scorer_matches_questions_with_generation_suffix() -> None:
    scorer = _hybrid_answer_scorer()

    question = "letter q\n\nReturn only the final answer in the form `Answer: <answer>`."
    detail = scorer.score_detailed(_message(question, "Answer: B"))
    assert detail is not None and detail.score == 1.0


def test_hybrid_answer_scorer_does_not_treat_option_explanations_as_final_letters() -> None:
    scorer = _hybrid_answer_scorer()

    response = "Option D: plausible but wrong.\nOption B: correct.\n\nFinal Answer: B"
    detail = scorer.score_detailed(_message("letter q", response))
    assert detail is not None and detail.score == 1.0

    scorer.answers["letter q"] = "D"
    detail = scorer.score_detailed(_message("letter q", response))
    assert detail is not None and detail.score == 0.05
