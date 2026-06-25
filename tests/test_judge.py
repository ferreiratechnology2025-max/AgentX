"""Tests for Judge Agent (unit tests only, no model inference)"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.judge import JudgeAgent, JudgeEvaluation, Verdict, get_judge


def test_verdict_enum():
    assert Verdict.APPROVED.value == "APPROVED"
    assert Verdict.REJECTED.value == "REJECTED"
    assert Verdict.NEEDS_REVISION.value == "NEEDS_REVISION"
    print("  PASS: Verdict enum")


def test_evaluation_dataclass():
    ev = JudgeEvaluation(score=8, verdict=Verdict.APPROVED, reasoning="good",
                         criteria_scores={"a": 8}, feedback="nice")
    assert ev.score == 8
    assert ev.verdict == Verdict.APPROVED
    assert ev.criteria_scores == {"a": 8}
    print("  PASS: JudgeEvaluation dataclass")


def test_parse_valid_json():
    judge = get_judge()
    result = judge._parse_evaluation(
        '{"score": 9, "verdict": "APPROVED", "reasoning": "Well done", '
        '"criteria_scores": {"correctness": 9}, "feedback": "Great"}'
    )
    assert result.score == 9
    assert result.verdict == Verdict.APPROVED
    assert result.criteria_scores == {"correctness": 9}
    print("  PASS: parse valid JSON")


def test_parse_embedded_json():
    judge = get_judge()
    result = judge._parse_evaluation(
        'Some text {"score": 6, "verdict": "NEEDS_REVISION", '
        '"reasoning": "Needs work", "criteria_scores": {"a": 6}, "feedback": "Fix it"}'
    )
    assert result.score == 6
    assert result.verdict == Verdict.NEEDS_REVISION
    print("  PASS: parse embedded JSON")


def test_parse_invalid_verdict_fallback():
    judge = get_judge()
    result = judge._parse_evaluation(
        '{"score": 3, "verdict": "INVALID", "reasoning": "Bad", '
        '"criteria_scores": {}, "feedback": "No"}'
    )
    assert result.score == 3
    assert result.verdict == Verdict.REJECTED
    print("  PASS: invalid verdict fallback to score")


def test_parse_no_json():
    judge = get_judge()
    result = judge._parse_evaluation("This is not JSON at all")
    assert result.score == 0
    assert result.verdict == Verdict.NEEDS_REVISION  # Judge retorna NEEDS_REVISION para dar chance de retry
    print("  PASS: no JSON fallback")


def test_rubrics():
    judge = get_judge()
    assert "correctness" in judge.RUBRICS["coding"]
    assert "accuracy" in judge.RUBRICS["research"]
    assert "helpfulness" in judge.RUBRICS["general"]
    assert len(judge.RUBRICS["coding"]) == 4
    print("  PASS: rubrics by role")


def test_build_prompt():
    judge = get_judge()
    prompt = judge._build_evaluation_prompt("Test task", "Test output",
                                            judge.RUBRICS["general"], None)
    assert "Test task" in prompt
    assert "Test output" in prompt
    assert "correctness" in prompt
    print("  PASS: build evaluation prompt")


if __name__ == "__main__":
    print("Judge Agent Unit Tests:")
    test_verdict_enum()
    test_evaluation_dataclass()
    test_parse_valid_json()
    test_parse_embedded_json()
    test_parse_invalid_verdict_fallback()
    test_parse_no_json()
    test_rubrics()
    test_build_prompt()
    print("\nAll Judge tests passed!")
