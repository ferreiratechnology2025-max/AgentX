"""Tests for Pydantic ReActOutput validation"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import ValidationError
from tools.schemas import ReActOutput, ReActToolCall


def test_valid_tool_call():
    """Tool call valido deve passar."""
    output = ReActOutput(
        thought="I need to check the date",
        action=ReActToolCall(name="current_datetime", arguments={}),
    )
    assert output.thought == "I need to check the date"
    assert output.action.name == "current_datetime"
    assert output.final_answer is None
    print("  PASS: valid tool call")


def test_valid_final_answer():
    """Final answer valido deve passar."""
    output = ReActOutput(
        thought="I have the answer",
        final_answer="The date is 19/06/2026",
    )
    assert output.final_answer == "The date is 19/06/2026"
    assert output.action is None
    print("  PASS: valid final answer")


def test_action_with_arguments():
    """Tool call com argumentos deve passar."""
    output = ReActOutput(
        thought="Need to calculate",
        action=ReActToolCall(name="calculator", arguments={"expression": "2+2"}),
    )
    assert output.action.arguments["expression"] == "2+2"
    print("  PASS: action with arguments")


def test_both_none_should_fail():
    """Output sem action E sem final_answer deve falhar."""
    try:
        ReActOutput(thought="test", action=None, final_answer=None)
        assert False, "Deveria ter levantado ValidationError"
    except ValidationError:
        print("  PASS: both None fails validation")


def test_tool_call_with_empty_args():
    """Tool call com arguments vazio deve passar."""
    output = ReActOutput(
        thought="test",
        action=ReActToolCall(name="current_datetime", arguments={}),
    )
    assert output.action.arguments == {}
    print("  PASS: empty arguments")


if __name__ == "__main__":
    print("Pydantic Validation Tests:")
    test_valid_tool_call()
    test_valid_final_answer()
    test_action_with_arguments()
    test_both_none_should_fail()
    test_tool_call_with_empty_args()
    print("\nAll tests passed!")
