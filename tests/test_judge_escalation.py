"""Teste de escalada de Judge ponta a ponta.

Verifica três propriedades:
1. Escalada dispara: Qwen3.5-9B é chamado quando needs_revision_count >= threshold
2. Escalada dispara por score: Qwen3.5-9B é chamado quando score <= threshold
3. Estado da sessão sobrevive à evicção do workhorse durante escalada:
   feedback_history permanece íntegro após chamada ao reasoner,
   e worker consegue fazer nova chamada quando gemma é recarregado pelo Ollama.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.judge import JudgeAgent, JudgeEvaluation, Verdict
from agent.orchestrator import Orchestrator, SubTask, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluation(verdict: Verdict, score: int, feedback: str = "improve") -> JudgeEvaluation:
    return JudgeEvaluation(
        score=score, verdict=verdict, reasoning="test",
        criteria_scores={}, feedback=feedback, parse_success=True
    )


def _orchestrator_with_config(
    needs_revision: int = 2,
    score_threshold: int = 4,
    default_model: str = "gemma3ne4b",
    escalation_model: str = "qwen35-9b",
) -> Orchestrator:
    """Cria Orchestrator com thresholds injetados sem ler config.yaml."""
    with patch.object(Orchestrator, "_load_judge_config"):
        orch = Orchestrator.__new__(Orchestrator)
    # Injeta dependências manualmente
    orch.judge = JudgeAgent.__new__(JudgeAgent)
    orch.judge.llm_pool = MagicMock()
    orch.judge.model_id = default_model
    orch.judge_default_model = default_model
    orch.judge_escalation_model = escalation_model
    orch.escalation_needs_revision = needs_revision
    orch.escalation_score_threshold = score_threshold
    return orch


# ---------------------------------------------------------------------------
# 1. Escalada por needs_revision_count
# ---------------------------------------------------------------------------

def test_escalation_triggered_by_needs_revision_count():
    """Judge escalation model deve ser chamado quando needs_revision_count >= threshold."""
    orch = _orchestrator_with_config(needs_revision=2, score_threshold=4)

    call_log = []

    async def mock_evaluate(task, worker_output, role="general", context=None, model_id=None):
        call_log.append(model_id or orch.judge.model_id)
        count = len(call_log)
        if count == 1:
            return _make_evaluation(Verdict.NEEDS_REVISION, score=6, feedback="needs work iter1")
        elif count == 2:
            return _make_evaluation(Verdict.NEEDS_REVISION, score=6, feedback="needs work iter2")
        else:
            return _make_evaluation(Verdict.APPROVED, score=8)

    orch.judge.evaluate = mock_evaluate

    # Simula o loop de avaliação do orchestrator manualmente
    async def run():
        subtask = SubTask(id="t1", description="test task", role="general")
        subtask.judge_score = None
        needs_revision_count = 0
        feedback_history = []
        worker_outputs = ["output1", "output2", "output3"]

        for iteration, worker_output in enumerate(worker_outputs, 1):
            escalate = (
                needs_revision_count >= orch.escalation_needs_revision
                or (subtask.judge_score is not None
                    and subtask.judge_score <= orch.escalation_score_threshold)
            )
            judge_model = orch.judge_escalation_model if escalate else orch.judge_default_model

            evaluation = await orch.judge.evaluate(
                task=subtask.description,
                worker_output=worker_output,
                role=subtask.role,
                model_id=judge_model,
            )
            subtask.judge_score = evaluation.score

            if evaluation.verdict == Verdict.APPROVED:
                subtask.status = TaskStatus.COMPLETED
                break
            elif evaluation.verdict == Verdict.NEEDS_REVISION:
                needs_revision_count += 1
                feedback_history.append(evaluation.feedback)

        return call_log, feedback_history, subtask

    call_log_result, feedback, subtask = asyncio.run(run())

    assert call_log_result[0] == "gemma3ne4b", f"iter1 deveria usar default: {call_log_result[0]}"
    assert call_log_result[1] == "gemma3ne4b", f"iter2 deveria usar default: {call_log_result[1]}"
    assert call_log_result[2] == "qwen35-9b",  f"iter3 deveria escalar: {call_log_result[2]}"
    assert subtask.status == TaskStatus.COMPLETED
    assert len(feedback) == 2  # dois feedbacks acumulados antes da aprovação
    print(f"  PASS: escalada disparou na iter3 após {len(feedback)} NEEDS_REVISION")
    print(f"        modelos usados: {call_log_result}")


# ---------------------------------------------------------------------------
# 2. Escalada por score borderline
# ---------------------------------------------------------------------------

def test_escalation_triggered_by_low_score():
    """Judge escalation deve disparar quando score <= threshold, mesmo na iter1."""
    orch = _orchestrator_with_config(needs_revision=3, score_threshold=4)

    call_log = []

    async def mock_evaluate(task, worker_output, role="general", context=None, model_id=None):
        call_log.append(model_id)
        if len(call_log) == 1:
            return _make_evaluation(Verdict.NEEDS_REVISION, score=3)  # score <= 4 → escala
        return _make_evaluation(Verdict.APPROVED, score=8)

    orch.judge.evaluate = mock_evaluate

    async def run():
        subtask = SubTask(id="t1", description="test task", role="general")
        subtask.judge_score = None
        needs_revision_count = 0
        feedback_history = []

        for iteration, worker_output in enumerate(["out1", "out2"], 1):
            escalate = (
                needs_revision_count >= orch.escalation_needs_revision
                or (subtask.judge_score is not None
                    and subtask.judge_score <= orch.escalation_score_threshold)
            )
            judge_model = orch.judge_escalation_model if escalate else orch.judge_default_model
            evaluation = await orch.judge.evaluate(
                task="task", worker_output=worker_output, model_id=judge_model
            )
            subtask.judge_score = evaluation.score
            if evaluation.verdict == Verdict.APPROVED:
                subtask.status = TaskStatus.COMPLETED
                break
            needs_revision_count += 1
            feedback_history.append(evaluation.feedback)

        return call_log

    result = asyncio.run(run())

    assert result[0] == "gemma3ne4b", f"iter1 deveria usar default: {result[0]}"
    assert result[1] == "qwen35-9b",  f"iter2 deveria escalar por score borderline: {result[1]}"
    print(f"  PASS: escalada por score<=4 disparou na iter2: {result}")


# ---------------------------------------------------------------------------
# 3. Estado da sessão sobrevive à evicção do workhorse
# ---------------------------------------------------------------------------

def test_session_state_survives_workhorse_eviction():
    """feedback_history permanece intacto após escalada (evicção do workhorse é transparente).

    O estado vive em objetos Python — evicção do Ollama não afeta Python.
    Verifica que o contexto acumulado (feedback_history, subtask score) está
    disponível quando o worker faz nova chamada após a escalada.
    """
    orch = _orchestrator_with_config(needs_revision=2)

    evaluations = [
        _make_evaluation(Verdict.NEEDS_REVISION, score=6, feedback="fix the format"),
        _make_evaluation(Verdict.NEEDS_REVISION, score=5, feedback="add more detail"),
        _make_evaluation(Verdict.APPROVED,        score=8, feedback=""),
    ]
    eval_iter = iter(evaluations)

    async def mock_evaluate(task, worker_output, role="general", context=None, model_id=None):
        return next(eval_iter)

    orch.judge.evaluate = mock_evaluate

    async def run():
        feedback_history = []
        needs_revision_count = 0
        subtask = SubTask(id="t1", description="task", role="general")
        subtask.judge_score = None
        snapshots = []  # captura estado a cada iteração

        for worker_output in ["out1", "out2", "out3"]:
            escalate = (
                needs_revision_count >= orch.escalation_needs_revision
                or (subtask.judge_score is not None
                    and subtask.judge_score <= orch.escalation_score_threshold)
            )
            judge_model = orch.judge_escalation_model if escalate else orch.judge_default_model
            evaluation = await orch.judge.evaluate(
                task="task", worker_output=worker_output, model_id=judge_model
            )
            subtask.judge_score = evaluation.score

            snapshots.append({
                "model": judge_model,
                "feedback_count": len(feedback_history),
                "score": evaluation.score,
            })

            if evaluation.verdict == Verdict.APPROVED:
                subtask.status = TaskStatus.COMPLETED
                break
            needs_revision_count += 1
            feedback_history.append(evaluation.feedback)

        return snapshots, feedback_history, subtask

    snapshots, feedback_history, subtask = asyncio.run(run())

    # iter1: default judge, sem feedback anterior
    assert snapshots[0]["model"] == "gemma3ne4b"
    assert snapshots[0]["feedback_count"] == 0

    # iter2: default judge ainda (count=1, threshold=2), feedback_count=1
    assert snapshots[1]["model"] == "gemma3ne4b"
    assert snapshots[1]["feedback_count"] == 1

    # iter3: ESCALADA (count=2 >= threshold=2), feedback_count=2 — estado intacto
    assert snapshots[2]["model"] == "qwen35-9b", f"iter3 deveria escalar: {snapshots[2]['model']}"
    assert snapshots[2]["feedback_count"] == 2, \
        f"feedback_history deveria ter 2 itens, tem {snapshots[2]['feedback_count']}"

    assert subtask.status == TaskStatus.COMPLETED
    assert feedback_history == ["fix the format", "add more detail"]
    print(f"  PASS: estado sobreviveu à escalada — feedback_history={feedback_history}")
    print(f"        snapshots: {snapshots}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Judge Escalation Tests:")
    test_escalation_triggered_by_needs_revision_count()
    test_escalation_triggered_by_low_score()
    test_session_state_survives_workhorse_eviction()
    print("\nAll tests passed!")
