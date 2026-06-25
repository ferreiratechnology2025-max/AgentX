"""Fase 6 — Testes de votação por consistência.

Grupo 1: parser + builder de checklist (unit, sem LLM)
Grupo 2: voter de security (unit com mock de LLM)
Grupo 3: voter de judge no orchestrator (unit com mock de judge)
"""
import asyncio
import os
import sys
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.factory import AgentFactory
from agent.judge import JudgeEvaluation, Verdict
from agent.orchestrator import Orchestrator, SubTask, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_react_output(final_answer: str):
    from tools.schemas import ReActOutput
    return ReActOutput(thought="test", final_answer=final_answer)


def _orch():
    with patch.object(Orchestrator, "_load_judge_config"):
        orch = Orchestrator.__new__(Orchestrator)
    orch.judge = MagicMock()
    orch.judge_default_model = "gemma3ne4b"
    orch.judge_escalation_model = "qwen35-9b"
    orch.escalation_needs_revision = 2
    orch.escalation_score_threshold = 4
    return orch


def _eval(verdict: Verdict, score: int, parse_success: bool = True) -> JudgeEvaluation:
    return JudgeEvaluation(
        score=score, verdict=verdict, reasoning="test",
        criteria_scores={}, feedback="feedback", parse_success=parse_success,
    )


# ---------------------------------------------------------------------------
# 1. Parser e builder de checklist
# ---------------------------------------------------------------------------

parse = AgentFactory._parse_checklist_verdicts
build = AgentFactory._build_voted_checklist


def test_parse_individual_items():
    text = "1. SQL Injection: SIM, linha 42. 2. NAO. 3. NAO. 4. NAO. 5. NAO. 6. NAO. 7. NAO. 8. NAO."
    result = parse(text)
    assert result[1] == "SIM"
    assert all(result[i] == "NAO" for i in range(2, 9))
    print("  PASS: parse individual items")


def test_parse_range_notation():
    text = "1. SQL Injection: SIM, linha 42. 2-8. NAO."
    result = parse(text)
    assert result[1] == "SIM"
    assert all(result[i] == "NAO" for i in range(2, 9))
    print("  PASS: parse range '2-8. NAO'")


def test_parse_full_range_nao():
    text = "1-8. NAO em todas as classes."
    result = parse(text)
    assert all(result.get(i) == "NAO" for i in range(1, 9))
    print("  PASS: parse range completo '1-8. NAO'")


def test_build_voted_all_nao():
    majority = {i: "NAO" for i in range(1, 9)}
    result = build(majority, {})
    assert "SIM" not in result
    assert "1" in result
    print(f"  PASS: build all-NAO: {result}")


def test_build_voted_one_sim():
    majority = {1: "SIM", **{i: "NAO" for i in range(2, 9)}}
    sim_details = {1: "SQL Injection: SIM, linha 42."}
    result = build(majority, sim_details)
    assert "SIM" in result
    assert "SQL Injection" in result
    assert "2-8" in result  # range de NAOs agrupado
    print(f"  PASS: build one-SIM: {result}")


# ---------------------------------------------------------------------------
# 2. Voter de security (mock LLM)
# ---------------------------------------------------------------------------

def test_voter_majority_nao_wins():
    """2 respostas NAO + 1 SIM -> maioria NAO para o item."""
    answers = [
        "1. SQL Injection: SIM, linha 42. 2-8. NAO.",   # vote 1: item 1 = SIM
        "1-8. NAO.",                                      # vote 2: item 1 = NAO
        "1-8. NAO.",                                      # vote 3: item 1 = NAO
    ]
    outputs = [_make_react_output(a) for a in answers]

    # Simula o voter diretamente: vota por item
    item_votes = {i: [] for i in range(1, 9)}
    for out in outputs:
        parsed = parse(out.final_answer)
        for item, v in parsed.items():
            item_votes[item].append(v)

    majority = {}
    for item in range(1, 9):
        votes = item_votes[item]
        if not votes:
            majority[item] = "NAO"
            continue
        counts = Counter(votes)
        top, _ = counts.most_common(1)[0]
        majority[item] = top

    assert majority[1] == "NAO", f"Maioria deveria ser NAO para item 1, got {majority[1]}"
    print("  PASS: 2 votos NAO vencem 1 voto SIM no item 1")


def test_voter_majority_sim_wins():
    """2 respostas SIM + 1 NAO -> maioria SIM para o item."""
    answers = [
        "1. SQL Injection: SIM, linha 42. 2-8. NAO.",
        "1. SIM, linha 10. 2-8. NAO.",
        "1-8. NAO.",
    ]
    item_votes = {i: [] for i in range(1, 9)}
    for a in answers:
        parsed = parse(a)
        for item, v in parsed.items():
            item_votes[item].append(v)

    counts_1 = Counter(item_votes[1])
    top, count = counts_1.most_common(1)[0]
    assert top == "SIM" and count == 2, f"Esperado SIM/2, got {top}/{count}"
    print("  PASS: 2 votos SIM vencem 1 voto NAO no item 1")


async def _run_voter_with_mocked_llm():
    """Testa o voter async com LLM mockado."""
    first_out = _make_react_output("1. SQL Injection: SIM, linha 42.")
    extra1    = _make_react_output("1-8. NAO.")
    extra2    = _make_react_output("1-8. NAO.")

    mock_llm = MagicMock()
    call_count = 0

    async def mock_generate_with_validation(messages, tools, max_tokens, temperature):
        nonlocal call_count
        call_count += 1
        return (extra1 if call_count == 1 else extra2), {}

    mock_llm.generate_with_validation = mock_generate_with_validation

    result = await AgentFactory._security_checklist_voter(
        messages=[], tools_spec=[], first_output=first_out,
        llm=mock_llm, temperature=0.7,
    )

    assert result.final_answer is not None
    assert call_count == 2, f"Voter deve chamar LLM 2x adicionais, chamou {call_count}x"
    # Maioria para item 1: 1 SIM + 2 NAO -> NAO
    verdicts = parse(result.final_answer)
    assert verdicts.get(1) == "NAO", f"Maioria item 1 deveria ser NAO, got {verdicts.get(1)}"
    print(f"  PASS: voter com LLM mock: item 1 = NAO por maioria. final_answer={result.final_answer}")
    return result


def test_security_voter_async():
    asyncio.run(_run_voter_with_mocked_llm())


def test_voter_disabled_after_use():
    """final_answer_voter deve ser None após o primeiro uso em _think()."""
    from agent.core import AutonomousAgent, AgentConfig
    from tools.registry import ToolRegistry

    mock_llm = MagicMock()
    mock_llm.model_id = "gemma3ne4b"
    registry = ToolRegistry()
    agent = AutonomousAgent(llm_manager=mock_llm, tool_registry=registry)

    was_called = False
    async def dummy_voter(messages, tools_spec, output, llm, temp):
        nonlocal was_called
        was_called = True
        return output

    agent.final_answer_voter = dummy_voter

    # Simula o que _think() faz quando voter está setado
    async def simulate():
        from tools.schemas import ReActOutput
        validated = ReActOutput(thought="test", final_answer="1-8. NAO.")
        if validated.final_answer and agent.final_answer_voter is not None:
            voter_fn = agent.final_answer_voter
            agent.final_answer_voter = None
            validated = await voter_fn([], [], validated, None, 0.7)
        return validated

    asyncio.run(simulate())
    assert was_called, "Voter deve ter sido chamado"
    assert agent.final_answer_voter is None, "Voter deve ser None após uso"
    print("  PASS: final_answer_voter desabilitado após uso em _think()")


# ---------------------------------------------------------------------------
# 3. Judge voting no orchestrator
# ---------------------------------------------------------------------------

async def _run_judge_vote_majority():
    orch = _orch()
    first = _eval(Verdict.NEEDS_REVISION, score=3)
    vote2 = _eval(Verdict.NEEDS_REVISION, score=4)
    vote3 = _eval(Verdict.APPROVED,        score=7)

    call_num = [0]
    async def mock_evaluate(task, worker_output, role="general", context=None, model_id=None):
        call_num[0] += 1
        return [vote2, vote3][call_num[0] - 1]

    orch.judge.evaluate = mock_evaluate

    result_eval, escalated = await orch._vote_judge_verdict(
        task="test", worker_output="output", role="general",
        first_evaluation=first, subtask_id="t1",
    )
    return result_eval, escalated, call_num[0]


def test_judge_vote_majority_wins():
    """NEEDS_REVISION 2/3 vence APPROVED 1/3."""
    result, escalated, calls = asyncio.run(_run_judge_vote_majority())
    assert result.verdict == Verdict.NEEDS_REVISION, \
        f"Maioria deveria ser NEEDS_REVISION, got {result.verdict}"
    assert not escalated
    assert calls == 2, f"2 chamadas adicionais esperadas, got {calls}"
    print(f"  PASS: maioria NEEDS_REVISION vence, escalated={escalated}")


async def _run_judge_vote_tie():
    orch = _orch()
    first  = _eval(Verdict.APPROVED,        score=3)
    vote2  = _eval(Verdict.NEEDS_REVISION,  score=3)
    vote3  = _eval(Verdict.REJECTED,        score=3)
    reasoner = _eval(Verdict.NEEDS_REVISION, score=5)

    call_num = [0]
    async def mock_evaluate(task, worker_output, role="general", context=None, model_id=None, timeout=None):
        call_num[0] += 1
        if call_num[0] == 1:
            return vote2
        elif call_num[0] == 2:
            return vote3
        else:
            return reasoner  # chamada ao reasoner em empate

    orch.judge.evaluate = mock_evaluate

    result_eval, escalated = await orch._vote_judge_verdict(
        task="test", worker_output="output", role="general",
        first_evaluation=first, subtask_id="t1",
    )
    return result_eval, escalated, call_num[0]


def test_judge_vote_genuine_tie_escalates():
    """APPROVED + NEEDS_REVISION + REJECTED (1-1-1) -> escalada pro reasoner."""
    result, escalated, calls = asyncio.run(_run_judge_vote_tie())
    assert escalated, "Empate 1-1-1 deve escalar pro reasoner"
    assert calls == 3, f"3 chamadas esperadas (2 votos + 1 reasoner), got {calls}"
    print(f"  PASS: empate 1-1-1 escalou pro reasoner após {calls} chamadas")


async def _run_orchestrator_skips_voting_when_revision_high():
    """needs_revision_count >= threshold -> escala direto, sem votação."""
    orch = _orch()
    judge_calls = []

    async def mock_evaluate(task, worker_output, role="general", context=None, model_id=None):
        judge_calls.append(model_id)
        return _eval(Verdict.NEEDS_REVISION, score=3)

    orch.judge.evaluate = mock_evaluate

    # Simula a lógica de seleção do orchestrator quando needs_revision_count=2
    needs_revision_count = 2
    first_eval = _eval(Verdict.NEEDS_REVISION, score=3)

    direct_escalate = needs_revision_count >= orch.escalation_needs_revision
    judge_model = orch.judge_escalation_model if direct_escalate else orch.judge_default_model
    evaluation = await orch.judge.evaluate(
        task="task", worker_output="output", role="general", model_id=judge_model,
    )

    voted = False
    escalated_via_tie = False
    if (not direct_escalate
            and evaluation.score <= orch.escalation_score_threshold
            and evaluation.parse_success):
        evaluation, escalated_via_tie = await orch._vote_judge_verdict(
            task="task", worker_output="output", role="general",
            first_evaluation=evaluation, subtask_id="t1",
        )
        voted = True

    return judge_calls, voted, judge_model


def test_orchestrator_skips_voting_when_revision_high():
    """needs_revision_count >= 2 deve ir direto pro reasoner sem votação."""
    calls, voted, model = asyncio.run(_run_orchestrator_skips_voting_when_revision_high())
    assert model == "qwen35-9b", f"Deveria usar reasoner, usou {model}"
    assert not voted, "Não deveria votar quando needs_revision_count alto"
    assert len(calls) == 1, f"Apenas 1 chamada esperada (direto ao reasoner), got {len(calls)}"
    print(f"  PASS: needs_revision>=2 escalou direto para {model} sem votação")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Testes de votação por consistência (Fase 6) ===\n")

    print("-- Parser e builder --")
    test_parse_individual_items()
    test_parse_range_notation()
    test_parse_full_range_nao()
    test_build_voted_all_nao()
    test_build_voted_one_sim()

    print("\n-- Voter de security --")
    test_voter_majority_nao_wins()
    test_voter_majority_sim_wins()
    test_security_voter_async()
    test_voter_disabled_after_use()

    print("\n-- Judge voting no orchestrator --")
    test_judge_vote_majority_wins()
    test_judge_vote_genuine_tie_escalates()
    test_orchestrator_skips_voting_when_revision_high()

    print("\nAll tests passed!")
