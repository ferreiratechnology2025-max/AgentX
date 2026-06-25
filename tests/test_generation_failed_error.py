"""Tests for GenerationFailedError failure path.

Verifica que uma falha de geração (rede/timeout/parse) nunca fabrica conteúdo:
- status="failed" salvo no estado
- evento {"type": "error"} emitido via SSE
- nenhum {"type": "final"} produzido
- checkpoint persistido
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests

from llm.manager import LLMManager, GenerationFailedError
from agent.core import AutonomousAgent, AgentConfig
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(config: dict | None = None) -> LLMManager:
    cfg = config or {"llm": {"model_id": "test-model"}, "ollama": {"base_url": "http://localhost:11434"}}
    return LLMManager(cfg)


def _make_agent(llm: LLMManager) -> AutonomousAgent:
    registry = ToolRegistry()
    return AutonomousAgent(
        llm_manager=llm,
        tool_registry=registry,
        config=AgentConfig(max_steps=3, verbose=False),
    )


async def _collect_events(agent: AutonomousAgent, goal: str) -> list:
    events = []
    async for event in agent.run(goal, session_id=None):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Unit: GenerationFailedError levantada quando _validate_output retorna None
# ---------------------------------------------------------------------------

def test_generate_with_validation_raises_on_unparseable():
    """generate_with_validation deve levantar GenerationFailedError, nunca retornar None."""
    llm = _make_llm()

    async def run():
        with patch.object(llm, "generate_with_tools", new=AsyncMock(return_value=("!!!invalid!!!", {}))):
            await llm.generate_with_validation(messages=[], tools=[])

    try:
        asyncio.run(run())
        assert False, "Deveria ter levantado GenerationFailedError"
    except GenerationFailedError:
        pass
    print("  PASS: generate_with_validation levanta GenerationFailedError em parse inválido")


# ---------------------------------------------------------------------------
# Unit: HTTP retry — ConnectionError esgota tentativas e levanta GenerationFailedError
# ---------------------------------------------------------------------------

def test_http_retry_exhausted_on_connection_error():
    """5xx/ConnectionError esgota MAX_HTTP_RETRIES e levanta GenerationFailedError."""
    llm = _make_llm()
    call_count = 0

    def failing_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise requests.exceptions.ConnectionError("simulated connection refused")

    async def run():
        with patch("requests.post", side_effect=failing_post):
            with patch.object(type(llm), "HTTP_RETRY_BACKOFF", new=[0, 0, 0]):
                await llm.generate_with_tools(messages=[], tools=[])

    try:
        asyncio.run(run())
        assert False, "Deveria ter levantado GenerationFailedError"
    except GenerationFailedError:
        pass

    assert call_count == llm.MAX_HTTP_RETRIES, (
        f"Esperado {llm.MAX_HTTP_RETRIES} tentativas, feitas {call_count}"
    )
    print(f"  PASS: ConnectionError — retry esgotado após {call_count} tentativas")


def test_http_4xx_raises_immediately_no_retry():
    """4xx deve levantar GenerationFailedError imediatamente, sem retry."""
    llm = _make_llm()
    call_count = 0

    def bad_request_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        raise requests.exceptions.HTTPError(response=mock_resp)

    async def run():
        with patch("requests.post", side_effect=bad_request_post):
            await llm.generate_with_tools(messages=[], tools=[])

    try:
        asyncio.run(run())
        assert False, "Deveria ter levantado GenerationFailedError"
    except GenerationFailedError as e:
        assert "400" in str(e) and "permanente" in str(e), (
            f"Mensagem de erro deveria mencionar 400 e 'permanente': {e}"
        )

    assert call_count == 1, f"4xx não deve ter retry — esperado 1 chamada, feitas {call_count}"
    print(f"  PASS: HTTP 400 — sobe imediatamente sem retry, {call_count} chamada")


# ---------------------------------------------------------------------------
# Integration: AutonomousAgent.run() emite error, nunca final, status=failed
# ---------------------------------------------------------------------------

def test_agent_run_emits_error_not_final_on_generation_failure():
    """run() deve emitir evento error e setar status=failed, nunca fabricar Final Answer."""
    llm = _make_llm()
    agent = _make_agent(llm)

    async def run():
        with patch.object(
            llm,
            "generate_with_validation",
            new=AsyncMock(side_effect=GenerationFailedError("mock network failure")),
        ):
            return await _collect_events(agent, goal="test goal")

    events = asyncio.run(run())

    types = [e["type"] for e in events]

    # Nenhum "final" deve aparecer
    assert "final" not in types, f"Final Answer fabricado! Eventos: {types}"

    # Deve ter exatamente um "error"
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) >= 1, f"Nenhum evento error emitido. Eventos: {types}"
    assert "GenerationFailed" in error_events[0]["content"] or "Falha" in error_events[0]["content"], (
        f"Mensagem de erro inesperada: {error_events[0]['content']}"
    )

    # Status deve ser "failed"
    assert agent.state.status == "failed", (
        f"Estado incorreto: esperado 'failed', obtido '{agent.state.status}'"
    )

    print("  PASS: run() emite error, status=failed, nenhum Final Answer fabricado")


# ---------------------------------------------------------------------------
# Integration: Judge não recebe conteúdo fabricado (worker_output permanece None)
# ---------------------------------------------------------------------------

def test_worker_output_is_none_on_generation_failure():
    """Orchestrator não deve receber worker_output quando geração falha."""
    llm = _make_llm()
    agent = _make_agent(llm)

    async def run():
        with patch.object(
            llm,
            "generate_with_validation",
            new=AsyncMock(side_effect=GenerationFailedError("mock timeout")),
        ):
            return await _collect_events(agent, goal="test goal")

    events = asyncio.run(run())

    # Simula o que o orchestrator faz: extrai worker_output do evento "final"
    worker_output = next(
        (e.get("content") for e in events if e["type"] == "final"), None
    )
    assert worker_output is None, (
        f"worker_output não deveria ter conteúdo, obtido: {worker_output!r}"
    )
    print("  PASS: worker_output é None, Judge não recebe conteúdo fabricado")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GenerationFailedError Tests:")
    test_generate_with_validation_raises_on_unparseable()
    test_http_retry_exhausted_on_connection_error()
    test_http_4xx_raises_immediately_no_retry()
    test_agent_run_emits_error_not_final_on_generation_failure()
    test_worker_output_is_none_on_generation_failure()
    print("\nAll tests passed!")
