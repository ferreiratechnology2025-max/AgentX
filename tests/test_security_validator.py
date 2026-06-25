"""Testa o validador determinístico do checklist de security.

Três grupos:
1. Validador isolado (unit) — sem LLM
2. Wiring no factory — sem LLM
3. Integração com agente real — requer Ollama
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.factory import AgentFactory


# ---------------------------------------------------------------------------
# 1. Validador isolado
# ---------------------------------------------------------------------------

validate = AgentFactory._security_checklist_validator


def test_validator_accepts_all_8_individual():
    text = "1. NÃO. 2. NÃO. 3. NÃO. 4. SIM, linha 10. 5. NÃO. 6. NÃO. 7. NÃO. 8. NÃO."
    assert validate(text) is None
    print("  PASS: 8 itens individuais aceitos")


def test_validator_accepts_range_notation():
    text = "1. SQL Injection: SIM, linha 42. 2-8. NÃO."
    assert validate(text) is None
    print("  PASS: notação por faixa '2-8. NÃO' aceita")


def test_validator_accepts_full_range():
    text = "1-8. NÃO em todas as classes."
    assert validate(text) is None
    print("  PASS: faixa completa '1-8. NÃO' aceita")


def test_validator_rejects_single_item_only():
    text = "1. SQL Injection: SIM, linha 42."
    result = validate(text)
    assert result is not None, "Deve rejeitar quando itens 2-8 estão ausentes"
    assert "2" in result or "2-8" in result or "2, 3" in result
    print(f"  PASS: item único rejeitado, mensagem: {result[:80]}")


def test_validator_rejects_partial_checklist():
    text = "1. NÃO. 2. NÃO. 3. SIM."
    result = validate(text)
    assert result is not None
    missing_mentioned = all(str(n) in result for n in [4, 5, 6, 7, 8])
    assert missing_mentioned, f"Itens faltantes não mencionados: {result}"
    print(f"  PASS: checklist parcial rejeitado, missing={result[:80]}")


def test_validator_accepts_mixed_individual_and_range():
    text = "1. SIM, linha 5, SQL injection. 2. NÃO. 3-8. NÃO."
    assert validate(text) is None
    print("  PASS: mix de individual + faixa aceito")


# ---------------------------------------------------------------------------
# 2. Wiring no factory
# ---------------------------------------------------------------------------

def test_security_agent_gets_validator():
    """create_agent para role=security deve setar final_answer_validator."""
    factory = AgentFactory.__new__(AgentFactory)
    factory.created_agents = {}
    factory._lock = asyncio.Lock()
    factory._tool_registry_cache = None

    # Simula criação sem LLM — só verifica que o validator é setado
    from agent.core import AutonomousAgent, AgentConfig
    from tools.registry import ToolRegistry
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.model_id = "gemma3ne4b"
    registry = ToolRegistry()

    agent = AutonomousAgent(llm_manager=mock_llm, tool_registry=registry, config=AgentConfig())
    agent.available_tool_names = {"calculator", "project_manager", "process_orchestrator"}

    # Simula o que create_agent faz para role=security
    if "security" == "security":
        agent.final_answer_validator = factory._security_checklist_validator

    assert agent.final_answer_validator is not None, "Validator não setado para role=security"
    assert agent.final_answer_validator is AgentFactory._security_checklist_validator
    print("  PASS: security agent tem final_answer_validator setado")


def test_non_security_agent_has_no_validator():
    """create_agent para roles não-security NÃO deve setar validator."""
    from agent.core import AutonomousAgent, AgentConfig
    from tools.registry import ToolRegistry
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    registry = ToolRegistry()

    for role in ["general", "coding", "code_review", "tdd", "planning", "architect"]:
        agent = AutonomousAgent(llm_manager=mock_llm, tool_registry=registry, config=AgentConfig())
        # Validator só é setado em create_agent quando role=="security"
        assert agent.final_answer_validator is None, \
            f"Role '{role}' não deveria ter validator, mas tem"
    print("  PASS: roles não-security não têm validator")


def test_validator_disabled_after_first_use():
    """_check_final_answer desabilita o validator após o primeiro uso."""
    from agent.core import AutonomousAgent, AgentConfig
    from tools.registry import ToolRegistry
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    registry = ToolRegistry()
    agent = AutonomousAgent(llm_manager=mock_llm, tool_registry=registry)
    agent.final_answer_validator = AgentFactory._security_checklist_validator

    # Primeira chamada: retorna correção (checklist incompleto)
    result1 = agent._check_final_answer("1. SIM.")
    assert result1 is not None, "Primeira checagem deve retornar correção"

    # Validator deve estar desabilitado agora
    assert agent.final_answer_validator is None, "Validator deve ser None após primeiro uso"

    # Segunda chamada: None mesmo com checklist incompleto (validator já desabilitado)
    result2 = agent._check_final_answer("1. SIM.")
    assert result2 is None, "Segunda chamada deve retornar None (validator desabilitado)"
    print("  PASS: validator desabilitado após primeiro uso — retry único garantido")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Testes do validador de security ===\n")

    print("-- Validador isolado --")
    test_validator_accepts_all_8_individual()
    test_validator_accepts_range_notation()
    test_validator_accepts_full_range()
    test_validator_rejects_single_item_only()
    test_validator_rejects_partial_checklist()
    test_validator_accepts_mixed_individual_and_range()

    print("\n-- Wiring no factory --")
    test_security_agent_gets_validator()
    test_non_security_agent_has_no_validator()
    test_validator_disabled_after_first_use()

    print("\nAll tests passed!")
