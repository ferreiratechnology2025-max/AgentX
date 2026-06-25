"""Verifica que os agentes security e code_review produzem output binário por item.

Testa dois aspectos:
1. O system prompt carregado pelo factory contém as diretivas esperadas.
2. A resposta do modelo (real, via Ollama) segue o padrão binário
   (SIM/NÃO por item, não parecer geral).

Atenção: os testes de resposta real (test_*_output_format) requerem Ollama rodando.
Os testes de carregamento de prompt (test_*_prompt_loaded) são puramente locais.
"""

import asyncio
import re
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import AsyncMock, MagicMock, patch
from agent.factory import AgentFactory


# ---------------------------------------------------------------------------
# 1. Testes de carregamento de prompt (locais, sem LLM)
# ---------------------------------------------------------------------------

def _load_agent_file(role: str) -> str:
    from pathlib import Path
    p = Path("data/knowledge/ecc_agents") / f"{role}.md"
    return p.read_text(encoding="utf-8")


def test_security_prompt_has_binary_directive():
    content = _load_agent_file("security")
    assert "SIM" in content and "NÃO" in content, "Prompt security deve conter diretiva SIM/NÃO"
    assert "Regra dura" in content, "Prompt security deve conter 'Regra dura'"
    assert "project_manager" in content, "Few-shot de security deve referenciar project_manager"
    print("  PASS: security prompt contém diretiva binária e few-shot correto")


def test_code_review_prompt_has_binary_directive():
    content = _load_agent_file("code_review")
    assert "SIM" in content and "NÃO" in content, "Prompt code_review deve conter diretiva SIM/NÃO"
    assert "Regra dura" in content, "Prompt code_review deve conter 'Regra dura'"
    assert "project_manager" in content, "Few-shot de code_review deve referenciar project_manager"
    print("  PASS: code_review prompt contém diretiva binária e few-shot correto")


def test_security_tools_include_project_manager():
    """ROLE_TOOLS de security deve incluir project_manager (necessário para ler arquivos)."""
    tools = AgentFactory.ROLE_TOOLS.get("security", [])
    assert "project_manager" in tools, f"security ROLE_TOOLS deve ter project_manager, tem: {tools}"
    print(f"  PASS: security ROLE_TOOLS = {tools}")


def test_tdd_tools_include_process_orchestrator():
    """ROLE_TOOLS de tdd deve incluir process_orchestrator (necessário para rodar pytest)."""
    tools = AgentFactory.ROLE_TOOLS.get("tdd", [])
    assert "process_orchestrator" in tools, f"tdd ROLE_TOOLS deve ter process_orchestrator, tem: {tools}"
    print(f"  PASS: tdd ROLE_TOOLS = {tools}")


def test_all_roles_have_agent_files():
    """Todos os roles com prompts devem ter arquivo em ecc_agents/."""
    from pathlib import Path
    roles = ["security", "code_review", "tdd", "planning", "architect",
             "general", "coding", "research"]
    missing = []
    for role in roles:
        p = Path("data/knowledge/ecc_agents") / f"{role}.md"
        if not p.exists():
            missing.append(role)
    assert not missing, f"Arquivos de agente ausentes: {missing}"
    print(f"  PASS: todos os {len(roles)} roles têm arquivo ecc_agents/*.md")


def test_factory_loads_file_for_all_roles():
    """_load_ecc_prompt deve retornar conteúdo do arquivo pra qualquer role com arquivo."""
    factory = AgentFactory.__new__(AgentFactory)
    factory.created_agents = {}
    factory._lock = None
    factory._tool_registry_cache = None

    for role in ["security", "code_review", "tdd", "planning", "architect",
                 "general", "coding", "research"]:
        prompt = factory._load_ecc_prompt(role)
        assert len(prompt) > 50, f"Prompt para '{role}' muito curto ({len(prompt)} chars) — arquivo não carregado?"
        assert prompt != AgentFactory.DEFAULT_PROMPTS.get(role, ""), \
            f"Role '{role}' ainda usando DEFAULT_PROMPT em vez do arquivo ECC"
    print("  PASS: _load_ecc_prompt carrega arquivo para todos os 8 roles")


# ---------------------------------------------------------------------------
# 2. Teste de formato de saída real (requer Ollama + gemma3ne4b)
# ---------------------------------------------------------------------------

VULNERABLE_CODE = """
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect("db.sqlite3")
    query = f"SELECT * FROM users WHERE id={user_id}"
    return conn.execute(query).fetchone()
"""

CLEAN_CODE = """
import sqlite3

def get_user(user_id: int) -> dict | None:
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
    return cursor.fetchone()
"""


def _looks_like_binary_checklist(text: str) -> bool:
    """Retorna True se a resposta contém padrão de checklist binário (N. ... SIM/NÃO).

    Aceita formatos como:
      "1. SIM, linha 42..."
      "1. SQL Injection: SIM, linha 42..."
      "1-8. NÃO."
      "2-8. NAO em todas as classes."
    """
    has_numbered_item = bool(re.search(r"\b[1-8][\.\-]", text))
    has_verdict = bool(re.search(r"\b(SIM|N[AÃ]O)\b", text, re.IGNORECASE))
    return has_numbered_item and has_verdict


def _extract_final_answer(text: str) -> str:
    """Extrai o final_answer de uma resposta JSON, ou retorna o texto inteiro."""
    import json
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "final_answer" in data:
            return data["final_answer"]
    except (json.JSONDecodeError, TypeError):
        pass
    return text


def _looks_like_general_review(text: str) -> bool:
    """Retorna True se a resposta é um parecer geral (sem estrutura binária).

    Verifica apenas o final_answer, não o thought.
    """
    final_answer = _extract_final_answer(text)
    if not final_answer or not final_answer.strip():
        return False
    general_phrases = [
        "o código parece", "the code looks", "overall", "generally",
        "this code is", "the implementation", "bem estruturado"
    ]
    lower = final_answer.lower()
    return any(phrase in lower for phrase in general_phrases)


async def _run_security_format_test():
    """Envia código vulnerável ao agente security e verifica formato da resposta."""
    from llm.manager import LLMManager, Message
    from agent.factory import AgentFactory
    from pathlib import Path

    factory = AgentFactory.__new__(AgentFactory)
    factory.created_agents = {}
    factory._lock = asyncio.Lock()
    factory._tool_registry_cache = None

    system_prompt = factory._load_ecc_prompt("security")
    from agent.ecc_loader import load_ecc_rules_for_role
    ecc_rules = load_ecc_rules_for_role("security")
    if ecc_rules:
        system_prompt += f"\n\n## REGRAS ECC\n{ecc_rules}"

    llm = LLMManager({
        "llm": {"model_id": "gemma3ne4b", "num_ctx": 8192},
        "ollama": {"base_url": "http://localhost:11434"},
    })

    task = f"Analise este código Python para vulnerabilidades:\n\n```python{VULNERABLE_CODE}```"
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=task),
    ]

    response, _ = await llm.generate(
        task, max_tokens=512, temperature=0.1,
        system_prompt=system_prompt
    )
    return response


async def _run_code_review_format_test():
    """Envia código ao agente code_review e verifica formato da resposta."""
    from llm.manager import LLMManager
    from agent.factory import AgentFactory
    from agent.ecc_loader import load_ecc_rules_for_role

    factory = AgentFactory.__new__(AgentFactory)
    factory.created_agents = {}
    factory._lock = asyncio.Lock()
    factory._tool_registry_cache = None

    system_prompt = factory._load_ecc_prompt("code_review")
    ecc_rules = load_ecc_rules_for_role("code_review")
    if ecc_rules:
        system_prompt += f"\n\n## REGRAS ECC\n{ecc_rules}"

    llm = LLMManager({
        "llm": {"model_id": "gemma3ne4b", "num_ctx": 8192},
        "ollama": {"base_url": "http://localhost:11434"},
    })

    task = f"Revise este código Python:\n\n```python{CLEAN_CODE}```"
    response, _ = await llm.generate(task, max_tokens=512, temperature=0.1,
                                      system_prompt=system_prompt)
    return response


def test_security_output_is_binary_format():
    """Security agent deve produzir checklist binário, não parecer geral."""
    response = asyncio.run(_run_security_format_test())
    final_answer = _extract_final_answer(response)
    print(f"\n  [security] Resposta:\n{response[:600]}")
    assert _looks_like_binary_checklist(final_answer), \
        f"Resposta não tem formato binário (SIM/NÃO por item):\n{final_answer[:400]}"
    assert not _looks_like_general_review(response), \
        f"Resposta parece parecer geral, não checklist:\n{final_answer[:400]}"
    print("  PASS: security output é checklist binário")


def test_code_review_output_is_binary_format():
    """Code review agent deve produzir checklist binário, não parecer geral."""
    response = asyncio.run(_run_code_review_format_test())
    final_answer = _extract_final_answer(response)
    print(f"\n  [code_review] Resposta:\n{response[:600]}")
    assert _looks_like_binary_checklist(final_answer), \
        f"Resposta não tem formato binário (SIM/NÃO por item):\n{final_answer[:400]}"
    assert not _looks_like_general_review(response), \
        f"Resposta parece parecer geral, não checklist:\n{final_answer[:400]}"
    print("  PASS: code_review output é checklist binário")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Testes de formato de agente ===\n")

    print("-- Carregamento de prompt (locais) --")
    test_security_prompt_has_binary_directive()
    test_code_review_prompt_has_binary_directive()
    test_security_tools_include_project_manager()
    test_tdd_tools_include_process_orchestrator()
    test_all_roles_have_agent_files()
    test_factory_loads_file_for_all_roles()

    print("\n-- Formato de saída real (requer Ollama) --")
    test_security_output_is_binary_format()
    test_code_review_output_is_binary_format()

    print("\nAll tests passed!")
