"""Built-in tools for the agent"""

import math
from datetime import datetime
from typing import Optional

from .base import Tool, ToolPermission
from .project_manager import PROJECT_MANAGER_TOOL
from .git_worker import GIT_WORKER_TOOL
from .process_orchestrator import PROCESS_ORCHESTRATOR_TOOL

# Será injetado pelo agente
_memory_instance = None

def set_memory_instance(memory):
    """Injeta a instância de memória (chamado durante inicialização)"""
    global _memory_instance
    _memory_instance = memory


# ============== FUNÇÕES DAS FERRAMENTAS ==============

async def calculator_func(expression: str) -> str:
    """Calculadora segura - avalia expressões matemáticas"""
    # Caracteres permitidos (nada perigoso)
    allowed_chars = set("0123456789+-*/().% ")
    if not all(c in allowed_chars for c in expression):
        return "Erro: Expressão contém caracteres inválidos"
    
    try:
        # Ambiente seguro para eval
        safe_dict = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }
        result = eval(expression, safe_dict, {})
        return f"Resultado: {result}"
    except ZeroDivisionError:
        return "Erro: Divisão por zero"
    except Exception as e:
        return f"Erro ao calcular: {str(e)}"


async def datetime_func() -> str:
    """Retorna data e hora atuais"""
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


async def save_memory_func(text: str, importance: float = 1.0) -> str:
    """Salva informação na memória persistente"""
    if _memory_instance is None:
        return "⚠️ Sistema de memória não disponível"
    
    try:
        memory_id = await _memory_instance.salvar(text, importance)
        return f"✅ Memória salva (ID: {memory_id}, importância: {importance})"
    except Exception as e:
        return f"❌ Erro ao salvar memória: {str(e)}"


# ============== DEFINIÇÃO DAS FERRAMENTAS ==============

CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Calcula expressões matemáticas. Exemplos: '2 + 2', '15% of 250', '(10*5)/2'",
    parameters={
        "expression": {
            "type": "string",
            "description": "Expressão matemática para calcular"
        }
    },
    func=calculator_func,
    permission=ToolPermission.SAFE,
    max_uses_per_session=20
)

DATETIME_TOOL = Tool(
    name="current_datetime",
    description="Obtém a data e hora atuais do sistema",
    parameters={},
    func=datetime_func,
    permission=ToolPermission.SAFE,
    max_uses_per_session=10
)

SAVE_MEMORY_TOOL = Tool(
    name="save_memory",
    description="Salva informações importantes na memória de longo prazo do sistema",
    parameters={
        "text": {
            "type": "string",
            "description": "Informação para memorizar"
        },
        "importance": {
            "type": "number",
            "description": "Importância (0-2, padrão 1.0)",
            "default": 1.0
        }
    },
    func=save_memory_func,
    permission=ToolPermission.SAFE,
    max_uses_per_session=15
)

# Lista completa de ferramentas built-in
BUILTIN_TOOLS = [CALCULATOR_TOOL, DATETIME_TOOL, SAVE_MEMORY_TOOL, PROJECT_MANAGER_TOOL, GIT_WORKER_TOOL, PROCESS_ORCHESTRATOR_TOOL]
