"""Base classes for tool system"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional
from enum import Enum
import asyncio


class ToolPermission(Enum):
    """Níveis de permissão para ferramentas"""
    SAFE = "safe"              # Execução sem confirmação
    CONFIRM = "confirm"        # Requer confirmação do usuário
    ADMIN = "admin"            # Apenas administradores


@dataclass
class ToolResult:
    """Resultado da execução de uma ferramenta"""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """Definição de uma ferramenta executável pelo agente"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    func: Callable
    permission: ToolPermission = ToolPermission.SAFE
    max_uses_per_session: int = 10
    timeout_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict no formato esperado pelo LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Executa a ferramenta com timeout e tratamento de erro"""
        try:
            # Executa com timeout
            result = await asyncio.wait_for(
                self.func(**kwargs),
                timeout=self.timeout_seconds
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=str(result)
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"[Timeout] Ferramenta '{self.name}' excedeu {self.timeout_seconds}s",
                error=f"Timeout após {self.timeout_seconds} segundos"
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"[Erro da ferramenta '{self.name}'] {e}",
                error=str(e)
            )
