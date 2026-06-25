"""Tool registry for managing available tools"""

from typing import Dict, List, Optional
from .base import Tool
from .builtin import BUILTIN_TOOLS


class ToolRegistry:
    """Gerencia centralizadamente o registro de ferramentas"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._load_builtin_tools()
    
    def _load_builtin_tools(self):
        """Carrega todas as ferramentas built-in"""
        for tool in BUILTIN_TOOLS:
            self.register(tool)
    
    def register(self, tool: Tool) -> None:
        """Registra uma nova ferramenta"""
        if tool.name in self._tools:
            print(f"Aviso: Substituindo ferramenta existente: {tool.name}")
        self._tools[tool.name] = tool
        print(f"Ferramenta registrada: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        """Remove uma ferramenta do registro"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Obtém uma ferramenta pelo nome"""
        return self._tools.get(name)
    
    def list_all(self) -> List[Tool]:
        """Lista todas as ferramentas registradas"""
        return list(self._tools.values())
    
    def to_llm_spec(self) -> List[Dict]:
        """Converte todas as ferramentas para formato LLM"""
        return [tool.to_dict() for tool in self._tools.values()]
    
    def get_names(self) -> List[str]:
        """Retorna lista de nomes das ferramentas"""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
