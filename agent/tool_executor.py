"""ToolExecutor — executes tools with permission checking, usage tracking, and output compaction."""
from typing import Dict, List, Optional
from tools.registry import ToolRegistry
from tools.base import Tool, ToolPermission, ToolResult


class PermissionRequiredException(Exception):
    """Lancada quando uma acao requer aprovacao do usuario."""
    pass


class ToolExecutor:
    """Executa ferramentas com verificacao de permissoes, limites de uso, e HITL."""

    def __init__(self, tool_registry: ToolRegistry, yolo_mode: bool = False):
        self.tool_registry = tool_registry
        self.yolo_mode = yolo_mode
        self.tool_usage: Dict[str, int] = {}
        self.permission_state: Optional[dict] = None  # preenchido quando pending approval

    def reset_usage_counts(self):
        self.tool_usage = {name: 0 for name in self.tool_registry.get_names()}

    async def execute(self, tool_name: str, arguments: dict, tool_id: str = "call_0") -> ToolResult:
        """Executa tool com verificacao de permissoes e compactacao de output."""
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name, success=False, output="",
                error=f"Ferramenta '{tool_name}' nao encontrada"
            )

        if self.tool_usage.get(tool_name, 0) >= tool.max_uses_per_session:
            return ToolResult(
                tool_name=tool_name, success=False, output="",
                error=f"Limite de {tool.max_uses_per_session} usos excedido para {tool_name}"
            )

        # INTERCEPTACAO DE SEGURANCA: CONFIRM tools
        if tool.permission == ToolPermission.CONFIRM:
            if not self.yolo_mode:
                self.permission_state = {
                    "call_id": tool_id,
                    "name": tool_name,
                    "arguments": arguments,
                }
                raise PermissionRequiredException(
                    f"Acao {tool_name} requer aprovacao"
                )

        # BARREIRA HITL: git_worker stage_and_commit (mesmo que SAFE)
        if (not self.yolo_mode and tool_name == "git_worker"
                and arguments.get("action") == "stage_and_commit"):
            self.permission_state = {
                "call_id": tool_id,
                "name": tool_name,
                "arguments": arguments,
            }
            raise PermissionRequiredException(
                "Commit ao repositorio requer aprovacao do operador"
            )

        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
        result = await tool.execute(**arguments)
        result.output = self._compact_output(result.output, tool_name)
        return result

    def _compact_output(self, raw: str, tool_name: str) -> str:
        """Compacta output de ferramenta para minimizar poluicao de contexto."""
        import json
        stripped = raw.strip()
        if stripped.startswith(("{", "[")):
            try:
                data = json.loads(stripped)
                compact = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
                if len(compact) < len(raw) * 0.7:
                    raw = compact
            except json.JSONDecodeError:
                pass
        MAX_OUTPUT = 600 if tool_name in ("project_manager", "save_memory") else 400
        if len(raw) > MAX_OUTPUT:
            raw = raw[:MAX_OUTPUT] + "... [truncado]"
        return raw
