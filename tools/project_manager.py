"""Project Manager tool - file operations within secure workspace"""

import os
import json
from pathlib import Path
from .base import Tool, ToolPermission
from .sandbox import SecurityError, ALLOWED_ROOTS, check_blocked, safe_resolve_path

WORKSPACE_DIR = os.path.abspath("./workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# Read-only access extended to project root so security/code_review agents can read source files.
# Write operations remain restricted to ALLOWED_ROOTS (workspace + data/knowledge) only.
PROJECT_ROOT = Path(".").resolve()
ALLOWED_READ_ROOTS = ALLOWED_ROOTS + [PROJECT_ROOT]


async def project_manager_func(action: str, path: str = "", content: str = None) -> str:
    """Gerencia arquivos de projeto, especificações e relatórios de progresso"""
    try:
        if action == "write":
            if not path:
                return "Erro: parâmetro 'path' é obrigatório para 'write'."
            if not content:
                return "Erro: Conteúdo vazio para ação de escrita."
            try:
                resolved = safe_resolve_path(path, ALLOWED_ROOTS)
                check_blocked(str(resolved))
            except SecurityError as e:
                return f"Acesso negado (escrita restrita ao workspace): {e}"
            os.makedirs(str(resolved.parent), exist_ok=True)
            with open(str(resolved), "w", encoding="utf-8") as f:
                f.write(content)
            return f"Sucesso: Arquivo '{path}' salvo/atualizado no workspace."

        elif action == "read":
            if not path:
                return "Erro: parâmetro 'path' é obrigatório para 'read'."
            try:
                resolved = safe_resolve_path(path, ALLOWED_READ_ROOTS)
                check_blocked(str(resolved))
            except SecurityError as e:
                return f"Acesso negado: {e}"
            if not resolved.exists():
                return f"Erro: Arquivo '{path}' não encontrado."
            conteudo_bruto = resolved.read_text(encoding="utf-8")
            try:
                dados_json = json.loads(conteudo_bruto)
                json_compacto = json.dumps(dados_json, separators=(',', ':'), ensure_ascii=False)
                if len(json_compacto) > 1500:
                    return (f"Conteúdo de {path} (JSON Compactado - Parcial):\n"
                            f"{json_compacto[:1500]}... [Cortado por Limite de Contexto]")
                return f"Conteúdo de {path} (JSON Minificado):\n{json_compacto}"
            except json.JSONDecodeError:
                if len(conteudo_bruto) > 1000:
                    return (f"Conteúdo de {path} (Texto Cortado):\n---\n"
                            f"{conteudo_bruto[:1000]}\n... [Truncado para preservar VRAM]")
                return f"Conteúdo de {path}:\n---\n{conteudo_bruto}\n---"

        elif action == "list":
            files = os.listdir(WORKSPACE_DIR)
            if not files:
                return "Workspace vazio. Nenhum arquivo de projeto encontrado."
            return "Arquivos no Workspace:\n" + "\n".join([f"- {file}" for file in files])

        else:
            return f"Erro: Ação '{action}' inválida. Use 'read', 'write' ou 'list'."

    except Exception as e:
        return f"Erro na operação de arquivo: {e}"


PROJECT_MANAGER_TOOL = Tool(
    name="project_manager",
    description=(
        "Lê, escreve e lista arquivos do workspace e do código-fonte do projeto. "
        "Use 'read' para consultar specs ou arquivos de código (ex: 'agent/judge.py', 'tools/base.py'); "
        "'write' para salvar resultados no workspace; 'list' para listar arquivos do workspace."
    ),
    parameters={
        "action": {
            "type": "string",
            "description": (
                "Ação a executar: 'read' (ler arquivo do workspace ou código-fonte), "
                "'write' (escrever no workspace) ou 'list' (listar arquivos do workspace)."
            )
        },
        "path": {
            "type": "string",
            "description": (
                "Caminho do arquivo. Para leitura: arquivo do workspace (ex: 'tasks.json') "
                "ou código-fonte (ex: 'agent/judge.py', 'api/routes.py'). "
                "Para escrita: caminho relativo dentro do workspace. Para 'list', omitir ou string vazia."
            ),
            "default": ""
        },
        "content": {
            "type": "string",
            "description": "Obrigatório apenas para 'write'. Conteúdo textual ou JSON stringificado.",
            "default": ""
        }
    },
    func=project_manager_func,
    permission=ToolPermission.CONFIRM,
    max_uses_per_session=10,
    timeout_seconds=5
)
