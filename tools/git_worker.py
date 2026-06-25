"""Git worker tool — controlled git operations (status, stage, commit)"""

import asyncio
import subprocess
import os

from .base import Tool, ToolPermission


async def git_worker_func(action: str, files: list = None, commit_message: str = None) -> str:
    """Executa operações Git controladas no repositório local."""
    repo_path = os.path.abspath(os.getcwd())

    try:
        if action == "status":
            def _status():
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, check=True, cwd=repo_path,
                )
                return result.stdout if result.stdout else "Repositório limpo, nada para commitar."
            return await asyncio.to_thread(_status)

        elif action == "stage_and_commit":
            if not files or not commit_message:
                return "Erro: 'files' e 'commit_message' são obrigatórios para esta ação."

            for file in files:
                if "env" in file or "private" in file or ".env" in file:
                    return f"Erro: Bloqueio de segurança. O arquivo '{file}' não pode ser comitado."

            def _stage_and_commit():
                for file in files:
                    subprocess.run(
                        ["git", "add", file],
                        capture_output=True, text=True, check=True, cwd=repo_path,
                    )
                result = subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    capture_output=True, text=True, check=True, cwd=repo_path,
                )
                return f"🚀 Commit realizado com sucesso!\n{result.stdout}"

            return await asyncio.to_thread(_stage_and_commit)

    except subprocess.CalledProcessError as e:
        erro = e.stderr if e.stderr else e.stdout
        return f"⚠️ Erro ao executar comando Git: {erro[:500]}"


GIT_WORKER_TOOL = Tool(
    name="git_worker",
    description="Executa operações Git controladas: 'status' (ver estado do repositório) ou 'stage_and_commit' (adicionar arquivos e commitar). Use sempre que alterar arquivos do workspace.",
    parameters={
        "action": {
            "type": "string",
            "enum": ["status", "stage_and_commit"],
            "description": "Ação Git a ser executada."
        },
        "files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista de arquivos para dar stage (obrigatório para stage_and_commit)."
        },
        "commit_message": {
            "type": "string",
            "description": "Mensagem de commit no formato Conventional Commits (obrigatório para stage_and_commit)."
        }
    },
    func=git_worker_func,
    permission=ToolPermission.CONFIRM,
    max_uses_per_session=10,
    timeout_seconds=30,
)
