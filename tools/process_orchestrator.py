"""Process orchestrator tool — start/status/stop dev runtimes safely"""
import asyncio
import os
import signal
from typing import List

from .base import Tool, ToolPermission

ALLOWED_COMMANDS = {"python", "uvicorn", "npm", "docker", "docker-compose", "poetry", "pip", "git", "node"}
BLOCKED_TOKENS = {"rm", "format", "sudo", "sh", "bash", "powershell", "cmd", "del", "rd", "rmdir", "wget", "curl"}


def _validate_safe(command: str, args: List[str]) -> None:
    """Valida que comando e argumentos não contêm tokens perigosos"""
    combined = command.lower()
    for a in args:
        combined += " " + a.lower()
    for token in BLOCKED_TOKENS:
        if token in combined.split():
            raise ValueError(f"Comando rejeitado por segurança: '{token}' não é permitido")
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Comando '{command}' não está na lista de runtimes homologados")


def _pid_exists(pid: int) -> bool:
    """Verifica se PID existe no SO de forma portável"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


async def process_orchestrator_func(action: str, command: str, args: List[str], pid: int = None) -> str:
    """Orquestra processos locais (start/status/stop)"""
    _validate_safe(command, args)

    if action == "start":
        process = await asyncio.create_subprocess_exec(
            command, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        pid = process.pid

        # Lê primeiras linhas com timeout curto para capturar log de inicialização
        lines = []
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(), timeout=2.0
            )
        except asyncio.TimeoutError:
            # Ainda rodando, captura o que produziu até agora
            if process.stdout:
                stdout_data = b"[Processo continua rodando em background...]"
            else:
                stdout_data = b""
            stderr_data = b""

        if stdout_data:
            out_text = stdout_data.decode("utf-8", errors="replace")
            lines = out_text.split("\n")[:5]
        if stderr_data:
            err_text = stderr_data.decode("utf-8", errors="replace")
            if lines:
                lines.append(f"stderr: {err_text[:200]}")
            else:
                lines = [f"stderr: {err_text[:200]}"]

        initial_log = "\n".join(lines) if lines else "(sem saída inicial)"
        return f"✅ Processo iniciado (PID={pid})\n{initial_log}"

    elif action == "status":
        if pid is None:
            return "Erro: PID é obrigatório para ação 'status'"
        running = _pid_exists(pid)
        if running:
            return f"✅ Processo {pid} está rodando"
        else:
            return f"❌ Processo {pid} não está rodando"

    elif action == "stop":
        if pid is None:
            return "Erro: PID é obrigatório para ação 'stop'"
        try:
            os.kill(pid, signal.SIGTERM)
            # Aguarda breve para encerramento limpo
            for _ in range(5):
                await asyncio.sleep(0.2)
                if not _pid_exists(pid):
                    return f"✅ Processo {pid} encerrado com sucesso"
            # Fallback: SIGKILL no Unix, TerminateProcess no Windows
            os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
            return f"✅ Processo {pid} encerrado (forçado)"
        except ProcessLookupError:
            return f"❌ Processo {pid} não encontrado"
        except PermissionError:
            return f"❌ Sem permissão para encerrar processo {pid}"
    else:
        return f"Erro: Ação '{action}' inválida. Use 'start', 'status' ou 'stop'."


PROCESS_ORCHESTRATOR_TOOL = Tool(
    name="process_orchestrator",
    description="Inicia, verifica ou derruba processos locais (ex: servidores de teste, scripts). Comandos homologados: python, uvicorn, npm, docker, docker-compose, poetry, pip, git, node.",
    parameters={
        "action": {
            "type": "string",
            "enum": ["start", "status", "stop"],
            "description": "Ação: start (iniciar), status (verificar PID), stop (derrubar)"
        },
        "command": {
            "type": "string",
            "description": "Runtime homologado (ex: python, uvicorn, npm, docker-compose)"
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Argumentos do comando (ex: ['main.py'], ['up', '-d'])"
        },
        "pid": {
            "type": "integer",
            "description": "PID do processo (obrigatório para status e stop)"
        }
    },
    func=process_orchestrator_func,
    permission=ToolPermission.CONFIRM,
    max_uses_per_session=15
)
