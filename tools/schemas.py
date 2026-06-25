"""Pydantic schemas for structured validation"""

from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Dict, Any, Optional, Literal


class ReActToolCall(BaseModel):
    """Chamada de ferramenta extraida do output do LLM"""
    name: str
    arguments: Dict[str, Any]


class ReActOutput(BaseModel):
    """Output validado do LLM no formato ReAct JSON"""
    thought: str
    action: Optional[ReActToolCall] = None
    final_answer: Optional[str] = None

    @validator("final_answer")
    def check_action_or_final(cls, v, values):
        action = values.get("action")
        if action is None and v is None:
            raise ValueError("Deve ter action ou final_answer")
        return v


class TelemetryData(BaseModel):
    """Métricas de performance de uma chamada de inferência"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    throughput_tps: float = 0.0


class GitWorkerArgs(BaseModel):
    """Validação estrita para a ferramenta git_worker"""
    action: Literal["status", "stage_and_commit"] = Field(
        ..., description="Ação Git a ser executada."
    )
    files: Optional[List[str]] = Field(
        None, description="Lista de arquivos para dar stage (obrigatório se action for 'stage_and_commit')."
    )
    commit_message: Optional[str] = Field(
        None, description="Mensagem de commit no padrão Conventional Commits (ex: 'feat(core): conclui marco m2')."
    )


class ProcessOrchestratorArgs(BaseModel):
    """Validação estrita para a ferramenta process_orchestrator"""
    action: Literal["start", "status", "stop"] = Field(
        ..., description="Ação a executar: start (iniciar processo), status (verificar PID), stop (derrubar processo)"
    )
    command: str = Field(
        ..., description="Comando do runtime (ex: python, uvicorn, npm, docker-compose)"
    )
    args: list[str] = Field(
        default_factory=list, description="Argumentos a passar para o comando (ex: ['main.py'], ['up', '-d'])"
    )
    pid: Optional[int] = Field(
        None, description="PID do processo (obrigatório para ações 'status' e 'stop')"
    )


class SessionCheckpoint(BaseModel):
    """Snapshot serializável do estado da sessão para persistência pós-reboot"""
    session_id: str = Field(..., description="ID único da sessão")
    goal: str = Field(..., description="Objetivo original da sessão")
    last_updated: str = Field(..., description="Timestamp ISO da última interação")
    steps_count: int = Field(0, description="Total de steps executados")
    status: str = Field("running", description="Status final da sessão")
    summary: str = Field("", description="Sumário executivo do que foi feito")
