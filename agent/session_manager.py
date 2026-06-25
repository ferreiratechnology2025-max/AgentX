"""Session checkpoint persistence — saves/loads compact state to disk"""

import os
import json
from datetime import datetime, timezone
from typing import Optional

from tools.schemas import SessionCheckpoint

SESSIONS_DIR = "./workspace/sessions"


def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def save_checkpoint(
    session_id: str,
    goal: str,
    status: str = "completed",
    steps_count: int = 0,
    summary: str = "",
    max_history: int = 6,
) -> None:
    """
    Grava snapshot compacto da sessão.
    Args:
        session_id: ID único
        goal: objetivo original
        status: completed | failed | running
        steps_count: total de steps
        summary: resumo executivo do que foi feito (até max_history steps)
    """
    _ensure_dir()

    # Limita o summary aos últimos N steps
    lines = summary.strip().split("\n")
    trimmed = "\n".join(lines[-max_history:]) if len(lines) > max_history else summary

    checkpoint = SessionCheckpoint(
        session_id=session_id,
        goal=goal,
        last_updated=datetime.now(timezone.utc).isoformat(),
        steps_count=steps_count,
        status=status,
        summary=trimmed,
    )

    with open(_path(session_id), "w", encoding="utf-8") as f:
        json.dump(
            checkpoint.model_dump(),
            f,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def load_checkpoint(session_id: str) -> Optional[SessionCheckpoint]:
    """Carrega checkpoint do disco. Retorna None se não existir ou estiver corrompido."""
    filepath = _path(session_id)
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionCheckpoint(**data)
    except (json.JSONDecodeError, IOError, Exception):
        return None


def list_sessions() -> list:
    """Lista IDs de sessões com checkpoint disponível."""
    _ensure_dir()
    return [
        f.replace(".json", "")
        for f in os.listdir(SESSIONS_DIR)
        if f.endswith(".json")
    ]


def delete_checkpoint(session_id: str) -> bool:
    """Remove checkpoint do disco."""
    filepath = _path(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
