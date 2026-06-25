"""Centralised filesystem sandbox — path validation + decorator"""

import os
import re
from pathlib import Path
from functools import wraps
from typing import List, Optional


class SecurityError(Exception):
    pass


ALLOWED_ROOTS = [
    Path("./workspace").resolve(),
    Path("./data/knowledge").resolve(),
]

BLOCKED_PATTERNS = [
    "private/",
    ".env",
    "models/",
    "__pycache__",
]


WORKSPACE_ROOT = Path("./workspace").resolve()


def safe_resolve_path(path: str, allowed_roots: Optional[List[Path]] = None) -> Path:
    """Resolve path e valida que esta dentro de allowed_roots.

    Se path e relativo e nao comeca com 'workspace/', resolve contra workspace/.
    Se ja comeca com 'workspace/', resolve normalmente.
    Critico: resolve symlinks ANTES de validar.
    """
    path_obj = Path(path)

    if not path_obj.is_absolute():
        parts = path_obj.parts
        if parts and parts[0] != "workspace":
            path_obj = WORKSPACE_ROOT / path_obj

    resolved = path_obj.resolve()
    roots = allowed_roots or ALLOWED_ROOTS

    for allowed_root in roots:
        allowed_resolved = allowed_root.resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return resolved
        except ValueError:
            continue

    raise SecurityError(f"Path escape: {path} -> {resolved}")


def check_blocked(path: str) -> None:
    """Verifica se path contem padrões bloqueados."""
    path_lower = path.lower().replace("\\", "/")
    for pattern in BLOCKED_PATTERNS:
        if pattern in path_lower:
            raise SecurityError(f"Path bloqueado: {path}")


def sandboxed_file_io(func):
    """Decorator que valida paths antes de operações de arquivo.
    
    Espera que a função tenha um argumento 'filename' (project_manager)
    ou 'path' contendo o caminho do arquivo.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extrai filename dos kwargs ou args
        filename = kwargs.get("filename", None)
        if filename is None and len(args) >= 2:
            # Segundo arg posicional é normalmente o filename
            pos_args = list(args)
            if len(pos_args) >= 2:
                filename = pos_args[1]

        if filename and filename.strip():
            # Resolve e valida o path
            safe_path = safe_resolve_path(filename)
            # Verifica padrões bloqueados
            check_blocked(str(safe_path))
            # Atualiza o filename nos kwargs/args
            if "filename" in kwargs:
                kwargs["filename"] = str(safe_path)
            else:
                args = (args[0], str(safe_path)) + args[2:]

        return await func(*args, **kwargs)
    return wrapper
