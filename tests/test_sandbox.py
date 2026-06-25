"""Tests for filesystem sandbox"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.sandbox import safe_resolve_path, check_blocked, SecurityError, ALLOWED_ROOTS


def test_allowed_path():
    """Paths dentro de workspace/ devem funcionar."""
    path = safe_resolve_path("workspace/test.txt")
    assert path == (Path("workspace/test.txt").resolve()), f"Unexpected: {path}"
    print("  PASS: allowed path")


def test_path_traversal_blocked():
    """Path com ../ que sai do workspace deve falhar."""
    try:
        safe_resolve_path("workspace/../../private/.env")
        assert False, "Deveria ter levantado SecurityError"
    except SecurityError as e:
        assert "Path escape" in str(e)
        print("  PASS: path traversal blocked")


def test_blocked_env_pattern():
    """Path contendo .env deve ser bloqueado."""
    try:
        check_blocked("workspace/private/.env")
        assert False, "Deveria ter levantado SecurityError"
    except SecurityError as e:
        assert "bloqueado" in str(e)
        print("  PASS: .env pattern blocked")


def test_blocked_private_pattern():
    """Path contendo private/ deve ser bloqueado."""
    try:
        check_blocked("private/config.key")
        assert False, "Deveria ter levantado SecurityError"
    except SecurityError as e:
        assert "bloqueado" in str(e)
        print("  PASS: private/ pattern blocked")


def test_blocked_models_pattern():
    """Path contendo models/ deve ser bloqueado."""
    try:
        check_blocked("models/model.gguf")
        assert False, "Deveria ter levantado SecurityError"
    except SecurityError as e:
        assert "bloqueado" in str(e)
        print("  PASS: models/ pattern blocked")


def test_symlink_attack():
    """Symlink dentro do workspace apontando para fora deve falhar."""
    workspace = Path("./workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Cria um arquivo fora do workspace para o symlink apontar
    outside = Path("./outside_test.txt").resolve()
    try:
        outside.write_text("secreto")
        
        # Cria symlink dentro do workspace
        link_path = workspace / "evil_link.txt"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        os.symlink(str(outside), str(link_path))
        
        try:
            safe_resolve_path(str(link_path))
            assert False, "Deveria ter levantado SecurityError (symlink escape)"
        except SecurityError as e:
            assert "Path escape" in str(e)
            print("  PASS: symlink attack blocked")
            
    except OSError as e:
        # Symlink pode nao ser suportado no Windows sem admin
        print(f"  SKIP: symlink test (OS error: {e})")
    finally:
        if outside.exists():
            outside.unlink()
        if link_path.exists() or (hasattr(link_path, 'is_symlink') and link_path.is_symlink()):
            try:
                link_path.unlink()
            except:
                pass


if __name__ == "__main__":
    print("Sandbox Tests:")
    test_allowed_path()
    test_path_traversal_blocked()
    test_blocked_env_pattern()
    test_blocked_private_pattern()
    test_blocked_models_pattern()
    test_symlink_attack()
    print("\nAll tests passed!")
