"""Garbage collector for expired session checkpoints (>7 days)"""

import os
import json
from datetime import datetime, timedelta, timezone

SESSIONS_DIR = "./workspace/sessions"
LIMITE_DIAS = 7


def run_garbage_collector(dry_run: bool = False) -> dict:
    """
    Varre o diretório de sessões e remove checkpoints cujo last_updated
    seja anterior a LIMITE_DIAS. Também remove arquivos corrompidos.

    :param dry_run: Se True, apenas simula sem deletar.
    """
    if not os.path.exists(SESSIONS_DIR):
        return {"status": "success", "arquivos_deletados": 0, "espaco_liberado_bytes": 0}

    agora = datetime.now(timezone.utc)
    limite = agora.replace(tzinfo=None) - timedelta(days=LIMITE_DIAS)

    deletados = 0
    espaco = 0

    print(f" [GC] Varrendo {SESSIONS_DIR} (limite: {LIMITE_DIAS} dias, dry_run={dry_run})")

    for filename in os.listdir(SESSIONS_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(SESSIONS_DIR, filename)
        remover = False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            last = data.get("last_updated")
            if last:
                last_dt = datetime.fromisoformat(last.replace("Z", ""))
                if last_dt.tzinfo:
                    last_dt = last_dt.replace(tzinfo=None)
                if last_dt < limite:
                    remover = True
                    motivo = f"Inativo desde {last}"
        except (json.JSONDecodeError, KeyError, ValueError, IOError):
            remover = True
            motivo = "Arquivo corrompido"

        if remover:
            tamanho = os.path.getsize(filepath)
            espaco += tamanho
            deletados += 1
            if not dry_run:
                os.remove(filepath)
                print(f"    {filename} ({motivo}, {tamanho}B)")
            else:
                print(f"   [DRY] {filename} ({motivo}, {tamanho}B)")

    print(f" [GC] Concluído. Removidos: {deletados} | Espaço: {espaco}B")
    return {
        "status": "success",
        "dry_run": dry_run,
        "arquivos_deletados": deletados,
        "espaco_liberado_bytes": espaco,
    }


if __name__ == "__main__":
    run_garbage_collector(dry_run=False)
