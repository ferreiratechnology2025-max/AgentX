"""E2E smoke test — Orchestrator ponta a ponta via HTTP SSE.

Sobe o servidor uvicorn, envia um goal que força decomposição em múltiplos
roles (incluindo security), parseia o stream SSE e reporta o resultado.

Goal escolhido: review de um arquivo real do repo que toca security + code_review
+ architect, naturalmente próximo do threshold do judge (resposta checklist curta).
"""

import asyncio
import json
import subprocess
import sys
import time

import httpx

SERVER = "http://localhost:8000"
TIMEOUT_SERVER_START = 30   # s
TIMEOUT_GOAL = 360          # s — orquestração completa com voter

# Goal deliberadamente concreto: toca 3 roles, security produz checklist que
# o judge pode avaliar como borderline se for muito telegráfico
GOAL = (
    "Analise o arquivo agent/judge.py sob três ângulos: "
    "(1) verifique vulnerabilidades de segurança (role=security); "
    "(2) avalie a qualidade do código (role=code_review); "
    "(3) proponha uma melhoria de arquitetura (role=architect). "
    "Cada subtarefa deve ser respondida de forma concisa."
)

# Paleta de eventos para exibição
EVENT_ICONS = {
    "orchestrator_start":    "===>",
    "decomposition_start":   "...",
    "decomposition_complete": "[PLAN]",
    "subtask_start":         "  [>]",
    "subtask_complete":      "  [OK]",
    "judge_evaluation":      "  [J?]",
    "judge_result":          "  [J=]",
    "thought":               "    ~",
    "action":                "    T",
    "observation":           "    O",
    "final":                 "  [FA]",
    "synthesis_start":       "...",
    "orchestrator_complete": "[===]",
    "error":                 "[ERR]",
    "status":                "   .",
}


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.routes:app",
         "--host", "127.0.0.1", "--port", "8000",
         "--log-level", "warning"],
        cwd="D:/AgentX",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + TIMEOUT_SERVER_START
    while time.time() < deadline:
        try:
            r = httpx.get(f"{SERVER}/health", timeout=2)
            if r.status_code == 200:
                print(f"[server] ready (pid={proc.pid})")
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Servidor nao ficou pronto a tempo")


async def stream_goal(goal: str):
    events = []
    judge_results = []
    roles_seen = set()
    voter_fired = False
    tie_escalations = 0

    print(f"\nGoal: {goal[:100]}...\n")

    async with httpx.AsyncClient(timeout=TIMEOUT_GOAL) as client:
        async with client.stream(
            "POST", f"{SERVER}/orchestrator/run",
            json={"goal": goal},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                events.append(ev)
                t = ev.get("type", "?")
                icon = EVENT_ICONS.get(t, "   ?")

                # Display formatado por tipo
                if t == "orchestrator_start":
                    print(f"{icon} Orchestrator iniciado")

                elif t == "decomposition_complete":
                    subs = ev.get("subtasks", [])
                    print(f"{icon} {len(subs)} subtarefas:")
                    for s in subs:
                        print(f"         - [{s['role']}] {s['description'][:70]}")
                        roles_seen.add(s["role"])

                elif t == "subtask_start":
                    print(f"\n{icon} [{ev['role']}] {ev['description'][:60]}")

                elif t == "thought":
                    snippet = ev.get("content", "")[:80].replace("\n", " ")
                    tps = ev.get("telemetry", {}).get("throughput_tps", "?")
                    print(f"{icon} {snippet}  [{tps} tok/s]")

                elif t == "action":
                    print(f"{icon} {ev.get('tool')}({json.dumps(ev.get('arguments', {}))[:60]})")

                elif t == "observation":
                    print(f"{icon} {ev.get('content','')[:80]}")

                elif t == "final":
                    fa = ev.get("content", "")
                    print(f"{icon} {fa[:120]}")

                elif t == "judge_evaluation":
                    model = ev.get("judge_model", "?")
                    esc = ev.get("escalated", False)
                    flag = " [ESCALATED]" if esc else ""
                    print(f"{icon} [{model}]{flag}")

                elif t == "judge_result":
                    score = ev.get("score")
                    verdict = ev.get("verdict", "?")
                    voted = ev.get("voted", False)
                    tie = ev.get("escalated_via_tie", False)
                    if voted:
                        voter_fired = True
                    if tie:
                        tie_escalations += 1
                    vote_tag = " [VOTED]" if voted else ""
                    tie_tag  = " [TIE->REASONER]" if tie else ""
                    print(f"{icon} score={score} verdict={verdict}{vote_tag}{tie_tag}")
                    judge_results.append({
                        "score": score, "verdict": verdict,
                        "voted": voted, "tie": tie,
                    })

                elif t == "subtask_complete":
                    print(f"{icon} [{ev.get('subtask_id')}] "
                          f"status={ev.get('status')} score={ev.get('judge_score')} "
                          f"iters={ev.get('iterations')}")

                elif t == "orchestrator_complete":
                    fr = ev.get("final_result", "")
                    print(f"\n{icon} Resultado final:\n{fr[:400]}")
                    summary = ev.get("subtasks_summary", [])
                    print(f"\n{icon} Resumo de subtarefas:")
                    for s in summary:
                        print(f"         {s['id']}: {s['status']} | score={s['judge_score']} | iters={s['iterations']}")

                elif t == "error":
                    print(f"{icon} {ev.get('content')}")

    return events, judge_results, roles_seen, voter_fired, tie_escalations


def main():
    server_proc = None
    try:
        server_proc = start_server()
        loop = asyncio.new_event_loop()
        events, judge_results, roles_seen, voter_fired, tie_escalations = \
            loop.run_until_complete(stream_goal(GOAL))
        loop.close()

        print("\n" + "=" * 60)
        print("RESULTADO E2E")
        print("=" * 60)
        print(f"  Eventos totais:       {len(events)}")
        print(f"  Roles na decompose:   {roles_seen}")
        print(f"  Security voter ativo: {'SIM' if voter_fired else 'NAO (score nao borderline)'}")
        print(f"  Judge results:        {len(judge_results)}")
        for jr in judge_results:
            tags = []
            if jr["voted"]:       tags.append("VOTED")
            if jr["tie"]:         tags.append("TIE->REASONER")
            print(f"    score={jr['score']} verdict={jr['verdict']}"
                  f"{' [' + ', '.join(tags) + ']' if tags else ''}")
        print(f"  Tie escalations:      {tie_escalations}")

        # Critérios de sucesso
        checks = {
            "decomposição ocorreu":         any(e["type"] == "decomposition_complete" for e in events),
            "security role presente":       "security" in roles_seen,
            "pelo menos 2 roles":           len(roles_seen) >= 2,
            "judge avaliou":                len(judge_results) > 0,
            "resultado final sintetizado":  any(e["type"] == "orchestrator_complete" for e in events),
        }
        print("\n  Checks:")
        all_ok = True
        for check, passed in checks.items():
            mark = "OK" if passed else "FALHOU"
            print(f"    [{mark}] {check}")
            if not passed:
                all_ok = False

        print(f"\n  Veredicto: {'PASSOU' if all_ok else 'FALHOU'}")

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait(timeout=5)
            print("\n[server] encerrado")


if __name__ == "__main__":
    main()
