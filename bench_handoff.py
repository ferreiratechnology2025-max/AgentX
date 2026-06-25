"""Benchmark de handoff router→workhorse com ambos os modelos residentes."""

import asyncio
import sys
import time
import statistics
import subprocess

import requests

sys.path.insert(0, ".")

from llm.manager import LLMManager, Message
from tools.builtin import CALCULATOR_TOOL
from tools.project_manager import PROJECT_MANAGER_TOOL
from agent.prompt_builder import build_system_prompt

OLLAMA = "http://localhost:11434"
N = 5

ROUTER_PROMPTS = [
    "Classify: 'sort a list in Python'. Role: coding or general?",
    "Classify: 'capital of France'. Role: research or general?",
    "Classify: 'review code for bugs'. Role: code_review or security?",
    "Classify: 'write unit tests'. Role: tdd or coding?",
    "Classify: 'design database schema'. Role: architect or planning?",
]

WORKHORSE_PROMPT = "Calculate the total cost: 3 items at $24.99 each, plus 8% tax."


def gpu():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        text=True,
    ).strip()
    return out


def loaded_models():
    ps = requests.get(OLLAMA + "/api/ps").json()
    return [m["name"] for m in ps.get("models", [])]


async def main():
    granite = LLMManager({"llm": {"model_id": "granite4htiny", "num_ctx": 4096},
                          "ollama": {"base_url": OLLAMA}})
    gemma   = LLMManager({"llm": {"model_id": "gemma3ne4b",    "num_ctx": 8192},
                          "ollama": {"base_url": OLLAMA}})

    tool_objs = [CALCULATOR_TOOL, PROJECT_MANAGER_TOOL]
    tools = [t.to_dict() for t in tool_objs]
    sys_p = build_system_prompt(tool_objs)
    msgs = [
        Message(role="system", content=sys_p),
        Message(role="user", content=WORKHORSE_PROMPT),
    ]

    print(f"Baseline: {gpu()}")

    print("\nWarm-up granite...", end=" ", flush=True)
    await granite.generate("ping", max_tokens=4, temperature=0.0)
    print(f"done  [{gpu()}]")

    print("Warm-up gemma...", end=" ", flush=True)
    await gemma.generate_with_validation(msgs, tools=tools, max_tokens=256, temperature=0.7)
    print(f"done  [{gpu()}]")

    print(f"Modelos residentes: {loaded_models()}")

    print("\n--- Handoff intercalado router->workhorse (N=5 pares) ---")
    router_tps, worker_tps = [], []
    router_lat, worker_lat = [], []

    for i in range(N):
        # Router call
        t0 = time.perf_counter()
        _, u = await granite.generate(ROUTER_PROMPTS[i], max_tokens=64, temperature=0.1)
        rt = time.perf_counter() - t0
        rtps = round(u.get("completion_tokens", 0) / max(0.001, rt), 1)
        router_tps.append(rtps)
        router_lat.append(round(rt * 1000))

        # Workhorse call — sem reload, ambos residentes
        t0 = time.perf_counter()
        _, u = await gemma.generate_with_validation(
            msgs, tools=tools, max_tokens=256, temperature=0.7
        )
        wt = time.perf_counter() - t0
        wtps = round(u.get("completion_tokens", 0) / max(0.001, wt), 1)
        worker_tps.append(wtps)
        worker_lat.append(round(wt * 1000))

        print(
            f"  iter {i+1}: router {rtps:5.1f} tok/s ({router_lat[-1]:4d}ms)"
            f" | workhorse {wtps:5.1f} tok/s ({worker_lat[-1]:5d}ms)"
        )

    still_loaded = loaded_models()
    print(f"\nGPU final: {gpu()}")
    print(f"Modelos ainda carregados: {still_loaded}")

    print("\n=== RESULTADO ===")
    print(f"  granite4htiny  mean={round(statistics.mean(router_tps),1):5.1f} tok/s"
          f"  lat={round(statistics.mean(router_lat)):4d} ms")
    print(f"  gemma3ne4b     mean={round(statistics.mean(worker_tps),1):5.1f} tok/s"
          f"  lat={round(statistics.mean(worker_lat)):5d} ms")

    both_resident = len(still_loaded) == 2
    print(f"\n  Handoff overhead: {'ZERO — ambos residentes durante todo o benchmark' if both_resident else 'ATENCAO — evicao ocorreu, modelo foi recarregado'}")
    print(f"  VRAM com ambos: {gpu().split(',')[0].strip()} / 12288 MiB")


if __name__ == "__main__":
    asyncio.run(main())
