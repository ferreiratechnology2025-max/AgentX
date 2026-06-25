"""Benchmark de throughput real — Granite router + Gemma workhorse.

Mede tokens/seg no caso comum (sem escalada):
- granite4htiny: prompts curtos de roteamento (o que o router faz de verdade)
- gemma3ne4b:    prompts de ReAct com tool schema (o que o workhorse faz de verdade)

Metodologia:
- 1 chamada de warm-up descartada (carregamento do modelo)
- N=5 chamadas cronometradas por modelo
- tokens/seg calculado como completion_tokens / latência_real
- GPU medida antes e depois
"""

import asyncio
import sys
import time
import statistics
import subprocess

sys.path.insert(0, ".")

from llm.manager import LLMManager
from agent.prompt_builder import build_system_prompt
from tools.builtin import CALCULATOR_TOOL
from tools.project_manager import PROJECT_MANAGER_TOOL

OLLAMA_BASE = "http://localhost:11434"
N_RUNS = 5

# Prompts representativos de cada tier
ROUTER_PROMPTS = [
    "Classify this request: 'write a Python function to sort a list'. Role: coding or general?",
    "Classify: 'what is the capital of France'. Role: research or general?",
    "Classify: 'review this code for bugs'. Role: code_review or security?",
    "Classify: 'create unit tests for auth module'. Role: tdd or coding?",
    "Classify: 'design database schema for e-commerce'. Role: architect or planning?",
]

WORKHORSE_PROMPT = (
    "Calculate the total cost: 3 items at $24.99 each, plus 8% tax. "
    "Use the calculator tool to get the exact result."
)


def gpu_stats() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"], text=True
        ).strip()
        used, free, util = out.split(", ")
        return f"VRAM {used} used / {free} free | GPU {util}"
    except Exception:
        return "nvidia-smi unavailable"


def unload_model(model_id: str) -> None:
    """Força o Ollama a descarregar o modelo da VRAM (keep_alive=0)."""
    import requests as _r
    try:
        _r.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model_id, "keep_alive": 0},
            timeout=15,
        )
        print(f"  unloaded {model_id}")
    except Exception as e:
        print(f"  unload {model_id} warning: {e}")


async def bench_generate(llm: LLMManager, prompts: list[str], label: str) -> dict:
    """Warm-up + N runs cronometrados em generate() simples."""
    print(f"\n  [{label}] warm-up...", end=" ", flush=True)
    await llm.generate(prompts[0], max_tokens=64, temperature=0.1)
    print("done")

    tps_list = []
    latencies = []
    tokens_list = []

    for i, prompt in enumerate(prompts[:N_RUNS]):
        t0 = time.perf_counter()
        text, usage = await llm.generate(prompt, max_tokens=64, temperature=0.1)
        elapsed = time.perf_counter() - t0

        ctok = usage.get("completion_tokens", 0)
        tps = round(ctok / max(0.001, elapsed), 1)
        tps_list.append(tps)
        latencies.append(round(elapsed * 1000))
        tokens_list.append(ctok)
        print(f"    run {i+1}: {ctok} tokens / {elapsed:.2f}s = {tps} tok/s")

    return {
        "label": label,
        "tps_mean": round(statistics.mean(tps_list), 1),
        "tps_median": round(statistics.median(tps_list), 1),
        "tps_min": min(tps_list),
        "tps_max": max(tps_list),
        "lat_mean_ms": round(statistics.mean(latencies)),
        "tokens_mean": round(statistics.mean(tokens_list), 1),
    }


async def bench_generate_with_validation(llm: LLMManager, label: str) -> dict:
    """Warm-up + N runs cronometrados em generate_with_validation (schema nativo)."""
    from llm.manager import Message
    from agent.prompt_builder import build_system_prompt

    tool_objects = [CALCULATOR_TOOL, PROJECT_MANAGER_TOOL]
    tools = [t.to_dict() for t in tool_objects]
    sys_prompt = build_system_prompt(tool_objects)
    messages = [
        Message(role="system", content=sys_prompt),
        Message(role="user", content=WORKHORSE_PROMPT),
    ]

    print(f"\n  [{label}] warm-up...", end=" ", flush=True)
    await llm.generate_with_validation(messages, tools=tools, max_tokens=256, temperature=0.7)
    print("done")

    tps_list = []
    latencies = []
    tokens_list = []

    for i in range(N_RUNS):
        t0 = time.perf_counter()
        output, usage = await llm.generate_with_validation(
            messages, tools=tools, max_tokens=256, temperature=0.7
        )
        elapsed = time.perf_counter() - t0

        ctok = usage.get("completion_tokens", 0)
        tps = round(ctok / max(0.001, elapsed), 1)
        tps_list.append(tps)
        latencies.append(round(elapsed * 1000))
        tokens_list.append(ctok)
        action = output.action.name if output and output.action else "final_answer"
        print(f"    run {i+1}: {ctok} tokens / {elapsed:.2f}s = {tps} tok/s  [{action}]")

    return {
        "label": label,
        "tps_mean": round(statistics.mean(tps_list), 1),
        "tps_median": round(statistics.median(tps_list), 1),
        "tps_min": min(tps_list),
        "tps_max": max(tps_list),
        "lat_mean_ms": round(statistics.mean(latencies)),
        "tokens_mean": round(statistics.mean(tokens_list), 1),
    }


async def main():
    print("=" * 60)
    print("Throughput Benchmark — Granite router + Gemma workhorse")
    print("=" * 60)
    print(f"\nGPU baseline: {gpu_stats()}")

    def make_llm(model_id, num_ctx):
        return LLMManager({
            "llm": {"model_id": model_id, "num_ctx": num_ctx},
            "ollama": {"base_url": OLLAMA_BASE},
        })

    granite = make_llm("granite4htiny", num_ctx=4096)
    gemma   = make_llm("gemma3ne4b",    num_ctx=8192)

    # Unload qualquer modelo residual antes de começar
    print("\nUnloading residual models...")
    unload_model("granite4htiny")
    unload_model("gemma3ne4b")
    unload_model("qwen35-9b")
    await asyncio.sleep(2)
    print(f"  GPU clean: {gpu_stats()}")

    print("\n--- granite4htiny (router, routing prompts) ---")
    r_granite = await bench_generate(granite, ROUTER_PROMPTS, "granite4htiny")
    print(f"  GPU: {gpu_stats()}")

    # Descarrega granite antes de carregar gemma — evita OOM
    print("\nUnloading granite4htiny before gemma bench...")
    unload_model("granite4htiny")
    await asyncio.sleep(2)
    print(f"  GPU after unload: {gpu_stats()}")

    print("\n--- gemma3ne4b (workhorse, ReAct + schema) ---")
    r_gemma = await bench_generate_with_validation(gemma, "gemma3ne4b")
    print(f"  GPU: {gpu_stats()}")

    unload_model("gemma3ne4b")

    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    for r in [r_granite, r_gemma]:
        tag = ""
        if r["tps_mean"] >= 15:
            tag = " [interativo OK]"
        elif r["tps_mean"] >= 8:
            tag = " [aceitavel]"
        else:
            tag = " [lento para interativo]"
        print(f"\n  {r['label']}")
        print(f"    tok/s: mean={r['tps_mean']}  median={r['tps_median']}  "
              f"min={r['tps_min']}  max={r['tps_max']}{tag}")
        print(f"    latencia media: {r['lat_mean_ms']} ms")
        print(f"    tokens/req: {r['tokens_mean']}")

    print(f"\n  GPU final: {gpu_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
