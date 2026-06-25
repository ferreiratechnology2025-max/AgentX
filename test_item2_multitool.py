"""Item 2 — Retest multi-tool 5-10x contra qwen2.5:7b real.

Testa consistência do schema nativo (MAX_PARSE_ATTEMPTS=1) usando o
system prompt real do agente (build_system_prompt), que é o que acontece
em produção. Sem system prompt, action_name é string livre e o modelo inventa nomes.

Casos: calculator (5x) + project_manager (5x) = 10 chamadas no total.
Threshold de aprovação: ≥9/10 (90%).
"""

import asyncio
import sys
sys.path.insert(0, '.')

from llm.pool import get_llm_pool
from llm.manager import Message, GenerationFailedError
from agent.prompt_builder import build_system_prompt
from tools.builtin import CALCULATOR_TOOL
from tools.project_manager import PROJECT_MANAGER_TOOL

CALCULATOR_CASES = [
    "What is 2 + 3? Use the calculator tool.",
    "Calculate 144 / 12 using the calculator tool.",
    "Use the calculator tool: what is 17 * 8?",
    "What is the square root of 81? Use the calculator tool.",
    "Use the calculator tool to compute 1000 - 337.",
]

PROJECT_MANAGER_CASES = [
    "List all files in the workspace using the project_manager tool.",
    "Use project_manager to list the contents of the workspace directory.",
    "Check what files exist in the workspace. Use project_manager with action 'list'.",
    "Use the project_manager tool with action 'list' to see workspace files.",
    "List workspace files with the project_manager tool.",
]


async def run_case(
    llm, system_prompt: str, prompt: str, expected_tool: str, label: str
) -> dict:
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=prompt),
    ]
    try:
        output, usage = await llm.generate_with_validation(
            messages=messages,
            tools=[],          # tools já estão no system prompt; schema garante o formato
            max_tokens=256,
            temperature=0.7,
        )
        got_tool = output.action.name if output.action else "(final_answer)"
        success = output.action is not None and output.action.name == expected_tool
        tag = "PASS" if success else "FAIL"
        note = ""
        if output.action and output.action.name != expected_tool:
            note = f" → chamou '{output.action.name}'"
        elif not output.action:
            fa = (output.final_answer or "")[:70]
            note = f" → Final Answer em vez de tool: {fa}"
        print(f"  [{tag}] {label}: {prompt[:55]}...{note}")
        return {"success": success, "tool": got_tool, "expected": expected_tool,
                "tokens": usage.get("completion_tokens", 0)}
    except GenerationFailedError as e:
        print(f"  [FAIL] {label}: GenerationFailedError — {str(e)[:100]}")
        return {"success": False, "tool": "GenerationFailedError", "expected": expected_tool, "tokens": 0}
    except Exception as e:
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        return {"success": False, "tool": str(type(e).__name__), "expected": expected_tool, "tokens": 0}


async def main():
    print("=" * 65)
    print("Item 2 — Multi-tool retest (qwen2.5:7b, schema nativo)")
    print("=" * 65)

    pool = get_llm_pool()
    llm = await pool.get_model(role="general")

    # System prompts com os tools reais (igual ao agente em produção)
    calc_sys = build_system_prompt([CALCULATOR_TOOL])
    pm_sys   = build_system_prompt([PROJECT_MANAGER_TOOL])

    results = []

    print(f"\n[calculator — 5 casos]")
    for i, prompt in enumerate(CALCULATOR_CASES, 1):
        r = await run_case(llm, calc_sys, prompt, "calculator", f"calc-{i}")
        results.append(r)

    print(f"\n[project_manager — 5 casos]")
    for i, prompt in enumerate(PROJECT_MANAGER_CASES, 1):
        r = await run_case(llm, pm_sys, prompt, "project_manager", f"pm-{i}")
        results.append(r)

    total     = len(results)
    passed    = sum(1 for r in results if r["success"])
    calc_pass = sum(1 for r in results[:5] if r["success"])
    pm_pass   = sum(1 for r in results[5:] if r["success"])

    print("\n" + "=" * 65)
    print("RESULTADO")
    print("=" * 65)
    print(f"  calculator:      {calc_pass}/5")
    print(f"  project_manager: {pm_pass}/5")
    print(f"  TOTAL:           {passed}/{total} ({100 * passed // total}%)")

    if passed == total:
        print("\nItem 2: PASS — 100% consistência com schema nativo")
    elif passed >= int(total * 0.9):
        print(f"\nItem 2: PASS parcial — {passed}/{total} (≥90%)")
    else:
        print(f"\nItem 2: FAIL — {passed}/{total} abaixo do threshold de 90%")
        failures = [r for r in results if not r["success"]]
        print(f"  Ferramentas chamadas nos FAILs: {[r['tool'] for r in failures]}")

    print("\nItem 3: PENDENTE — Modelfile phi-4-mini-reasoning + TEMPLATE check")


if __name__ == "__main__":
    asyncio.run(main())
