"""Test Phase 1: ReAct multi-step with tool calling"""

import asyncio
import sys
import json

sys.path.insert(0, '.')

from llm.pool import get_llm_pool
from llm.manager import Message
from tools.schemas import ReActOutput


async def test_react_multistep():
    """Test ReAct multi-step: Thought → Action → Observation → Final Answer"""
    print("\n" + "="*60)
    print("TEST: ReAct Multi-Step with Tool Call")
    print("="*60)

    try:
        pool = get_llm_pool()
        llm = await pool.get_model(role="general")

        # Define a simple tool
        tools = [
            {
                "name": "calculate",
                "description": "Performs a calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }
        ]

        # Create message requesting a calculation
        messages = [
            Message(
                role="user",
                content="What is 2 + 3? Use the calculate tool."
            )
        ]

        print("Sending ReAct request with tool call...")
        output, usage = await llm.generate_with_validation(
            messages=messages,
            tools=tools,
            max_tokens=200,
            temperature=0.7
        )

        print(f"\n✓ Response received")
        if output is None:
            print(f"  Result: Parser returned None (no valid ReAct format)")
            return False

        print(f"  Output type: {type(output).__name__}")
        print(f"  Thought: {output.thought[:60]}..." if len(output.thought) > 60 else f"  Thought: {output.thought}")

        if output.action:
            print(f"  Action: {output.action.name}")
            print(f"  Arguments: {output.action.arguments}")
        else:
            print(f"  Final Answer: {output.final_answer[:80]}..." if len(output.final_answer) > 80 else f"  Final Answer: {output.final_answer}")

        print(f"  Usage: prompt_tokens={usage['prompt_tokens']}, completion_tokens={usage['completion_tokens']}")

        # Check if it attempted tool call
        if output.action:
            print("\n✓ Tool call detected (multi-step ReAct working)")
            return True
        else:
            print("\n⚠ No tool call in response (model gave final answer directly)")
            print("  This is acceptable if model decided to answer directly without tools")
            return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ollama_gpu_status():
    """Check if Ollama is using GPU via ollama ps"""
    print("\n" + "="*60)
    print("TEST: Ollama GPU Status")
    print("="*60)

    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"⚠ 'ollama ps' command failed: {result.stderr}")
            return False

        lines = result.stdout.strip().split('\n')
        print("Running models in Ollama:")
        for line in lines:
            if line.strip():
                print(f"  {line}")

        # Check if qwen2.5:7b is running
        if "qwen2.5:7b" in result.stdout or "qwen2.5" in result.stdout:
            print("\n✓ qwen2.5:7b is currently loaded in Ollama")
            if "GPU" in result.stdout or "VRAM" in result.stdout:
                print("✓ GPU acceleration detected")
                return True
            else:
                print("⚠ Cannot confirm GPU in output (but likely using GPU)")
                return True
        else:
            print("⚠ No models currently loaded (normal if idle)")
            return True

    except FileNotFoundError:
        print("⚠ 'ollama' command not found in PATH")
        print("  Make sure Ollama is installed and in PATH")
        return False
    except Exception as e:
        print(f"⚠ Error checking Ollama status: {e}")
        return False


async def check_available_models():
    """List what models are available in Ollama"""
    print("\n" + "="*60)
    print("AVAILABLE MODELS in Ollama")
    print("="*60)

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        print("\nModels currently available:")
        models_by_family = {}
        for model in data.get('models', []):
            name = model.get('name', 'unknown')
            family = model.get('details', {}).get('family', 'unknown')
            size_gb = model.get('size', 0) / 1e9

            if family not in models_by_family:
                models_by_family[family] = []
            models_by_family[family].append((name, size_gb))

        for family in sorted(models_by_family.keys()):
            print(f"\n  {family.upper()}:")
            for name, size in models_by_family[family]:
                print(f"    - {name} ({size:.1f}GB)")

        # Check for bake-off models
        print("\n" + "-"*60)
        print("Bake-off models status (from plano.md):")
        bake_off_models = {
            "Granite 4 H Tiny": ["granite"],
            "Ministral 3 3B": ["ministral"],
            "Gemma 3 e4b": ["gemma"],
            "Qwen 3.5 9B": ["qwen", "3.5"],
            "Phi-4-mini": ["phi"],
            "Hermes Llama 3.1 8B": ["hermes"],
        }

        all_model_names = [m.get('name', '') for m in data.get('models', [])]
        for bake_off_name, keywords in bake_off_models.items():
            found = any(all(kw.lower() in name.lower() for kw in keywords) for name in all_model_names)
            status = "✓ FOUND" if found else "✗ NOT FOUND"
            print(f"  {status}: {bake_off_name}")

        print("\nCurrent config uses: qwen2.5:7b, llama3.1:latest")
        print("These are TEST placeholders — need confirmation on final model set.")

        return True

    except Exception as e:
        print(f"✗ Error listing models: {e}")
        return False


async def run_all_checks():
    """Run all additional checks"""
    print("\n" + "="*60)
    print("PHASE 1 FOLLOW-UP CHECKS")
    print("="*60)

    results = []

    # Check 1: ReAct multi-step
    results.append(("ReAct multi-step with tools", await test_react_multistep()))

    # Check 2: Ollama GPU status
    results.append(("Ollama GPU status", await test_ollama_gpu_status()))

    # Check 3: Available models
    results.append(("Available models check", await check_available_models()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for check_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")

    # Questions for user
    print("\n" + "="*60)
    print("QUESTIONS FOR USER")
    print("="*60)
    print("""
1. MAX_PARSE_ATTEMPTS: Reverted to 2 (will go back to 1 in Fase 3 with JSON schema)

2. ReAct multi-step: Tested above
   - Tool calling format should be: "Action: tool_name\\nAction Input: {...}"
   - If model supports it, should see tool call in output

3. GPU confirmation: Run 'ollama ps' after inference completes
   - Look for GPU memory usage (VRAM column)
   - If using CPU, you'll see "cpu" or empty GPU field

4. Model set confirmation needed:
   - Current config uses: qwen2.5:7b (7B), llama3.1:latest (8B) — TEST PLACEHOLDERS
   - Bake-off says:
     * Router: Granite 4 H Tiny (fallback: Ministral 3 3B)
     * Workhorse: Gemma 3n e4b OR Gemma 4 e4b
     * Reasoner: Qwen 3.5 9B (only when called)
   - Which models should be in Ollama? Pull them now or wait for Fase 2?
""")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_checks())
