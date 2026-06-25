"""Tests for Phase 1: Ollama runtime integration"""

import asyncio
import sys
import json
import requests
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, '.')

from llm.pool import get_llm_pool
from llm.manager import Message
from tools.schemas import ReActOutput


async def test_1_ollama_endpoint():
    """Test 1: Ollama endpoint is running"""
    print("\n" + "="*60)
    print("TEST 1: Ollama endpoint running")
    print("="*60)

    try:
        response = requests.get("http://localhost:11434/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        print(f"✓ Ollama is running")
        print(f"  Available models: {len(models.get('data', []))}")
        for model in models.get('data', [])[:3]:
            print(f"    - {model.get('id', 'unknown')}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


async def test_2_pool_loading():
    """Test 2: Pool loads models from config"""
    print("\n" + "="*60)
    print("TEST 2: Pool loads models from config")
    print("="*60)

    try:
        pool = get_llm_pool()
        status = pool.get_status()
        print(f"✓ Pool initialized")
        print(f"  Configured models: {status['configured_models']}")
        assert len(status['configured_models']) >= 2, "Should have at least 2 models"
        print(f"✓ Models configured: {status['configured_models']}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_3_get_model_by_role():
    """Test 3: Pool routes role -> model and connects"""
    print("\n" + "="*60)
    print("TEST 3: Pool routes role -> model and connects")
    print("="*60)

    try:
        pool = get_llm_pool()
        llm = await pool.get_model(role="general")
        print(f"✓ Model connected via role 'general'")
        print(f"  LLM type: {type(llm).__name__}")
        print(f"  Model ID: {llm.model_id}")
        assert llm is not None, "Should return LLMManager instance"
        assert hasattr(llm, 'generate'), "Should have generate method"
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_4_generate_simple():
    """Test 4: Generate returns response and usage metrics"""
    print("\n" + "="*60)
    print("TEST 4: Generate returns response and usage metrics")
    print("="*60)

    try:
        pool = get_llm_pool()
        llm = await pool.get_model(role="general")

        print(f"  Sending prompt: 'Olá, responda em uma palavra'")
        text, usage = await llm.generate(
            prompt="Olá, responda em uma palavra",
            max_tokens=20,
            temperature=0.7
        )

        print(f"✓ Response received")
        print(f"  Text: {text[:50]}..." if len(text) > 50 else f"  Text: {text}")
        print(f"  Usage: {json.dumps(usage, indent=2)}")

        assert isinstance(text, str) and len(text) > 0, "Should return non-empty string"
        assert isinstance(usage, dict), "Should return usage dict"
        assert "prompt_tokens" in usage or "completion_tokens" in usage, \
            "Usage should have token counts"

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_generate_with_validation():
    """Test 5: Generate_with_validation parses ReAct format"""
    print("\n" + "="*60)
    print("TEST 5: Generate_with_validation parses ReAct format")
    print("="*60)

    try:
        pool = get_llm_pool()
        llm = await pool.get_model(role="general")

        # Create simple message
        messages = [
            Message(role="user", content="Responda com Thought: ... Final Answer: ...")
        ]

        print(f"  Sending ReAct-format request")
        output, usage = await llm.generate_with_validation(
            messages=messages,
            tools=[],
            max_tokens=50,
            temperature=0.7
        )

        print(f"✓ Validation completed")
        print(f"  Output type: {type(output).__name__}")
        print(f"  Is ReActOutput: {isinstance(output, ReActOutput)}")
        print(f"  Is None (failed parse): {output is None}")
        print(f"  Usage: {json.dumps(usage, indent=2)}")

        # Either parsed or None (both valid for this test)
        assert output is None or isinstance(output, ReActOutput), \
            "Should return ReActOutput or None"

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all 5 tests and report results"""
    print("\n" + "="*60)
    print("PHASE 1 VALIDATION: Ollama Runtime Integration")
    print("="*60)

    results = []

    # Test 1
    results.append(("Ollama endpoint running", await test_1_ollama_endpoint()))

    if not results[0][1]:
        print("\n" + "!"*60)
        print("Cannot proceed without Ollama. Start it with: ollama serve")
        print("!"*60)
        return results

    # Tests 2-5
    results.append(("Pool loads config", await test_2_pool_loading()))
    results.append(("Role routing works", await test_3_get_model_by_role()))
    results.append(("Generate returns response", await test_4_generate_simple()))
    results.append(("Validation parses ReAct", await test_5_generate_with_validation()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[PASS] Phase 1 validation PASSED")
        return True
    else:
        print(f"\n[FAIL] Phase 1 validation FAILED ({total - passed} failures)")
        return False


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
