"""Test dual model loading: Worker (Qwen Q3_K_M) + Judge (Phi-4-mini)"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import subprocess
import pytest
from llm.pool import get_llm_pool


def nvidia_vram_mb():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            shell=True).decode().strip()
        return int(out)
    except Exception:
        return 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="Legacy test for llama-cpp-python architecture; Ollama pool uses different API")
async def test_dual_load():
    pool = get_llm_pool()
    
    print(f"VRAM inicial (nvidia-smi): {nvidia_vram_mb()} MB")
    
    # 1. Carregar Worker
    print("\n1. Carregando Worker (Qwen 2.5 7B Q3_K_M)...")
    worker = await pool.get_model(role="coding")
    pool_vram = pool._get_current_vram_usage()
    nv_vram = nvidia_vram_mb()
    print(f"   Worker: {worker.model_path.split('/')[-1]}")
    print(f"   Pool estimate: {pool_vram:.1f}GB | nvidia-smi: {nv_vram}MB ({nv_vram/1024:.1f}GB)")
    
    # 2. Carregar Judge
    print("\n2. Carregando Judge (Phi-4-mini)...")
    judge = await pool.get_model(role="judging")
    pool_vram2 = pool._get_current_vram_usage()
    nv_vram2 = nvidia_vram_mb()
    print(f"   Judge: {judge.model_path.split('/')[-1]}")
    print(f"   Pool estimate: {pool_vram2:.1f}GB | nvidia-smi: {nv_vram2}MB ({nv_vram2/1024:.1f}GB)")
    
    # 3. Modelos diferentes
    assert worker.model_path != judge.model_path
    print(f"\n3. MODELOS DIFERENTES [OK]")
    
    # 4. VRAM check (nvidia-smi real, nao pool estimate)
    phys_max = 12288  # 12GB em MB
    print(f"4. VRAM real: {nv_vram2}MB / {phys_max}MB ({nv_vram2/phys_max*100:.0f}%)")
    assert nv_vram2 < phys_max * 0.98, f"VRAM real > 98%! {nv_vram2}/{phys_max}"
    print(f"   Folga: {phys_max - nv_vram2}MB [OK]")
    
    # 5. Inferencia em ambos
    print("\n5. Testando inferencia...")
    w_resp, w_usage = await worker.generate("Say 'hello from worker' in 2 words", max_tokens=8, temperature=0.1)
    print(f"   Worker: '{w_resp.strip()}'")
    
    j_resp, j_usage = await judge.generate("Say 'hello from judge' in 2 words", max_tokens=8, temperature=0.1)
    print(f"   Judge: '{j_resp.strip()}'")
    
    different = w_resp.strip() != j_resp.strip()
    print(f"\n6. Respostas diferentes: {different}")
    print("\n=== DUAL LOAD TEST PASSED ===")
    
    # Cleanup
    await pool.unload_all()
    print("Cleanup completo")


if __name__ == "__main__":
    asyncio.run(test_dual_load())
