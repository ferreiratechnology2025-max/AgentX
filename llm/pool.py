"""LLM Pool Manager - Roteamento de modelos Ollama"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path

import yaml
import requests

from llm.manager import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuracao de um modelo (Ollama)"""
    id: str
    role_preference: List[str]
    num_ctx: Optional[int] = None


@dataclass
class ConnectedModel:
    """Modelo conectado com metadados de uso"""
    llm: LLMManager
    last_used: float


class LLMPoolManager:
    """
    Pool de modelos Ollama com roteamento por role e gestao de VRAM.

    Uso:
        pool = LLMPoolManager("config.yaml")
        llm = await pool.get_model(role="coding")
        response = await llm.generate(...)
    """

    VRAM_WARNING_PERCENT = 85.0
    ROUTER_MODEL_ID = "granite4htiny"

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, ConnectedModel] = {}
        self._lock = asyncio.Lock()
        self.app_config: Dict = {}
        self.base_url: str = "http://localhost:11434"
        self.auto_unload: bool = True
        self.auto_unload_timeout: int = 300

        self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.app_config = yaml.safe_load(f)

        self.base_url = self.app_config.get('ollama', {}).get('base_url', 'http://localhost:11434')

        pool_cfg = self.app_config.get('llm_pool', {})
        self.auto_unload = pool_cfg.get('auto_unload', True)
        self.auto_unload_timeout = pool_cfg.get('auto_unload_timeout', 300)

        llm_config = self.app_config.get('llm_pool', {})
        for model_cfg in llm_config.get('models', []):
            model = ModelConfig(
                id=model_cfg['id'],
                role_preference=model_cfg.get('role_preference', ['general']),
                num_ctx=model_cfg.get('num_ctx'),
            )
            self.models[model.id] = model

        logger.info(f"LLM Pool Ollama: {len(self.models)} modelos configurados")

    def _get_vram_usage(self) -> dict:
        """Retorna uso de VRAM via nvidia-smi (sem dependencia de torch)."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    used = float(parts[0].strip())
                    total = float(parts[1].strip())
                    return {
                        'used_gb': round(used / 1024, 2),
                        'total_gb': round(total / 1024, 2),
                        'free_gb': round((total - used) / 1024, 2),
                        'usage_percent': round((used / total) * 100, 1) if total > 0 else 0.0,
                    }
        except FileNotFoundError:
            logger.debug("nvidia-smi not found (non-GPU environment)")
        except Exception as e:
            logger.warning(f"VRAM check failed: {e}")
        return {'used_gb': 0, 'total_gb': 0, 'free_gb': 0, 'usage_percent': 0}

    def _unload_least_recently_used(self) -> bool:
        """Descarrega modelo LRU (exceto Router). Retorna True se descarregou."""
        candidates = [
            (mid, info.last_used)
            for mid, info in self.loaded_models.items()
            if mid != self.ROUTER_MODEL_ID
        ]
        if not candidates:
            logger.warning("VRAM full and no unloadable models (Router only)")
            return False

        candidates.sort(key=lambda x: x[1])
        victim_id = candidates[0][0]
        logger.info(f"Unloading LRU model '{victim_id}' to free VRAM")
        self._unload_model(victim_id)
        return True

    def _unload_model(self, model_id: str):
        """Descarrega modelo da VRAM via Ollama API (keep_alive=0)."""
        if model_id not in self.loaded_models:
            return
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_id, "prompt": "", "keep_alive": 0},
                timeout=10
            )
            if resp.status_code == 200:
                del self.loaded_models[model_id]
                logger.info(f"Model '{model_id}' unloaded")
            else:
                logger.warning(f"Ollama unload returned {resp.status_code}: {resp.text[:200]}")
                del self.loaded_models[model_id]
        except Exception as e:
            logger.error(f"Failed to unload model '{model_id}': {e}")
            del self.loaded_models[model_id]

    def _route_by_role(self, role: str) -> Optional[str]:
        """Roteia role para melhor modelo disponivel."""
        for model_id, model in self.models.items():
            if role in model.role_preference:
                return model_id
        for model_id, model in self.models.items():
            if 'general' in model.role_preference:
                return model_id
        return None

    async def get_model(
        self,
        model_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> LLMManager:
        """
        Obtem modelo LLMManager (conecta com Ollama).

        Args:
            model_id: ID especifico do modelo
            role: Papel do agente (roteamento automatico)

        Returns:
            LLMManager instanciado (conectado a Ollama endpoint)
        """
        async with self._lock:
            target_id = model_id
            if target_id is None and role:
                target_id = self._route_by_role(role)
            if target_id is None:
                raise ValueError(f"Nao foi possivel determinar modelo para role '{role}'")
            if target_id not in self.models:
                raise ValueError(f"Modelo '{target_id}' nao configurado")

            # Se ja carregado, atualiza last_used e retorna
            if target_id in self.loaded_models:
                self.loaded_models[target_id].last_used = time.time()
                logger.debug(f"Modelo '{target_id}' ja conectado")
                return self.loaded_models[target_id].llm

            # VRAM check: descarrega LRU se > 85%
            vram = self._get_vram_usage()
            if vram['usage_percent'] > self.VRAM_WARNING_PERCENT:
                logger.warning(f"VRAM at {vram['usage_percent']}%, unloading LRU before loading '{target_id}'")
                if not self._unload_least_recently_used():
                    raise RuntimeError(
                        f"VRAM at {vram['usage_percent']}% and no models to unload. "
                        f"Cannot load '{target_id}'."
                    )

            return await self._connect_model(target_id)

    async def _connect_model(self, model_id: str) -> LLMManager:
        """Conecta com modelo via Ollama."""
        if model_id not in self.models:
            raise ValueError(f"Modelo '{model_id}' nao configurado")

        logger.info(f"Conectando modelo '{model_id}' (Ollama)...")

        llm_config = dict(self.app_config)
        llm_config['llm']['model_id'] = model_id
        llm_config['llm']['num_ctx'] = self.models[model_id].num_ctx

        def _connect():
            return LLMManager(llm_config)

        llm = await asyncio.to_thread(_connect)
        self.loaded_models[model_id] = ConnectedModel(llm=llm, last_used=time.time())
        logger.info(f"Modelo '{model_id}' conectado (Ollama endpoint)")
        return llm

    def get_status(self) -> dict:
        """Retorna status do pool incluindo VRAM."""
        vram = self._get_vram_usage()
        return {
            'configured_models': list(self.models.keys()),
            'connected_models': list(self.loaded_models.keys()),
            'vram': vram,
        }

    async def unload_all(self):
        """Descarrega todos os modelos."""
        for model_id in list(self.loaded_models.keys()):
            self._unload_model(model_id)


_pool_instance: Optional[LLMPoolManager] = None


def get_llm_pool() -> LLMPoolManager:
    """Obtem instancia singleton do pool."""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = LLMPoolManager()
    return _pool_instance
