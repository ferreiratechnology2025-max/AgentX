"""LRU Cache for embeddings"""

import threading
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Any
import numpy as np


class CacheLRU:
    """Cache LRU com TTL para embeddings"""
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retorna valor do cache se existir e não expirou"""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Verifica TTL
            if datetime.now().timestamp() - self._timestamps[key] > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            # Move para o fim (mais recente)
            self._cache.move_to_end(key)
            return self._cache[key]
    
    def set(self, key: str, value: np.ndarray) -> None:
        """Armazena valor no cache"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove o item mais antigo
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                del self._timestamps[oldest]
            
            self._cache[key] = value
            self._timestamps[key] = datetime.now().timestamp()
            self._cache.move_to_end(key)
    
    def clear(self) -> None:
        """Limpa todo o cache"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
