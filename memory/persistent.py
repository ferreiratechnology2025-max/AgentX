"""Persistent memory with SQLite + FAISS"""

import asyncio
import sqlite3
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from pathlib import Path

import faiss

from .cache import CacheLRU


@dataclass
class MemoriaItem:
    """Item de memória individual"""
    id: int
    texto: str
    embedding: np.ndarray
    timestamp: float
    importancia: float
    acessos: int
    sumario: Optional[str] = None


class MemoriaPersistente:
    """Sistema de memória com SQLite para texto e FAISS para busca semântica"""
    
    def __init__(
        self,
        db_path: str = "data/memoria.db",
        index_path: str = "data/memoria.index",
        embedding_dim: int = 384,
        max_memories: int = 1000
    ):
        self.db_path = db_path
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        
        # Cache quente
        self.cache = CacheLRU(max_size=200, ttl_seconds=1800)
        
        # Embedder (será inicializado sob demanda)
        self._embedder = None
        
        # Inicializa bancos
        self._init_sqlite()
        self._init_faiss()
    
    def _get_embedder(self):
        """Lazy loading do modelo de embedding"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            # Força CPU para liberar GPU para o LLM
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self._embedder.to('cpu')
        return self._embedder
    
    def _init_sqlite(self):
        """Inicializa banco SQLite com índices otimizados"""
        # Garante diretório existe
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memorias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT NOT NULL,
                embedding_blob BLOB,
                timestamp REAL NOT NULL,
                importancia REAL DEFAULT 1.0,
                acessos INTEGER DEFAULT 0,
                sumario TEXT,
                hash TEXT UNIQUE
            )
        """)
        
        # Índices para busca rápida
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memorias(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_importancia ON memorias(importancia DESC)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON memorias(hash)")
    
    def _init_faiss(self):
        """Inicializa índice FAISS (CPU para economizar VRAM)"""
        if Path(self.index_path).exists():
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"✅ Índice FAISS carregado: {self.index.ntotal} vetores")
                return
            except Exception as e:
                print(f"⚠️ Erro ao carregar índice: {e}")
        
        # Cria índice IVF para busca eficiente
        nlist = min(50, self.max_memories // 10)  # Número de clusters
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, max(1, nlist), faiss.METRIC_L2)
        self.index.nprobe = 5  # Número de clusters a verificar
        self.trained = False
    
    def _compute_hash(self, texto: str) -> str:
        """Gera hash único para o texto"""
        return hashlib.md5(texto.encode('utf-8')).hexdigest()
    
    async def salvar(
        self,
        texto: str,
        importancia: float = 1.0,
        sumario: Optional[str] = None
    ) -> int:
        """
        Salva uma memória de forma assíncrona
        
        Args:
            texto: Conteúdo da memória
            importancia: Score de importância (0-2)
            sumario: Versão resumida (opcional)
        
        Returns:
            ID da memória salva
        """
        # Verifica duplicata por hash
        text_hash = self._compute_hash(texto)
        cursor = self.conn.execute(
            "SELECT id FROM memorias WHERE hash = ?",
            (text_hash,)
        )
        existing = cursor.fetchone()
        if existing:
            # Atualiza importância e acessos
            self.conn.execute(
                "UPDATE memorias SET importancia = ?, acessos = acessos + 1 WHERE id = ?",
                (importancia, existing[0])
            )
            self.conn.commit()
            return existing[0]
        
        # Verifica limite e faz pruning
        cursor = self.conn.execute("SELECT COUNT(*) FROM memorias")
        count = cursor.fetchone()[0]
        
        if count >= self.max_memories:
            await self._prune_old_memories()
        
        # Gera embedding em thread separada (CPU)
        def generate_embedding():
            embedder = self._get_embedder()
            return embedder.encode([texto], show_progress_bar=False, convert_to_numpy=True)[0]
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, generate_embedding)
        
        # Salva no SQLite
        cursor = self.conn.execute("""
            INSERT INTO memorias (texto, embedding_blob, timestamp, importancia, sumario, hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            texto,
            pickle.dumps(embedding),
            datetime.now().timestamp(),
            importancia,
            sumario,
            text_hash
        ))
        
        memory_id = cursor.lastrowid
        self.conn.commit()
        
        # Atualiza índice FAISS
        embedding_array = np.array([embedding]).astype('float32')
        if not self.trained:
            if self.index.ntotal >= 39:  # Mínimo para treinar IVF
                self.index.train(embedding_array)
                self.trained = True
            else:
                # Treina com dados dummy para poder adicionar
                dummy = np.random.rand(50, self.embedding_dim).astype('float32')
                self.index.train(dummy)
                self.trained = True
        
        self.index.add(embedding_array)
        faiss.write_index(self.index, self.index_path)
        
        return memory_id
    
    async def _prune_old_memories(self, n: int = 20):
        """Remove as memórias menos importantes"""
        # Calcula score combinado: importância * log(acessos + 1)
        cursor = self.conn.execute("""
            SELECT id FROM memorias
            ORDER BY importancia * (acessos + 1) ASC
            LIMIT ?
        """, (n,))
        
        ids_to_remove = [row[0] for row in cursor.fetchall()]
        
        if ids_to_remove:
            placeholders = ','.join('?' * len(ids_to_remove))
            self.conn.execute(f"DELETE FROM memorias WHERE id IN ({placeholders})", ids_to_remove)
            self.conn.commit()
            
            # Recria índice FAISS
            self._rebuild_faiss_index()
    
    def _rebuild_faiss_index(self):
        """Reconstrói o índice FAISS a partir do SQLite"""
        cursor = self.conn.execute("SELECT embedding_blob FROM memorias")
        rows = cursor.fetchall()
        
        if not rows:
            self._init_faiss()
            return
        
        embeddings = []
        for row in rows:
            emb = pickle.loads(row[0])
            embeddings.append(emb)
        
        vectors = np.array(embeddings).astype('float32')
        
        # Recria índice
        nlist = min(50, len(vectors) // 10)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, max(1, nlist), faiss.METRIC_L2)
        self.index.nprobe = 5
        
        if len(vectors) >= 39:
            self.index.train(vectors)
            self.trained = True
        
        self.index.add(vectors)
        faiss.write_index(self.index, self.index_path)
    
    async def buscar(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Busca memórias semanticamente similares
        
        Args:
            query: Texto da consulta
            k: Número de resultados
            min_score: Score mínimo de similaridade
        
        Returns:
            Lista de memórias com scores
        """
        # Verifica cache
        cache_key = f"{query}_{k}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Gera embedding da query
        def generate_query_embedding():
            embedder = self._get_embedder()
            return embedder.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(None, generate_query_embedding)
        
        # Busca no FAISS
        k_search = min(k * 3, self.index.ntotal) if self.index.ntotal > 0 else 0
        
        if k_search == 0:
            return []
        
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            k_search
        )
        
        # Recupera textos do SQLite
        resultados = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            
            # FAISS usa 0-index, SQLite usa 1-index
            cursor = self.conn.execute(
                "SELECT id, texto, importancia, acessos FROM memorias WHERE rowid = ?",
                (idx + 1,)
            )
            row = cursor.fetchone()
            if row:
                # Atualiza contador de acessos
                self.conn.execute(
                    "UPDATE memorias SET acessos = acessos + 1 WHERE id = ?",
                    (row[0],)
                )
                
                # Score de similaridade (distância L2 -> similaridade)
                score = 1.0 / (1.0 + dist)
                
                if score >= min_score:
                    resultados.append({
                        'id': row[0],
                        'texto': row[1],
                        'importancia': row[2],
                        'acessos': row[3],
                        'score': float(score),
                        'distancia': float(dist)
                    })
        
        # Ordena por score
        resultados.sort(key=lambda x: x['score'], reverse=True)
        resultados = resultados[:k]
        
        # Salva no cache
        self.cache.set(cache_key, resultados)
        
        return resultados
    
    async def buscar_por_id(self, memory_id: int) -> Optional[Dict]:
        """Busca memória por ID"""
        cursor = self.conn.execute(
            "SELECT id, texto, importancia, acessos, timestamp FROM memorias WHERE id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'texto': row[1],
                'importancia': row[2],
                'acessos': row[3],
                'timestamp': row[4]
            }
        return None
    
    async def deletar(self, memory_id: int) -> bool:
        """Remove uma memória pelo ID"""
        cursor = self.conn.execute("DELETE FROM memorias WHERE id = ?", (memory_id,))
        self.conn.commit()
        
        if cursor.rowcount > 0:
            self._rebuild_faiss_index()
            return True
        return False
    
    async def listar_recentes(self, limit: int = 50) -> List[Dict]:
        """Lista memórias mais recentes"""
        cursor = self.conn.execute("""
            SELECT id, texto, importancia, timestamp
            FROM memorias
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        return [
            {'id': row[0], 'texto': row[1], 'importancia': row[2], 'timestamp': row[3]}
            for row in cursor.fetchall()
        ]
    
    def __del__(self):
        """Fecha conexões"""
        if hasattr(self, 'conn'):
            self.conn.close()
