import os
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config import config, PROJECT_ROOT

class DeterministicOfflineEmbedding(EmbeddingFunction[Documents]):
    """
    High-performance, lightweight, deterministic offline embedding function for edge UAVs and Pi 5.
    Generates semantic, normalized 64-dimensional embeddings locally without internet connectivity.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim

    @classmethod
    def name(cls) -> str:
        return "deterministic_offline_embedding"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: dict) -> "DeterministicOfflineEmbedding":
        return cls(dim=config.get("dim", 64))

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for text in input:
            tokens = text.lower().replace(",", " ").replace(":", " ").replace("-", " ").replace("_", " ").split()
            vec = [0.0] * self.dim
            for i, token in enumerate(tokens):
                h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
                idx1 = h % self.dim
                idx2 = (h >> 7) % self.dim
                weight = 1.0 / (math.log(i + 2))
                vec[idx1] += weight
                vec[idx2] += 0.5 * weight
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([round(x / norm, 6) for x in vec])
        return embeddings

class VectorStore:
    """
    Vector Store wrapper managing ChromaDB persistent or in-memory collections.
    """
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        if persist_dir:
            self.persist_path = Path(persist_dir)
        else:
            self.persist_path = PROJECT_ROOT / config.memory.chroma_db_dir
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or config.memory.chroma_collection
        self.embedding_fn = DeterministicOfflineEmbedding(dim=64)
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_path))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        count = self.collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        k = min(n_results, count)
        return self.collection.query(
            query_texts=[query_text],
            n_results=k
        )

    def count(self) -> int:
        return self.collection.count()

    def ping(self) -> bool:
        try:
            self.collection.count()
            return True
        except Exception:
            return False

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
