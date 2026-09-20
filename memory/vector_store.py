import logging
import math
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from config import PROJECT_ROOT, config

logger = logging.getLogger("aerocortex.vector_store")

try:
    from pinecone import Pinecone, ServerlessSpec

    PINECONE_AVAILABLE = True
except ImportError:
    Pinecone = None  # type: ignore[misc, assignment]
    ServerlessSpec = None  # type: ignore[misc, assignment]
    PINECONE_AVAILABLE = False

# chromadb is optional — VECTOR_FALLBACK=none never imports it (Lambda)
chromadb = None  # type: ignore
Documents = Any
EmbeddingFunction = object
Embeddings = Any


def _chroma_allowed() -> bool:
    return (os.getenv("VECTOR_FALLBACK") or "chroma").strip().lower() != "none"


def _import_chromadb():
    global chromadb, Documents, EmbeddingFunction, Embeddings
    if chromadb is not None:
        return True
    if not _chroma_allowed():
        return False
    try:
        import chromadb as _ch
        from chromadb.api.types import Documents as _D, EmbeddingFunction as _EF, Embeddings as _E

        chromadb = _ch
        Documents = _D
        EmbeddingFunction = _EF
        Embeddings = _E
        return True
    except ImportError:
        logger.warning("chromadb not installed; vector fallback unavailable")
        return False


def empty_query_result() -> Dict[str, Any]:
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


class DeterministicOfflineEmbedding:
    """
    High-performance, lightweight, deterministic offline embedding function for edge UAVs and Pi 5.
    Generates semantic, normalized 64-dimensional embeddings locally without internet connectivity.
    Compatible with chromadb EmbeddingFunction protocol when chromadb is installed.
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

    def _embed_texts(self, texts: Any) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
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

    def __call__(self, input: Any) -> Any:
        return self._embed_texts(input)

    def embed_query(self, input: Any) -> Any:
        return self._embed_texts(input)

    def embed_documents(self, input: Any) -> Any:
        return self._embed_texts(input)

    def is_legacy(self) -> bool:
        return True


def _sanitize_metadata(metadatas: List[Dict[str, Any]], documents: List[str]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for meta, doc in zip(metadatas, documents):
        item: Dict[str, Any] = {}
        source = dict(meta or {})
        source["_document"] = doc
        for key, value in source.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                item[str(key)] = value
            else:
                item[str(key)] = str(value)
        cleaned.append(item)
    return cleaned


@runtime_checkable
class VectorBackend(Protocol):
    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        ...

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        ...

    def count(self) -> int:
        ...

    def ping(self) -> bool:
        ...

    def reset(self) -> None:
        ...


class NullVectorBackend:
    """No-op vector backend when Chroma is disabled and Pinecone is unavailable."""

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        return None

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        return empty_query_result()

    def count(self) -> int:
        return 0

    def ping(self) -> bool:
        return False

    def reset(self) -> None:
        return None


class ChromaBackend:
    """Local Chroma persistent / ephemeral collection. Raspberry Pi fallback."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_fn: Optional[DeterministicOfflineEmbedding] = None,
    ):
        if persist_dir:
            self.persist_path = Path(persist_dir)
        else:
            self.persist_path = PROJECT_ROOT / config.memory.chroma_db_dir

        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or config.memory.chroma_collection
        self.embedding_fn = embedding_fn or DeterministicOfflineEmbedding(dim=64)

        if not _import_chromadb():
            raise RuntimeError("chromadb is not available (VECTOR_FALLBACK=none or not installed)")

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_path))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        count = self.collection.count()
        if count == 0:
            return empty_query_result()
        k = min(n_results, count)
        return self.collection.query(query_texts=[query_text], n_results=k)

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
            metadata={"hnsw:space": "cosine"},
        )


class PineconeBackend:
    """Pinecone serverless index using the same 64-dim offline embeddings as Chroma."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        embedding_fn: Optional[DeterministicOfflineEmbedding] = None,
        dimension: Optional[int] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
    ):
        if not PINECONE_AVAILABLE:
            raise RuntimeError("pinecone package is not installed")
        key = api_key or config.memory.pinecone_api_key
        if not key:
            raise RuntimeError("PINECONE_API_KEY is not set")

        self.embedding_fn = embedding_fn or DeterministicOfflineEmbedding(dim=64)
        self.index_name = index_name or config.memory.pinecone_index
        self.dimension = dimension or config.memory.pinecone_dimension
        self.cloud = cloud or config.memory.pinecone_cloud
        self.region = region or config.memory.pinecone_region
        self.host = host if host is not None else config.memory.pinecone_host
        self.pc = Pinecone(api_key=key)
        self.index = self._connect_index()

    def _index_names(self) -> List[str]:
        listing = self.pc.list_indexes()
        if hasattr(listing, "names"):
            return list(listing.names())
        names = []
        for item in listing:
            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")
            if name:
                names.append(name)
        return names

    def _connect_index(self):
        if self.host:
            return self.pc.Index(host=self.host)
        names = self._index_names()
        if self.index_name not in names:
            logger.info("Creating Pinecone index %s (dim=%s)", self.index_name, self.dimension)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        return self.pc.Index(self.index_name)

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        vectors = self.embedding_fn(documents)
        metas = _sanitize_metadata(metadatas, documents)
        payload = [
            {"id": str(doc_id), "values": values, "metadata": meta}
            for doc_id, values, meta in zip(ids, vectors, metas)
        ]
        self.index.upsert(vectors=payload)

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        count = self.count()
        if count == 0:
            return empty_query_result()
        k = min(n_results, count)
        vector = self.embedding_fn([query_text])[0]
        result = self.index.query(vector=vector, top_k=k, include_metadata=True)
        matches = getattr(result, "matches", None)
        if matches is None and isinstance(result, dict):
            matches = result.get("matches", [])
        matches = matches or []

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        distances: List[float] = []
        for match in matches:
            if isinstance(match, dict):
                match_id = str(match.get("id", ""))
                score = float(match.get("score", 0.0))
                meta = dict(match.get("metadata") or {})
            else:
                match_id = str(getattr(match, "id", ""))
                score = float(getattr(match, "score", 0.0) or 0.0)
                meta = dict(getattr(match, "metadata", None) or {})
            document = str(meta.pop("_document", "") or "")
            ids.append(match_id)
            documents.append(document)
            metadatas.append(meta)
            distances.append(round(max(0.0, min(2.0, 1.0 - score)), 6))
        return {"ids": [ids], "documents": [documents], "metadatas": [metadatas], "distances": [distances]}

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        if hasattr(stats, "total_vector_count"):
            return int(stats.total_vector_count or 0)
        if isinstance(stats, dict):
            return int(stats.get("total_vector_count") or 0)
        return 0

    def ping(self) -> bool:
        try:
            self.index.describe_index_stats()
            return True
        except Exception as exc:
            logger.warning("Pinecone ping failed: %s", exc)
            return False

    def reset(self) -> None:
        try:
            self.index.delete(delete_all=True)
        except Exception as exc:
            logger.warning("Pinecone reset failed: %s", exc)


class VectorStore:
    """
    Vector facade: Pinecone for the internet-connected REST API, Chroma when
    Pinecone is unset or unreachable (Raspberry Pi / offline).
    Public methods stay add_documents / query / count / ping / reset.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        pinecone_backend: Optional[VectorBackend] = None,
    ):
        self.embedding_fn = DeterministicOfflineEmbedding(dim=64)
        self.collection_name = collection_name or config.memory.chroma_collection
        self._chroma_disabled = not _chroma_allowed()
        self._chroma: VectorBackend
        if self._chroma_disabled:
            self._chroma = NullVectorBackend()
        else:
            try:
                self._chroma = ChromaBackend(
                    persist_dir=persist_dir,
                    collection_name=self.collection_name,
                    embedding_fn=self.embedding_fn,
                )
            except Exception as exc:
                logger.warning("Chroma unavailable, using null vector backend: %s", exc)
                self._chroma = NullVectorBackend()
                self._chroma_disabled = True
        self._pinecone: Optional[VectorBackend] = pinecone_backend
        self.engine = "none" if self._chroma_disabled else "chroma"
        self._active: VectorBackend = self._chroma

        should_try_cloud = pinecone_backend is not None or (
            persist_dir is None
            and (collection_name is None or collection_name == config.memory.chroma_collection)
            and self._cloud_allowed()
        )
        if self._pinecone is None and should_try_cloud:
            self._pinecone = self._build_pinecone()
        if should_try_cloud:
            self.reconnect()

    def _cloud_allowed(self) -> bool:
        return bool(config.memory.pinecone_enabled and config.memory.pinecone_api_key)

    def _build_pinecone(self) -> Optional[VectorBackend]:
        try:
            backend = PineconeBackend(embedding_fn=self.embedding_fn)
            logger.info("Pinecone index connected: %s", config.memory.pinecone_index)
            return backend
        except Exception as exc:
            logger.warning("Pinecone init failed, using Chroma: %s", exc)
            return None

    def reconnect(self) -> str:
        if self._pinecone is not None:
            try:
                if self._pinecone.ping():
                    self._active = self._pinecone
                    self.engine = "pinecone"
                    return self.engine
            except Exception as exc:
                logger.warning("Pinecone reconnect failed: %s", exc)
        self._active = self._chroma
        self.engine = "none" if self._chroma_disabled else "chroma"
        return self.engine

    def _failover(self, exc: Exception) -> None:
        if self.engine == "pinecone":
            logger.warning("Pinecone operation failed, falling back to local vector backend: %s", exc)
            self._active = self._chroma
            self.engine = "none" if self._chroma_disabled else "chroma"

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        try:
            self._active.add_documents(ids, documents, metadatas)
        except Exception as exc:
            self._failover(exc)
            if self.engine in ("chroma", "none"):
                self._chroma.add_documents(ids, documents, metadatas)
            else:
                raise

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        try:
            return self._active.query(query_text, n_results=n_results)
        except Exception as exc:
            self._failover(exc)
            return self._chroma.query(query_text, n_results=n_results)

    def count(self) -> int:
        try:
            return self._active.count()
        except Exception as exc:
            self._failover(exc)
            return self._chroma.count()

    def ping(self) -> bool:
        self.reconnect()
        try:
            return bool(self._active.ping())
        except Exception:
            return False

    def chroma_ping(self) -> bool:
        if self._chroma_disabled:
            return False
        try:
            return bool(self._chroma.ping())
        except Exception:
            return False

    def pinecone_ping(self) -> bool:
        if self._pinecone is None:
            return False
        try:
            return bool(self._pinecone.ping())
        except Exception:
            return False

    def health(self) -> Dict[str, Any]:
        chroma_ok = self.chroma_ping()
        pine_configured = self._pinecone is not None or self._cloud_allowed()
        pine_ok = self.pinecone_ping()
        if pine_ok and self._pinecone is not None:
            self._active = self._pinecone
            self.engine = "pinecone"
            pine_status = "ok"
        else:
            self._active = self._chroma
            self.engine = "none" if self._chroma_disabled else "chroma"
            if not pine_configured:
                pine_status = "unset"
            elif chroma_ok:
                pine_status = "fallback_chroma"
            else:
                pine_status = "down"
        if self._chroma_disabled:
            chroma_status = "disabled"
        else:
            chroma_status = "ok" if chroma_ok else "down"
        return {
            "engine": self.engine,
            "pinecone": pine_status,
            "chroma": chroma_status,
            "ok": pine_ok or chroma_ok,
        }

    def reset(self) -> None:
        try:
            self._active.reset()
        except Exception as exc:
            logger.warning("Vector reset failed: %s", exc)
            self._chroma.reset()
            self._active = self._chroma
            self.engine = "none" if self._chroma_disabled else "chroma"
