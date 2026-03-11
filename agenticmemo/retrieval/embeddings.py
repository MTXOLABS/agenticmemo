"""Embedding backends for semantic similarity."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..exceptions import EmbeddingError


class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @abstractmethod
    async def encode(self, texts: list[str]) -> np.ndarray:
        """Return (N, D) float32 embeddings."""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def batch_cosine_similarity(
        self, query: np.ndarray, corpus: np.ndarray
    ) -> np.ndarray:
        """Vectorised cosine similarity: query (D,) vs corpus (N, D)."""
        query_norm = np.linalg.norm(query)
        corpus_norms = np.linalg.norm(corpus, axis=1)
        if query_norm == 0:
            return np.zeros(len(corpus))
        denom = corpus_norms * query_norm
        denom[denom == 0] = 1e-9
        return corpus.dot(query) / denom


class SentenceTransformerEmbeddings(EmbeddingBackend):
    """Local embeddings via sentence-transformers (no API key required).

    Default model: ``all-MiniLM-L6-v2`` — fast, 384-dim, good quality.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None   # lazy-loaded

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception:
                # Fallback: load from local cache only (no network check)
                self._model = SentenceTransformer(
                    self._model_name, local_files_only=True
                )
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from e

    async def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        try:
            vectors = self._model.encode(  # type: ignore[union-attr]
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}") from e


class OpenAIEmbeddings(EmbeddingBackend):
    """OpenAI text-embedding-3-small backend."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        self._model = model
        self._api_key = api_key
        self._client = None

    def _load(self) -> None:
        if self._client is not None:
            return
        try:
            import openai  # noqa: PLC0415
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        except ImportError as e:
            raise EmbeddingError(
                "openai not installed. Run: pip install agentmemento[openai]"
            ) from e

    async def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        try:
            resp = await self._client.embeddings.create(  # type: ignore[union-attr]
                model=self._model, input=texts
            )
            vecs = [d.embedding for d in resp.data]
            return np.array(vecs, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding error: {e}") from e
