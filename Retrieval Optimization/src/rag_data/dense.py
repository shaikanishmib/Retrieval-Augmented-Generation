"""Normalized dense retrieval with exact, FAISS, and local Qdrant indexes."""

import time
from pathlib import Path

import numpy as np


def normalize(vectors):
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite two-dimensional matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors / np.maximum(norms, 1e-12))


class Dense:
    def __init__(self, ids, vectors, backend="exact", storage=None):
        self.ids = list(ids)
        self.vectors = normalize(vectors)
        if len(set(self.ids)) != len(self.ids) or len(self.ids) != len(self.vectors) or not self.ids:
            raise ValueError("Unique IDs must match a nonempty embedding matrix")
        self.backend = backend
        start = time.perf_counter()
        if backend == "faiss":
            import faiss
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
        elif backend == "qdrant":
            from qdrant_client import QdrantClient, models
            self.index = QdrantClient(path=str(Path(storage))) if storage else QdrantClient(":memory:")
            if self.index.collection_exists("documents"):
                self.index.delete_collection("documents")
            self.index.create_collection("documents", vectors_config=models.VectorParams(size=self.vectors.shape[1], distance=models.Distance.COSINE))
            for offset in range(0, len(self.ids), 512):
                self.index.upsert("documents", points=[models.PointStruct(id=i, vector=self.vectors[i].tolist(), payload={"id": self.ids[i]}) for i in range(offset, min(offset + 512, len(self.ids)))])
        elif backend != "exact":
            raise ValueError(f"Unknown dense backend: {backend}")
        self.build_seconds = time.perf_counter() - start

    def search(self, query, k=20):
        if k < 1:
            raise ValueError("k must be positive")
        q = normalize(np.asarray(query).reshape(1, -1))[0]
        if len(q) != self.vectors.shape[1]:
            raise ValueError("Query and corpus embedding dimensions differ")
        k = min(k, len(self.ids))
        if self.backend == "faiss":
            scores, positions = self.index.search(q.reshape(1, -1), k)
            return sorted([(self.ids[i], float(s)) for i, s in zip(positions[0], scores[0])], key=lambda r: (-r[1], r[0]))
        if self.backend == "qdrant":
            points = self.index.query_points("documents", query=q.tolist(), limit=k).points
            return sorted([(p.payload["id"], p.score) for p in points], key=lambda r: (-r[1], r[0]))
        scores = self.vectors @ q
        positions = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in positions]

    def close(self):
        if self.backend == "qdrant":
            self.index.close()
