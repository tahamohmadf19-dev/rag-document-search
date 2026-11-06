"""Thin wrapper around Qdrant: a server when QDRANT_URL is set, otherwise local embedded mode."""
import os
from typing import Any, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStorage:
    def __init__(
        self,
        location: Optional[str] = None,
        collection: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "documents")
        if client is not None:
            self.client = client
        elif location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif location or os.getenv("QDRANT_URL"):
            self.client = QdrantClient(url=location or os.environ["QDRANT_URL"])
        else:
            self.client = QdrantClient(path=os.getenv("QDRANT_PATH", "./qdrant_data"))

    def _ensure_collection(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict[str, Any]]) -> int:
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors and payloads must have the same length.")
        if not ids:
            return 0
        self._ensure_collection(len(vectors[0]))
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(id=point_id, vector=list(vector), payload=payload)
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ],
        )
        return len(ids)

    def search(self, vector: Sequence[float], top_k: int = 5) -> dict[str, list]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self.client.collection_exists(self.collection):
            return {"contexts": [], "sources": [], "pages": [], "scores": []}
        hits = self.client.query_points(self.collection, query=list(vector), limit=top_k, with_payload=True).points
        return {
            "contexts": [hit.payload.get("text", "") for hit in hits],
            "sources": [hit.payload.get("source", "") for hit in hits],
            "pages": [hit.payload.get("page") for hit in hits],
            "scores": [float(hit.score) for hit in hits],
        }

    def close(self) -> None:
        self.client.close()
