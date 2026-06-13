from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from embedder import MetaMathEmbedder


COLLECTION_NAME = 'meta_math'

class QDrantVectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host="localhost",
            port=6333,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )
            print('Vector store Collection {} created'.format(COLLECTION_NAME))
        else:
            count = self.client.count(COLLECTION_NAME).count
            print(f"[VectorStore] Collection '{COLLECTION_NAME}' loaded ({count:,} points).")

    def is_empty(self):
        return self.client.count(COLLECTION_NAME).count == 0

    def upsert_chunks(self, ids, vectors, payloads):
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        batch = 256
        for start in range(0, len(points), batch):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[start:start + batch],
            )

    def search(self, query_vector, top_k):
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points
        return [
            {'score': h.score, **h.payload}
            for h in hits if h.score >= 0.45
        ]
