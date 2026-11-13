"""Luu va tim kiem vector bang Chroma (lưu bền vững trên đĩa).

Ly do thay PostgreSQL + pgvector: may lam bai khong co Docker nen khong
dung duoc docker-compose của starter. GUIDE.md cho phep chon vector store
tương đương; Chroma đáp ứng đủ ba yêu cầu: lưu bền vững trên đĩa, vector
similarity search với metric cosine, và metadata filtering (where).
Ghi rõ trong báo cáo và README cùng cách xác minh lại.

Mỗi chiến lược nằm trong một collection riêng (day6_A/B/C) để nạp lại
độc lập; ID chunk được sinh ổn định nên upsert lại không nhân đôi bản ghi.
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from .models import Chunk

# cac truong metadata duoc phep dung lam filter trong search
FILTERABLE_FIELDS = {"period"}


class ChromaStore:
    def __init__(self, path: str, dimensions: int) -> None:
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Chroma tu tao collection khi can; giu ham de dung chung hop dong
        voi ban pgvector (khoi tao bang o day)."""

    def _collection(self, strategy: str):
        return self.client.get_or_create_collection(
            name=f"day6_{strategy}",
            metadata={"hnsw:space": "cosine"},  # metric cosine cho ca 3 cai strategy
        )

    def reset(self, strategy: str) -> None:
        """Xoa collection de nap lai tu dau khi nguon thay doi.

        Cach xu ly van ban thay doi: corpus la snapshot, khi doi thi xoa
        sach roi nap lai thay vi tim chunk cu bi loai bo tung cai.
        """
        try:
            self.client.delete_collection(f"day6_{strategy}")
        except Exception:
            pass  # chua ton tai
        self._collection(strategy)

    def upsert(self, strategy: str, chunks: list[Chunk],
               vectors: list[list[float]]) -> None:
        """Ghi chunk + vector voi ID on dinh; kiem tra do dai truoc khi ghi."""
        if len(chunks) != len(vectors):
            raise RuntimeError(
                f"chunks ({len(chunks)}) khong khop vectors ({len(vectors)})")
        for vec in vectors:
            if len(vec) != self.dimensions:
                raise RuntimeError(
                    f"vector {len(vec)} chieu khong khop model {self.dimensions} chieu")
        collection = self._collection(strategy)
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{
                "document_id": c.document_id,
                "source": c.source,
                "page": c.page,
                "title": c.metadata.get("title", ""),
                "period": c.metadata.get("period", ""),
            } for c in chunks],
        )

    def count(self, strategy: str) -> int:
        return self._collection(strategy).count()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Tim top_k chunk gan nhat theo cosine distance.

        filters chi nhan cac truong trong FILTERABLE_FIELDS, gia tri do
        nguoi dung cung cap truoc khi truy xuat; tham so truyen vao where
        cua Chroma nen khong co noi chuoi SQL.
        """
        where = None
        if filters:
            field = filters.get("field", "")
            if field not in FILTERABLE_FIELDS:
                raise RuntimeError(f"Truong filter '{field}' khong duoc phep")
            where = {field: {"$eq": filters.get("value")}}

        collection = self._collection(strategy)
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        for i in range(len(result["ids"][0])):
            meta = result["metadatas"][0][i]
            distance = result["distances"][0][i]
            hits.append({
                "chunk_id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "score": round(1.0 - distance, 6),  # cosine similarity
                "document_id": meta.get("document_id"),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "title": meta.get("title"),
                "period": meta.get("period"),
            })
        return hits
