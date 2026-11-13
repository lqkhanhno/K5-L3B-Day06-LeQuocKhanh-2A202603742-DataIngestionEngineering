"""Embedding that qua API OpenAI. Khong co phuong an fallback mock.

Khi API loi (key sai, het quota, mat mang), ham se throw loi voi thong
bao ro rang thay vi tao vector gia. Gia tham chieu dung de tinh chi phi:
text-embedding-3-small = 0,02 USD / 1 trieu token (ba giai doan 2026).
"""
from __future__ import annotations

import os
import time
from typing import Protocol

import requests

API_URL = "https://api.openai.com/v1/embeddings"
# Gia USD cho 1 trieu token, chua thay doi tu khi model ra mat.
PRICE_PER_MILLION_TOKENS = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Tra ve mot vector cho moi input, giu dung thu tu."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Tra ve vector cau hoi tuong thích voi vector van ban da luu."""
        ...


class OpenAIEmbedder:
    """Dung chung mot model cho ca document va query (neu model do yeu cau
    them prefix cho query thi ghi ro tai day; text-embedding-3 khong can)."""

    def __init__(self, model_name: str = "text-embedding-3-small",
                 api_key: str | None = None, batch_size: int = 128) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY", "").strip().strip('"')
        if not key:
            raise RuntimeError(
                "Thiếu OPENAI_API_KEY trong starter/.env - khong the goi API that.")
        if model_name not in DIMENSIONS:
            raise RuntimeError(
                f"Model '{model_name}' chua duoc khai bao so chieu/gia trong embeddings.py")
        self.model_name = model_name
        self.dimensions = DIMENSIONS[model_name]
        self.batch_size = batch_size
        self.api_key = key
        self.doc_tokens = 0      # token dung cho vector van ban
        self.query_tokens = 0    # token dung cho vector cau hoi
        self.doc_embed_seconds = 0.0

    def _call(self, texts: list[str], kind: str = "doc") -> list[list[float]]:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI tra loi {response.status_code}: {response.text[:300]}")
        data = response.json()
        tokens = data.get("usage", {}).get("prompt_tokens", 0)
        if kind == "query":
            self.query_tokens += tokens
        else:
            self.doc_tokens += tokens
        # OpenAI tra ve dung thu tu input; sort theo index de chac chan.
        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        start = time.time()
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._call(texts[i:i + self.batch_size]))
        self.doc_embed_seconds += time.time() - start
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._call([question], kind="query")[0]

    def cost_usd(self) -> float:
        price = PRICE_PER_MILLION_TOKENS[self.model_name]
        return (self.doc_tokens + self.query_tokens) / 1_000_000 * price
