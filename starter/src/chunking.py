"""Hai cach chia chunk: cua so co dinh (A) va theo doan van (B, C).

Cung mot trang co the cho ra nhieu chunk; moi chunk deu giu day du
document_id / source / page de truy ve nguon, va mang metadata cua trang.
"""
from __future__ import annotations

import re

from .models import Chunk, DocumentPage

# Tach cau cho doan van qua dai: diem ket cau thuong gap trong tieng Viet.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Chia thanh cua so ky tu dinh kich thuoc, co de chong lap (overlap).

    Cua so luon cat tai khoang trang de khong ngat giua mot tu.
    """
    text = page.text
    if not text:
        return []
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    start = 0
    index = 1
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # dua end ve dau khoang trang gan nhat de khong cat tu
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-a{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=piece,
                metadata=dict(page.metadata),
            ))
            index += 1
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 1200) -> list[Chunk]:
    """Gom cac doan van (tach bang \n) thanh chunk toi da max_size ky tu.

    Doan van qua dai duoc tach theo cau de khong vuot gioi han. Chunk giu
    nguyen thu tu doan nen nghe doc lien mach hon cua so co dinh.
    """
    paragraphs = [p.strip() for p in page.text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= max_size:
            pieces.append(para)
            continue
        buf = ""
        for sent in SENTENCE_SPLIT.split(para):
            if buf and len(buf) + len(sent) + 1 > max_size:
                pieces.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf)

    chunks: list[Chunk] = []
    current: list[str] = []
    index = 1

    def flush() -> None:
        nonlocal current, index
        if not current:
            return
        text = " ".join(current)
        if chunks and len(text) < 150:
            # doan con lai qua ngan: gop vao chunk truoc cho tu nhien
            prev = chunks[-1].text
            if len(prev) + len(text) + 1 <= max_size + 200:
                chunks[-1].text = f"{prev} {text}"
                current = []
                return
        chunks.append(Chunk(
            chunk_id=f"{page.document_id}-p{page.page}-b{index}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=text.strip(),
            metadata=dict(page.metadata),
        ))
        index += 1
        current = []

    for piece in pieces:
        if current and sum(len(x) + 1 for x in current) + len(piece) + 1 > max_size:
            flush()
        current.append(piece)
    flush()
    return chunks
