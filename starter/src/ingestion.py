"""Doc va chuan hoa du lieu nguon.

Corpus chinh la data/corpus.json do scripts/build_corpus.py tao tu Wikipedia
tieng Viet (moi bai viet = mot don vi nguon, chia thanh cac "trang" tai ranh
gioi doan van). Van giu load_pdf de nguoi hoc co the thay corpus PDF khac.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

from .models import DocumentPage

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_text(text: str) -> str:
    """Chuan hoa Unicode NFC va gap khoang trang; giu nguyen dung chinh ta.

    Cac so sanh evidence o cac buoc sau deu chay tren kieu chuan hoa nay
    (them casefold o buoc so khop) nen van ban va quote luon khop kieu.
    """
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Trich van ban tung trang vat ly cua PDF; trang rong duoc ghi loi.

    Neu la trang scan (khong co text), ghi vao log loi de biet phan can OCR
    thay the; khong tu OCR trong ham nay.
    """
    from pypdf import PdfReader  # import tai cho de khong bat buoc khi dung corpus JSON

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    empty_pages: list[int] = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        if not text:
            empty_pages.append(i)
            continue
        pages.append(DocumentPage(
            document_id=document_id, source=source, page=i, text=text,
        ))
    if empty_pages:
        print(f"[ingestion] {document_id}: trang khong co text (can OCR hoac bo): {empty_pages}")
    return pages


def load_corpus(path: Path) -> list[DocumentPage]:
    """Doc corpus.json thanh danh sach trang, kèm metadata (title, period)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pages: list[DocumentPage] = []
    empty_pages: list[str] = []
    for doc in data["documents"]:
        for unit in doc["units"]:
            text = normalize_text(unit["text"])
            if not text:
                empty_pages.append(f"{doc['document_id']} trang {unit['page']}")
                continue
            pages.append(DocumentPage(
                document_id=doc["document_id"],
                source=doc["url"],
                page=unit["page"],
                text=text,
                metadata={
                    "title": doc["title"],
                    "period": doc["period"],
                    "license": doc["license"],
                    "accessed_at": doc["accessed_at"],
                },
            ))
    if empty_pages:
        print(f"[ingestion] trang rong bi bo: {empty_pages}")
    return pages
