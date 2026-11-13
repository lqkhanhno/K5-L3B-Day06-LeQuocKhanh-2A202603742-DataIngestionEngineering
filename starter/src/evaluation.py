"""Do hit cua ket qua truy xuat va giu trace day du tung cau.

Dinh nghia hit (theo GUIDE.md muc 4): mot chunk chi tinh la hit khi no
thuoc dung tai lieu va trang gold, dong thoi chua du evidence_quote da
kiem chung (so khop chuoi sau chuan hoa NFC + casefold + gap khoang trang).
Hit@1 tinh tren chunk dau tien; Evidence Hit@3 tinh neu co bat ky chunk
nao trong top-3 dat. Nhan coherence/grounding con lai cho nguoi cham thu
cong lam tren mau audit, khong suy ra tu so khop chuoi.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def norm(text: str) -> str:
    """Kieu so khop dung chung cho evidence: NFC, casefold, gap khoang trang."""
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Kiem tra tung hit: dung trang gold khong, chua du evidence khong."""
    gold_doc = question["gold_source_id"]
    gold_page = question["gold_page"]
    evidence = norm(question["evidence_quote"])

    labeled_hits: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        on_gold = hit.get("document_id") == gold_doc and hit.get("page") == gold_page
        has_evidence = evidence in norm(hit.get("text", ""))
        labeled_hits.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "document_id": hit.get("document_id"),
            "page": hit.get("page"),
            "score": hit.get("score"),
            "on_gold_page": on_gold,
            "contains_evidence": has_evidence,
            "text": hit.get("text", ""),
            "period": hit.get("period"),
            "source": hit.get("source"),
            "title": hit.get("title"),
        })

    hit1 = bool(labeled_hits and labeled_hits[0]["on_gold_page"]
                and labeled_hits[0]["contains_evidence"])
    hit3 = any(h["on_gold_page"] and h["contains_evidence"] for h in labeled_hits)

    return {
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question.get("expected_answer"),
        "gold_source_id": gold_doc,
        "gold_page": gold_page,
        "filter": question.get("filter"),
        "hit@1": hit1,
        "hit@3": hit3,
        "hits": labeled_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Gop Hit@1 va Evidence Hit@3 theo strategy va split (dev/test)."""
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for trace in traces:
        strategy = trace["strategy"]
        split = trace["split"]
        bucket = summary.setdefault(strategy, {}).setdefault(split, {
            "n": 0, "hit@1": 0, "hit@3": 0,
        })
        bucket["n"] += 1
        bucket["hit@1"] += int(trace["hit@1"])
        bucket["hit@3"] += int(trace["hit@3"])
    result: dict[str, Any] = {}
    for strategy, splits in summary.items():
        result[strategy] = {}
        for split, bucket in splits.items():
            n = bucket["n"]
            result[strategy][split] = {
                "n": n,
                "hit@1": f"{bucket['hit@1']}/{n}",
                "hit@3": f"{bucket['hit@3']}/{n}",
            }
    return result
