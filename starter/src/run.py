"""Pipeline Day 06: corpus -> chia chunk -> embedding -> Chroma -> benchmark.

Chay tu thu muc starter/:
    python -m src.run                      # chay day du, goi API embedding
    python -m src.run --reuse-vectors      # tai su dung vector da luu neu so
                                           # chunk khop; chi goi API cho
                                           # embedding cau hoi khi truy xuat

Cac ket qua ghi vao thu muc results/ o goc repo:
    trace_A.json / trace_B.json / trace_C.json  - top-3 moi cau hoi moi strategy
    metrics.json                                - Hit@1, Hit@3, thoi gian, chi phi
    audit_samples.json                          - mau cho cham thu cong 3 khia canh
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import OpenAIEmbedder, PRICE_PER_MILLION_TOKENS
from .evaluation import evaluate_query, norm, summarize
from .ingestion import load_corpus
from .ingestion import normalize_text as page_norm
from .models import Chunk, DocumentPage
from .store import ChromaStore

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STARTER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

COHERENCE_LOCATIONS = 10   # so vi tri noi dung de cham coherence thu cong
SNIPPET_LEN = 240          # do dai vi tri noi dung (ky tu dau trang)


def build_strategies(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    """Ba cau hinh: A cua so co dinh, B theo doan, C giong B (chi khac cach tim)."""
    strategies: dict[str, list[Chunk]] = {}
    strategies["A"] = [c for p in pages for c in fixed_size(p, size=500, overlap=50)]
    strategies["B"] = [c for p in pages for c in by_paragraph(p, max_size=1200)]
    strategies["C"] = [c for p in pages for c in by_paragraph(p, max_size=1200)]
    return strategies


def chunk_stats(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "total_chars": sum(lengths),
        "avg_len": round(statistics.mean(lengths), 1) if lengths else 0,
        "median_len": statistics.median(lengths) if lengths else 0,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
    }


def verify_benchmark(bench: dict, pages: list[DocumentPage]) -> None:
    """Kiem chung lai evidence ngay trong pipeline; dung lai thi khong chay."""
    page_map = {(p.document_id, p.page): p.text for p in pages}
    for q in bench["questions"]:
        key = (q["gold_source_id"], q["gold_page"])
        if key not in page_map:
            raise RuntimeError(f"{q['id']}: trang gold {key} khong ton tai trong corpus")
        if norm(q["evidence_quote"]) not in norm(page_map[key]):
            raise RuntimeError(f"{q['id']}: evidence khong nam trong trang gold")
    print(f"[benchmark] {len(bench['questions'])} cau deu khop evidence - OK")


def coherence_locations(pages: list[DocumentPage], strategies: dict[str, list[Chunk]]
                        ) -> list[dict]:
    """Lay COHERENCE_LOCATIONS vi tri dau tien (sap theo document_id, page).

    Vi tri = 240 ky tu dau cua trang. Cung mot vi tri co the roi vao nhieu
    chunk (ca A va B deu cover); tim chunk chua vi tri o tung strategy.
    """
    locations = []
    for page in sorted(pages, key=lambda p: (p.document_id, p.page))[:COHERENCE_LOCATIONS]:
        snippet = page_norm(page.text[:SNIPPET_LEN])
        entry: dict = {
            "document_id": page.document_id, "page": page.page,
            "location_text": snippet, "chunks": {},
        }
        for strategy, chunks in strategies.items():
            covering = [c for c in chunks
                        if c.document_id == page.document_id
                        and c.page == page.page
                        and norm(snippet) in norm(c.text)]
            entry["chunks"][strategy] = [
                {"chunk_id": c.chunk_id, "text": c.text} for c in covering
            ]
        locations.append(entry)
    return locations


def main() -> None:
    load_dotenv(STARTER / ".env")
    RESULTS.mkdir(exist_ok=True)
    run_started = time.time()

    # 1. Doc corpus va benchmark; kiem tra it nhat 30 trang noi dung.
    pages = load_corpus(ROOT / "data" / "corpus.json")
    print(f"[corpus] {len(set(p.document_id for p in pages))} tai lieu, "
          f"{len(pages)} trang, {sum(len(p.text) for p in pages)} ky tu")
    if len(pages) < 30:
        raise RuntimeError(f"Corpus chi co {len(pages)} trang - khong du 30 trang yeu cau")

    bench = json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    verify_benchmark(bench, pages)

    # 2. Chia chunk cho ba strategy va ghi thong ke.
    strategies = build_strategies(pages)
    stats = {name: chunk_stats(chunks) for name, chunks in strategies.items()}
    for name, s in stats.items():
        print(f"[chunk] {name}: {s['chunk_count']} chunk, "
              f"TB {s['avg_len']} ky tu, lon nhat {s['max_len']}")

    # 3. Embedding that qua OpenAI; loi API thi ket thuc, khong fallback mock.
    embedder = OpenAIEmbedder(
        model_name=os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small")
    print(f"[embed] model {embedder.model_name}, {embedder.dimensions} chieu")

    store = ChromaStore(path=str(STARTER / "vector_store"),
                        dimensions=embedder.dimensions)
    store.create_schema()

    # Tùy chọn --reuse-vectors: dung lai vector da luu trong store cho
    # strategy nao co so chunk khop (chi dung khi corpus va chunking khong
    # doi) de khong goi lai API embedding khi chay lai de kiem tra.
    reuse = "--reuse-vectors" in sys.argv
    if reuse:
        print("[option] --reuse-vectors: tai su dung vector da luu neu so chunk khop")

    reused: list[str] = []
    for name, chunks in strategies.items():
        stored = store.count(name)
        if reuse and stored == len(chunks):
            stats[name]["reingest_ok"] = True
            stats[name]["reused_vectors"] = True
            stats[name]["ingest_seconds"] = 0.0
            reused.append(name)
            print(f"[store] {name}: {stored}/{len(chunks)} chunk - "
                  f"dung lai vector da luu, bo qua embedding")
            continue
        t0 = time.time()
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.reset(name)                       # nap lai tu dau, khong nhan doi
        store.upsert(name, chunks, vectors)
        count_first = store.count(name)
        store.upsert(name, chunks, vectors)     # upsert lan nua de kiem tra idempotent
        count_second = store.count(name)
        if count_first != count_second:
            raise RuntimeError(f"{name}: nap lai bi nhan doi ({count_first} -> {count_second})")
        stats[name]["reingest_ok"] = count_first == len(chunks)
        stats[name]["reused_vectors"] = False
        stats[name]["ingest_seconds"] = round(time.time() - t0, 1)
        print(f"[store] {name}: {count_first}/{len(chunks)} chunk, "
              f"nap lai khong nhan doi: {stats[name]['reingest_ok']}")

    # 4. Chay 20 cau hoi tren ba strategy, top-3, luu trace day du.
    traces: list[dict] = []
    query_seconds = 0.0
    for name in ("A", "B", "C"):
        for q in bench["questions"]:
            t0 = time.time()
            qvec = embedder.embed_query(q["question"])
            query_seconds += time.time() - t0
            # Chi C ap dung filter khi cau hoi co filter (gia tri do nguoi dung
            # cung cap truoc); A va B luon tim khong filter.
            filters = q.get("filter") if name == "C" else None
            hits = store.search(name, qvec, top_k=3, filters=filters)
            trace = evaluate_query(q, hits)
            trace["strategy"] = name
            traces.append(trace)

    # 5. Gop chi so va luu tat ca artefacts.
    metrics = summarize(traces)
    run_stats = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": embedder.model_name,
        "dimensions": embedder.dimensions,
        "reused_strategies": reused,
        "vector_store": f"chromadb {__import__('chromadb').__version__}, "
                        f"cosine, collection day6_A/B/C, du lieu tai "
                        f"starter/vector_store",
        "document_tokens": embedder.doc_tokens,
        "query_tokens": embedder.query_tokens,
        "total_tokens": embedder.doc_tokens + embedder.query_tokens,
        "price_per_million_tokens": PRICE_PER_MILLION_TOKENS[embedder.model_name],
        "cost_usd": round(embedder.cost_usd(), 6),
        "doc_embed_seconds": round(embedder.doc_embed_seconds, 1),
        "query_embed_seconds": round(query_seconds, 1),
        "run_seconds": round(time.time() - run_started, 1),
        "chunk_stats": stats,
    }

    for name in ("A", "B", "C"):
        part = [t for t in traces if t["strategy"] == name]
        (RESULTS / f"trace_{name}.json").write_text(
            json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "run_stats": run_stats},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    # 6. Xuat mau de cham thu cong coherence, grounding, metadata.
    audit = {
        "coherence_locations": coherence_locations(pages, strategies),
        "grounding_test": [
            {"strategy": t["strategy"], "question_id": t["question_id"],
             "question": t["question"], "gold": f"{t['gold_source_id']} trang {t['gold_page']}",
             "evidence_quote": next(q["evidence_quote"] for q in bench["questions"]
                                    if q["id"] == t["question_id"]),
             "top3": t["hits"]}
            for t in traces
            if t["split"] == "test"
        ],
    }
    (RESULTS / "audit_samples.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")

    # 7. In bang ket qua de kiem tra nhanh.
    print("\n=== Hit@1 / Evidence Hit@3 ===")
    for name, splits in metrics.items():
        for split, m in splits.items():
            print(f"  {name} {split}: Hit@1 {m['hit@1']}  Hit@3 {m['hit@3']}")
    print(f"\nChi phi embedding: {run_stats['total_tokens']} token = "
          f"{run_stats['cost_usd']} USD; thoi gian chay: {run_stats['run_seconds']}s")
    print(f"Da ghi ket qua vao {RESULTS}")


if __name__ == "__main__":
    main()
