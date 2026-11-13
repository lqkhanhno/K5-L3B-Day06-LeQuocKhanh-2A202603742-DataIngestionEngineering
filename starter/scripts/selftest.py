"""Self-test cho bai lam Day 06 — chay tu thu muc starter/.

Ba che do:
    python scripts/selftest.py --env       # kiem tra .env co key va git khong
                                           # track .env — KHONG goi API
    python scripts/selftest.py             # test tung module: corpus, chunk,
                                           # embedding that (vai tram token),
                                           # luu/tim/loc tren Chroma
    python scripts/selftest.py --results   # doc ket qua da luu trong results/
                                           # va dem vector trong store,
                                           # KHONG goi API, khong ton tien

Mẹo: khong dung "python -c" cho cac lenh co dau nhay trong PowerShell 5.1 —
no lam mat dau ngoac kep. Dung file script nhu file nay thay the.
"""
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STARTER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STARTER))

FAILED = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "OK  " if cond else "SAI "
    if not cond:
        FAILED.append(label)
    print(f"{mark} {label}" + (f" - {detail}" if detail else ""))


def test_modules() -> None:
    from dotenv import load_dotenv
    load_dotenv(STARTER / ".env")

    from src.chunking import by_paragraph, fixed_size
    from src.embeddings import OpenAIEmbedder
    from src.evaluation import norm
    from src.ingestion import load_corpus
    from src.store import ChromaStore

    pages = load_corpus(ROOT / "data" / "corpus.json")
    check("1. corpus doc duoc", len(pages) >= 30, f"{len(pages)} trang")

    p = pages[0]
    a, b = fixed_size(p), by_paragraph(p)
    check("2. chia chunk A va B co metadata trang",
          len(a) >= 1 and len(b) >= 1 and all(c.page == p.page for c in a + b),
          f"A {len(a)} chunk, B {len(b)} chunk")

    emb = OpenAIEmbedder()
    vecs = emb.embed_documents([a[0].text, b[0].text])
    check("3. embedding that", len(vecs) == 2 and len(vecs[0]) == emb.dimensions,
          f"{emb.model_name}, {emb.dimensions} chieu, {emb.doc_tokens} token")

    store = ChromaStore(path=str(STARTER / "vector_store_selftest"),
                        dimensions=emb.dimensions)
    store.reset("SELFTEST")
    store.upsert("SELFTEST", [a[0], b[0]], vecs)
    n1 = store.count("SELFTEST")
    store.upsert("SELFTEST", [a[0], b[0]], vecs)
    check("4. nap lai khong nhan doi", n1 == 2 and store.count("SELFTEST") == 2,
          f"{n1} -> {store.count('SELFTEST')}")

    hits = store.search("SELFTEST", emb.embed_query(p.metadata.get("title", "lich su")), top_k=2)
    check("5. tim kiem tra ve ket qua", len(hits) == 2 and hits[0]["score"] > 0,
          f"score cao nhat {hits[0]['score']:.3f}")

    hits_f = store.search("SELFTEST", emb.embed_query("kiem tra"), top_k=2,
                          filters={"field": "period", "value": "khong ton tai"})
    check("6. filter gia tri khong co -> 0 hit", len(hits_f) == 0)

    evidence_page = next(p2 for p2 in pages if p2.document_id == "tran_bach_dang_938" and p2.page == 1)
    check("7. norm() khop evidence",
          norm("người Việt giành thắng lợi") in norm(evidence_page.text))

    shutil.rmtree(STARTER / "vector_store_selftest", ignore_errors=True)
    print()
    if FAILED:
        print("CO LOI:", FAILED)
        sys.exit(1)
    print("SELFTEST MODULE DAT — pipeline san sang chay buoc 8 (python -m src.run)")


def test_results() -> None:
    metrics_file = ROOT / "results" / "metrics.json"
    if not metrics_file.exists():
        print("Chua co results/metrics.json — chay python -m src.run truoc.")
        sys.exit(1)
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    for name, splits in data["metrics"].items():
        for split, m in splits.items():
            print(f"{name} {split}: Hit@1 {m['hit@1']}  Hit@3 {m['hit@3']}")
    st = data["run_stats"]
    print(f"chi phi: {st['cost_usd']} USD | token: {st['total_tokens']} | "
          f"chay: {st['run_seconds']}s")

    try:
        from src.store import ChromaStore
        store = ChromaStore(path=str(STARTER / "vector_store"), dimensions=1536)
        for name in "ABC":
            print(f"collection day6_{name}: {store.count(name)} chunk (luu ben vung tren dia)")
    except Exception as exc:
        print(f"store: chua doc duoc ({exc})")

    files = ["trace_A.json", "trace_B.json", "trace_C.json", "metrics.json",
             "audit_samples.json", "manual_labels.json"]
    missing = [f for f in files if not (ROOT / "results" / f).exists()]
    check("du 6 file ket qua", not missing, f"thieu: {missing}" if missing else "")

    labels = json.loads((ROOT / "results" / "manual_labels.json").read_text(encoding="utf-8"))
    check("coherence A 0.9 / B 2.0 / C 2.0",
          labels["coherence"]["average"] == {"A": 0.9, "B": 2.0, "C": 2.0})
    check("grounding A 1.6 / B 1.2 / C 1.2",
          labels["grounding"]["average"] == {"A": 1.6, "B": 1.2, "C": 1.2})
    check("metadata 45/45", labels["metadata"]["complete_and_correct"] == "45/45")
    print()
    print("SELFTEST RESULTS DAT — ket qua trong results/ khop voi bao cao"
          if not FAILED else f"CO LOI: {FAILED}")


def test_env() -> None:
    """Kiem tra .env va git ma khong goi API nao."""
    from dotenv import load_dotenv
    load_dotenv(STARTER / ".env")
    import os

    key = os.getenv("OPENAI_API_KEY", "").strip()
    check("starter/.env co OPENAI_API_KEY", bool(key),
          f"bat dau bang {key[:12]}, do dai {len(key)}" if key
          else "hoac file khong ton tai, hoac khong co key")
    model = os.getenv("EMBEDDING_MODEL", "").strip() or "text-embedding-3-small (mac dinh)"
    print(f"     model se dung: {model}")

    import subprocess
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except Exception as exc:
        print(f"     bo qua kiem tra git ({exc})")
        tracked = []
    bad = [line for line in tracked if line.endswith("/.env") or line == ".env"]
    check("git khong track file .env", not bad,
          f"dang bi track: {bad}" if bad else "chi co starter/.env.example trong git")
    print()
    if FAILED:
        print("CO LOI:", FAILED)
        sys.exit(1)
    print("SELFTEST ENV DAT — co the chay tiep python scripts/selftest.py hoac python -m src.run")


if __name__ == "__main__":
    if "--results" in sys.argv:
        test_results()
    elif "--env" in sys.argv:
        test_env()
    else:
        test_modules()
