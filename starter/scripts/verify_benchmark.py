"""Kiem chung moi evidence_quote nam dung tai trang gold trong corpus.

Bat buoc chay truoc khi chay pipeline. Chay tu thu muc starter/
  python scripts/verify_benchmark.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]


def norm(text: str) -> str:
    """Giong ham chuan hoa dung khi do hit trong evaluation.py."""
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text)


def main() -> int:
    corpus = json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
    bench = json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))

    pages = {(doc["document_id"], unit["page"]): unit["text"]
             for doc in corpus["documents"] for unit in doc["units"]}

    dev = sum(1 for q in bench["questions"] if q["split"] == "dev")
    test = sum(1 for q in bench["questions"] if q["split"] == "test")
    n_filter = sum(1 for q in bench["questions"] if "filter" in q)
    test_filter = sum(1 for q in bench["questions"] if "filter" in q and q["split"] == "test")
    print(f"Benchmark: {len(bench['questions'])} cau ({dev} dev / {test} test), "
          f"{n_filter} cau filter (test co {test_filter})")

    failed = 0
    for q in bench["questions"]:
        key = (q["gold_source_id"], q["gold_page"])
        if key not in pages:
            print(f"SAI {q['id']}: khong ton tai trang {key}")
            failed += 1
            continue
        ok = norm(q["evidence_quote"]) in norm(pages[key])
        if ok:
            print(f"OK   {q['id']} ({q['split']}) -> {q['gold_source_id']} trang {q['gold_page']}")
        else:
            print(f"SAI {q['id']}: evidence khong nam trong {q['gold_source_id']} trang {q['gold_page']}")
            print(f"     quote : {q['evidence_quote'][:90]}")
            failed += 1

    if failed:
        print(f"\n{failed} cau chua dat - can sua quote hoac trang gold.")
        return 1
    print("\nToan bo evidence da kiem chung dat. Benchmark san sang de chay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
