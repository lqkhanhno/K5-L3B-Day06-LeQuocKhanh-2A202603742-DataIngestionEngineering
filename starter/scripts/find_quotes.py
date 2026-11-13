"""Cong cu tim cau trong corpus de viet benchmark khop chinh xac.

Chay tu thu muc starter/  :  python scripts/find_quotes.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text)


def main() -> None:
    keywords = sys.argv[1:]
    corpus = json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
    for doc in corpus["documents"]:
        for unit in doc["units"]:
            n = norm(unit["text"])
            flat = re.sub(r"\s+", " ", unit["text"])
            for kw in keywords:
                kwn = norm(kw)
                idx = n.find(kwn)
                if idx >= 0:
                    start = max(0, idx - 160)
                    end = min(len(n), idx + len(kwn) + 260)
                    print(f"--- {doc['document_id']} p{unit['page']} [{doc['title']}]")
                    # in theo doan goc, dem it xuong dong thanh khoang trang
                    print(re.sub(r"\s+", " ", flat[start:end]))
                    print()
                    break


if __name__ == "__main__":
    main()
