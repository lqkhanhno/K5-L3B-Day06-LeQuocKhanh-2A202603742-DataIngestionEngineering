"""Sinh lai mau audit coherence voi vi tri o giua noi dung trang.

Vi tri = 240 ky tu bat dau tu 55% do dai trang (dung quy tac truoc khi xem
chunk nao cover, khong chinh sua sau khi diem). Chunk lay tu chuc nang chia
chunk nhung chinh la chunk da nap vao store (chunking la ham xac dinh).
Grounding giu nguyen tu trace that da chay.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import by_paragraph, fixed_size  # noqa: E402
from src.evaluation import norm  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402
from src.ingestion import normalize_text as page_norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SNIPPET_LEN = 240
OFFSET_RATIO = 0.55  # vi tri nam sau dau trang de thay cach cat chunk


def main() -> None:
    pages = load_corpus(ROOT / "data" / "corpus.json")
    page_map = {p.document_id + f"#p{p.page}": p for p in pages}
    strategies = {
        "A": [c for p in pages for c in fixed_size(p, size=500, overlap=50)],
        "B": [c for p in pages for c in by_paragraph(p, max_size=1200)],
    }

    locations = []
    for page in sorted(pages, key=lambda p: (p.document_id, p.page))[:10]:
        offset = int(len(page.text) * OFFSET_RATIO)
        snippet = page_norm(page.text[offset:offset + SNIPPET_LEN])
        entry = {
            "document_id": page.document_id,
            "page": page.page,
            "location_rule": f"240 ky tu tu vi tri {OFFSET_RATIO:.0%} do dai trang",
            "location_text": snippet,
            "chunks": {},
        }
        for strategy, chunks in strategies.items():
            # chunk cover = chunk chua doan dau vi tri (thu 80 ky tu, roi 40
            # neu vi tri dau tien bi cat rach qua ranh gioi chunk). Neu khong
            # co chunk nao chua tron 240 ky tu thi vi tri bi tach qua nhieu chunk.
            matching: list = []
            for head_len in (80, 40, 20):
                head = norm(snippet[:head_len])
                matching = [c for c in chunks
                            if c.document_id == page.document_id
                            and c.page == page.page
                            and head in norm(c.text)]
                if matching:
                    break
            entry["chunks"][strategy] = [
                {"chunk_id": c.chunk_id,
                 "contains_full_location": norm(snippet) in norm(c.text),
                 "text": c.text}
                for c in matching
            ]
        locations.append(entry)

    audit = json.loads((RESULTS / "audit_samples.json").read_text(encoding="utf-8"))
    audit["coherence_locations"] = locations
    audit["coherence_note"] = (
        "Vi tri chon theo quy tac co dinh (10 trang dau sap theo document_id, "
        "doan 240 ky tu tai 55% do dai trang) truoc khi xem chunk; C dung "
        "chung chunk voi B nen chi cham A va B.")
    (RESULTS / "audit_samples.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Da sinh lai {len(locations)} vi tri coherence tai {RESULTS/'audit_samples.json'}")


if __name__ == "__main__":
    main()
