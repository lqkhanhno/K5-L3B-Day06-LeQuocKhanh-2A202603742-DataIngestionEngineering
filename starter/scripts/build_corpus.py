"""Xay corpus lich su Viet Nam tu Wikipedia tieng Viet.

Moi bai viet la mot don vi nguon (document). Van ban duoc chuan hoa roi chia
thanh cac "trang" khoang 2000-2400 ky tu tai ranh gioi doan van. Ket qua:
  data/corpus.json  - noi dung da chuan hoa theo tung don vi/trang
  data/manifest.csv - khai bao nguon, giay phep, so trang, so ky tu

Cach dung: chay tu thu muc starter/
  python scripts/build_corpus.py
"""
from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import requests
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "https://vi.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0"
USER_AGENT = "Day06DataIngestionLab/1.0 (lab hoc phan; lien he qua Wikipedia)"

# Danh sach bai viet theo chu de lich su Viet Nam, kem nhan thoi ky dung de loc.
# Nhan "period" gan theo nam bat dau chu de chinh cua bai (nam nay duoc ghi trong
# chinh bai Wikipedia do) va chia bon moc: 938 (thang loi cua Ngo Quyen),
# 1428 (Le Loi lap nha Le so), 1858 (Phap tan cong Da Nang). Bon gia tri:
# co_dai (truoc 938), 938_1427, 1428_1857, 1858_ve_sau.
ARTICLES: list[tuple[str, str, str]] = [
    # (document_id, ten bai Wikipedia, period)
    ("hong_bang", "Hồng Bàng", "co_dai"),
    ("an_duong_vuong", "An Dương Vương", "co_dai"),
    ("nha_trieu", "Nhà Triệu", "co_dai"),
    ("khoi_nghia_hai_ba_trung", "Khởi nghĩa Hai Bà Trưng", "co_dai"),
    ("trieu_thi_trinh", "Triệu Thị Trinh", "co_dai"),
    ("ly_bi", "Lý Nam Đế", "co_dai"),
    ("ngo_quyen", "Ngô Quyền", "938_1427"),
    ("tran_bach_dang_938", "Trận Bạch Đằng (938)", "938_1427"),
    ("dinh_bo_linh", "Đinh Bộ Lĩnh", "938_1427"),
    ("nha_tien_le", "Nhà Tiền Lê", "938_1427"),
    ("nha_ly", "Nhà Lý", "938_1427"),
    ("ly_cong_uan", "Lý Công Uẩn", "938_1427"),
    ("ly_thuong_kiet", "Lý Thường Kiệt", "938_1427"),
    ("nha_tran", "Nhà Trần", "938_1427"),
    ("tran_hung_dao", "Trần Hưng Đạo", "938_1427"),
    ("khang_chien_nguyen_tran", "Kháng chiến chống quân Nguyên Mông", "938_1427"),
    ("tran_bach_dang_1288", "Trận Bạch Đằng (1288)", "938_1427"),
    ("ho_quy_ly", "Hồ Quý Ly", "938_1427"),
    ("nha_ho", "Nhà Hồ", "938_1427"),
    ("le_loi", "Lê Lợi", "938_1427"),
    ("khoi_nghia_lam_son", "Khởi nghĩa Lam Sơn", "938_1427"),
    ("nha_le_som", "Nhà Lê sơ", "1428_1857"),
    ("khoi_nghia_tay_son", "Khởi nghĩa Tây Sơn", "1428_1857"),
    ("nguyen_hue", "Nguyễn Huệ", "1428_1857"),
    ("ngoc_hoi_dong_da", "Trận Ngọc Hồi – Đống Đa", "1428_1857"),
    ("nha_nguyen", "Nhà Nguyễn", "1428_1857"),
    ("minh_mang", "Minh Mạng", "1428_1857"),
    ("phan_boi_chau", "Phan Bội Châu", "1858_ve_sau"),
    ("phan_chau_trinh", "Phan Châu Trinh", "1858_ve_sau"),
    ("khoi_nghia_yen_bai", "Khởi nghĩa Yên Bái", "1858_ve_sau"),
    ("ho_chi_minh", "Hồ Chí Minh", "1858_ve_sau"),
    ("cach_mang_thang_tam", "Cách mạng Tháng Tám", "1858_ve_sau"),
    ("chien_dong_duong", "Chiến tranh Đông Dương", "1858_ve_sau"),
    ("chien_dich_dien_bien_phu", "Chiến dịch Điện Biên Phủ", "1858_ve_sau"),
    ("hiep_dinh_geneve", "Hiệp định Genève", "1858_ve_sau"),
    ("duong_truong_son", "Đường Trường Sơn", "1858_ve_sau"),
    ("chien_dich_hcm", "Chiến dịch Hồ Chí Minh", "1858_ve_sau"),
    ("doi_moi_1986", "Đổi Mới", "1858_ve_sau"),
]

# Cac muc chi co lien ket/tham khao, bo di khi doc bai.
DROP_SECTIONS = {
    "xem thêm", "chú thích", "tham khảo", "liên kết ngoài", "ghi chú",
    "thư mục", "trang chủ đề", "khảo cổ học", "hình ảnh", "đọc thêm",
}

MAX_UNIT_CHARS = 2400
MIN_UNIT_CHARS = 900  # duoi muc nay thi gop vao trang truoc do


def strip_wikitext(raw: str) -> list[tuple[str, str]]:
    """Bien wikitext thanh danh sach (cap_do_muc, doan_van) theo thu tu."""
    # Bo chu thich HTML
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    # Bo ref
    raw = re.sub(r"<ref[^>/]*?/>", "", raw, flags=re.I)
    raw = re.sub(r"<ref[^>]*?>.*?</ref>", "", raw, flags=re.S | re.I)
    # Bo bang {| ... |} va the HTML thong dung
    raw = re.sub(r"\{\|.*?\|\}", "", raw, flags=re.S)
    raw = re.sub(r"</?[a-zA-Z][^>]*?>", "", raw)

    # Bo template long nhau {{ ... }} (lap de xu viec long cap)
    for _ in range(6):
        new = re.sub(r"\{\{[^{}]*\}\}", "", raw)
        if new == raw:
            break
        raw = new

    blocks: list[tuple[str, str]] = []
    section = "mo_dau"
    paragraph: list[str] = []

    def flush() -> None:
        text = " ".join(" ".join(paragraph).split())
        if len(text) >= 80:  # bo doan qua ngan la loi le
            blocks.append((section, text))
        paragraph.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        heading = re.match(r"^(={2,5})\s*(.+?)\s*\1$", stripped)
        if heading:
            flush()
            title = heading.group(2).strip()
            # Bo lien ket ngon ngu trong tieu muc [[vi:...]]
            title = re.sub(r"\[\[[a-zA-Z-]+:[^]]+\]\]", "", title).strip()
            section = title
            continue
        if not stripped:
            flush()
            continue
        paragraph.append(stripped)

    flush()
    return blocks


def clean_inline(text: str) -> str:
    """Don dep cu phap trong doan van: lien ket, chu dam/nghieng, danh sach."""
    # [[File:..|..]] / [[Thể loại:..]] bo hoan toan
    text = re.sub(r"\[\[(?:Tập tin|File|Thể loại|Category|Image):[^\]]*\]\]", "", text, flags=re.I)
    # [[muc|chu hien]] -> chu hien ; [[muc]] -> muc
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)              # chu dam/nghieng
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)  # [url nhan] -> nhan
    text = re.sub(r"https?://\S+", "", text)       # URL tron
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    # Markup anh con sot: "nhỏ|phải|200px|", "thế=", ngoac [[ ]] le, dau | don le
    text = re.sub(r"\b(?:nhỏ|trái|phải|giữa|thế=)\|", "", text)
    text = re.sub(r"\d+px\|", "", text)
    text = text.replace("[[", " ").replace("]]", " ")
    text = text.replace("|", " ")
    # Bo dau gach dau dong danh sach, giu lai van ban
    text = re.sub(r"^[*#:;]+\s*", "", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Chuan hoa Unicode NFC va khoang trang, giu nguyen dung chinh ta."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_units(blocks: list[tuple[str, str]]) -> list[dict]:
    """Gom doan van thanh cac trang <= MAX_UNIT_CHARS theo ranh gioi doan."""
    units: list[dict] = []
    current: list[str] = []
    current_len = 0

    def push() -> None:
        nonlocal current, current_len
        if not current:
            return
        text = normalize_text("\n".join(current))
        if units and len(text) < MIN_UNIT_CHARS:
            # Doan ngan: gop vao trang truoc neu co the
            prev = units[-1]["text"]
            if len(prev) + len(text) + 1 <= MAX_UNIT_CHARS + 600:
                units[-1]["text"] = normalize_text(prev + "\n" + text)
                current, current_len = [], 0
                return
        units.append({"text": text, "chars": len(text)})
        current, current_len = [], 0

    for _section, para in blocks:
        # Doan qua dai (thuong la danh sach gop lai): tach theo cau
        if len(para) > MAX_UNIT_CHARS:
            pieces: list[str] = []
            buf = ""
            for sent in re.split(r"(?<=[.!?…])\s+", para):
                if buf and len(buf) + len(sent) + 1 > MAX_UNIT_CHARS:
                    pieces.append(buf)
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                pieces.append(buf)
            para_pieces = pieces
        else:
            para_pieces = [para]

        for piece in para_pieces:
            if current_len + len(piece) + 1 > MAX_UNIT_CHARS and current:
                push()
            current.append(piece)
            current_len += len(piece) + 1
    push()

    for i, unit in enumerate(units, start=1):
        unit["page"] = i
    return units


def fetch_wikitext(doc_id: str, title: str, cache_dir: Path) -> str:
    """Lay wikitext voi bo dem tren dia de chay lai khong phai tai lai.

    Ten file cache dung document_id (da la ASCII khong dau) de ten file
    doc duoc va khong bi loi khi hien thi tren Windows.
    """
    cache_file = cache_dir / f"{doc_id}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    for attempt in range(4):
        resp = requests.get(
            API,
            params={"action": "parse", "page": title, "prop": "wikitext", "format": "json",
                    "redirects": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"  429 voi '{title}', doi {wait}s roi thu lai...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Wikipedia bao loi voi '{title}': {data['error'].get('info')}")
        text = data["parse"]["wikitext"]["*"]
        cache_file.write_text(text, encoding="utf-8")
        return text
    raise RuntimeError(f"'{title}': van bi 429 sau 4 lan thu")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)

    errors: list[str] = []
    documents = []
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "raw_wikitext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, title, period in ARTICLES:
        try:
            raw = fetch_wikitext(doc_id, title, cache_dir)
        except Exception as exc:  # ghi loi va tiep tuc
            errors.append(f"{doc_id} ({title}): {exc}")
            print(f"LOI  {doc_id}: {exc}")
            continue
        blocks = []
        for section, para in strip_wikitext(raw):
            if section.lower() in DROP_SECTIONS:
                continue
            cleaned = clean_inline(para)
            if cleaned:
                blocks.append((section, cleaned))
        units = split_units(blocks)
        total = sum(u["chars"] for u in units)
        if total < 1500:
            errors.append(f"{doc_id} ({title}): chi {total} ky tu sau chuan hoa, bo")
            print(f"BO  {doc_id}: qua ngan ({total} ky tu)")
            continue
        documents.append({
            "document_id": doc_id,
            "title": title,
            "url": f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "license": LICENSE,
            "accessed_at": date.today().isoformat(),
            "period": period,
            "total_chars": total,
            "units": units,
        })
        print(f"OK   {doc_id}: {len(units)} trang, {total} ky tu")
        time.sleep(1.5)

    corpus = {
        "generated_at": date.today().isoformat(),
        "source": "Wikipedia tieng Viet",
        "license": LICENSE,
        "attribution": "Nguoi viet cac bai Wikipedia tieng Viet, phat hanh theo giay phep CC BY-SA 4.0",
        "documents": documents,
    }
    (out_dir / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")

    # utf-8-sig (co BOM) de mo bang Excel tren Windows khong mat dau tieng Viet
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "source_id", "title", "source_path_or_url", "license_or_permission",
            "accessed_at", "filter_period", "page_ids", "page_count", "total_chars",
            "content_rule",
        ])
        for doc in documents:
            pages = ";".join(str(u["page"]) for u in doc["units"])
            writer.writerow([
                doc["document_id"], doc["title"], doc["url"], doc["license"],
                doc["accessed_at"], doc["period"], pages, len(doc["units"]),
                doc["total_chars"],
                "Mot bai Wikipedia = mot don vi nguon; moi don vi chia thanh cac trang "
                "toi da 2400 ky tu tai ranh gioi doan van, chi tinh trang >= 900 ky tu "
                "noi dung chuan hoa sau khi bo muc tham khao/lien ket.",
            ])

    total_pages = sum(len(d["units"]) for d in documents)
    print(f"\nTong: {len(documents)} don vi nguon, {total_pages} trang, "
          f"{sum(d['total_chars'] for d in documents)} ky tu.")
    if errors:
        print("Loi/bo qua:")
        for line in errors:
            print(f"  - {line}")
    if len(documents) < 30:
        raise SystemExit("Khong du 30 don vi nguon, can bo sung danh sach bai.")


if __name__ == "__main__":
    main()
