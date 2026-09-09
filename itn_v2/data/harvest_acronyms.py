"""Thu hoạch viết tắt thật từ data_v1 vào danh mục acronyms.json.

Đúng nguyên tắc của v2.txt: danh mục thuật ngữ phải là TÀI SẢN DỮ LIỆU xây từ
cách đọc quan sát được trong dữ liệu thật, không phải danh sách tự suy đoán —
alias bịa làm tăng false accept.

Nguồn: các cặp (dạng viết là chuỗi in hoa, dạng nói là cụm tiếng Việt) mà bộ
chuyển đổi hiện tại chưa tra được.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from ..catalog import get_catalog
from .from_data_v1 import split_tokens

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "itn_v2" / "catalogs" / "acronyms.json"
# Chỉ chữ cái: chuỗi có chữ số là mã khí tài / phiên hiệu đơn vị (M41, Z133),
# thuộc EQUIPMENT_ID chứ không phải viết tắt.
ACRONYM_RE = re.compile(r"^[A-ZĐ]{2,8}$")
# Số La Mã KHÔNG phải viết tắt. "Đại hội XIII" đọc là "đại hội mười ba"; nếu nạp
# XIII <- "mười ba" vào danh mục thì mọi cụm "mười ba" trong văn bản sẽ bị biến
# thành XIII. Đây đúng là kiểu false accept mà alias tự suy đoán gây ra.
ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
VN_PHRASE_RE = re.compile(r"^[\w\s]+$", re.UNICODE)
MIN_COUNT = 2


def _is_number_phrase(spoken):
    """Cụm đọc số thuần tuý thì không được coi là cách đọc viết tắt."""
    from ..normalizers.base import ParseError
    from ..normalizers.number import read_cardinal
    try:
        read_cardinal(spoken.split())
        return True
    except (ParseError, ValueError):
        return False


def collect(input_dir):
    pairs = Counter()
    for path in sorted((ROOT / input_dir).rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for chunk in doc.get("chunks", []):
            for result in chunk.get("all_text_results", []):
                if result.get("type") != "pronunciation_variant":
                    continue
                ww, _ = split_tokens(chunk["text"])
                sw, _ = split_tokens(result["text"])
                matcher = SequenceMatcher(
                    None, [w.lower() for w in ww], [w.lower() for w in sw])
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "equal" or j1 == j2:
                        continue
                    written = " ".join(ww[i1:i2]).strip()
                    spoken = " ".join(sw[j1:j2]).strip()
                    if len(written.split()) != 1 or not ACRONYM_RE.match(written):
                        continue
                    if ROMAN_RE.match(written):
                        continue
                    if not (2 <= len(spoken.split()) <= 8):
                        continue
                    if any(ch.isdigit() for ch in spoken):
                        continue
                    if _is_number_phrase(spoken):
                        continue
                    pairs[(written, spoken.lower())] += 1
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_v1/json")
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pairs = collect(args.input)
    catalog = get_catalog("acronyms")
    by_canonical = defaultdict(set)
    for (written, spoken), count in pairs.items():
        if count < args.min_count:
            continue
        existing, score = catalog.lookup(spoken)
        if existing == written:
            continue
        by_canonical[written].add(spoken)

    raw = json.loads(CATALOG.read_text(encoding="utf-8"))
    index = {e["canonical"]: e for e in raw["entries"]}
    added = 0
    for canonical, spokens in sorted(by_canonical.items()):
        entry = index.get(canonical)
        if entry is None:
            entry = {"canonical": canonical, "spoken": []}
            raw["entries"].append(entry)
            index[canonical] = entry
        for spoken in sorted(spokens):
            if spoken not in entry["spoken"]:
                entry["spoken"].append(spoken)
                added += 1
        print(f"  {canonical:10s} <- {sorted(spokens)}")

    print(f"\nthêm {added} cách đọc cho {len(by_canonical)} viết tắt "
          f"(ngưỡng xuất hiện >= {args.min_count})")
    if args.dry_run:
        print("chế độ thử, không ghi file")
        return 0
    raw["entries"].sort(key=lambda e: e["canonical"])
    CATALOG.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    print(f"đã ghi {CATALOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
