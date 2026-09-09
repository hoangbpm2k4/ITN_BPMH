"""Dựng corpus 1:1 từ data_v1 — mỗi văn bản đúng MỘT dạng nói.

JSON của data_v1 chính là bước pipeline v14 chuyển ITN (dạng viết) sang dạng
nói. Ta lấy thẳng kết quả đó làm corpus, KHÔNG dùng nó để suy ra nhãn.

Ba nguồn cặp:
  1. chunk có 1 biến thể phát âm  -> dùng luôn, hạ chữ thường        (1.500)
  2. chunk có 2 biến thể          -> chọn 1 bản sạch nhất                (77)
  3. chunk KHÔNG có biến thể      -> dạng nói = chính nó, viết thường (2.062)

Nguồn 3 vẫn là cặp hợp lệ và có giá trị: ASR trả về chữ thường không dấu câu,
nên ngay cả câu không có con số nào thì nhiệm vụ ITN vẫn còn — khôi phục viết
hoa tên riêng và đặt dấu câu.

Lọc rác của pipeline v14: nó đánh vần bậy các từ tiếng Việt thường
("qua 80" -> "Q U A tám mươi"). Lưu ý "TTXVN" -> "T T X V N" là ĐÚNG, không
phải rác — bộ lọc phải phân biệt được hai trường hợp này.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Chuỗi chữ cái đơn cách nhau ở ĐẦU vùng: "Q U A tám mươi" -> bắt "Q U A"
LEADING_LETTERS = re.compile(r"^((?:[a-zA-ZđĐ]\s+){1,}[a-zA-ZđĐ])(?:\s|$)")
# Từ thường hay đứng đầu câu. Nếu một trong số này VIẾT HOA mà đứng giữa câu và
# trước nó không có dấu câu, gần như chắc chắn ranh giới câu đã bị xoá mất khi
# pipeline v14 dán các câu lại với nhau.
SENTENCE_STARTERS = re.compile(
    r"^(?:Chưa|Trong|Ông|Bà|Câu|Ít|Sau|Trước|Khi|Nhưng|Và|Với|Từ|Để|Trên|Dưới|"
    r"Nay|Hôm|Vì|Do|Nếu|Tuy|Mặc|Theo|Đây|Đó|Cả|Một|Hai|Ba|Những|Các|Người|"
    r"Nhiều|Cũng|Thế|Sự|Việc|Ngày|Năm|Tháng|Hiện|Đến|Tại|Về|Bởi|Song|Rồi)$")
SENTENCE_END_CHARS = (",", ".", "?", "!", ":", ";")

# Từ tiếng Việt thường (không viết tắt): toàn chữ thường, có thể có dấu
VN_LOWER_WORD = re.compile(r"^[a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếề"
                           r"ểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]{2,}$")


# Pipeline v14 chèn "linh"/"lẻ" sai chỗ: "1968" -> "một nghìn chín trăm LINH sáu
# mươi tám". Người Việt không đọc thế — "linh"/"lẻ" chỉ đứng trước HÀNG ĐƠN VỊ
# ("một trăm linh hai" = 102). Dính 425/2587 câu (16%).
# Parser của ta đọc đúng cả hai cách nên pipeline không hỏng, nhưng ASR thật sẽ
# KHÔNG BAO GIỜ phát ra "linh sáu mươi" — train trên đó là dạy một mẫu không
# tồn tại trong phân bố triển khai.
SPURIOUS_LINH = re.compile(
    r"\b(?:linh|lẻ)\s+((?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín)\s+(?:mươi|mười))\b")


def fix_spurious_linh(text):
    return SPURIOUS_LINH.sub(r"\1", text)


def norm(text):
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def diff_regions(written, spoken):
    """Vùng khác nhau giữa hai chuỗi, theo từ."""
    w, s = written.split(), spoken.split()
    matcher = SequenceMatcher(None, [x.lower() for x in w], [x.lower() for x in s])
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        out.append((" ".join(w[i1:i2]), " ".join(s[j1:j2]), (j1, j2)))
    return out


def garbage_score(written, spoken):
    """Đếm vùng mà một TỪ THƯỜNG bị đánh vần ra chữ cái — lỗi của v14.

    Phân biệt hai trường hợp trông giống nhau:
        "TTXVN"  -> "T T X V N"        ĐÚNG, viết tắt đọc từng chữ cái
        "qua 80" -> "Q U A tám mươi"   RÁC,  từ thường bị đánh vần

    Cách phân biệt: ghép các chữ cái đứng đầu vùng lại; nếu chuỗi ghép trùng
    một TỪ THƯỜNG viết thường ở dạng viết thì đó là rác.
    """
    score = 0
    for w_chunk, s_chunk, _ in diff_regions(written, spoken):
        match = LEADING_LETTERS.match(s_chunk.strip())
        if not match:
            continue
        joined = match.group(1).replace(" ", "").lower()
        for word in w_chunk.split():
            if VN_LOWER_WORD.match(word) and word.lower() == joined:
                score += 1
                break
    return score


# Bộ tách câu thượng nguồn đôi khi cắt ngay giữa một con số ("5.000 tấn" ->
# câu mới bắt đầu bằng "000 tấn"), và bộ cào trang đôi khi nuốt cả mã
# JavaScript của trang báo vào phần văn bản.
FRAGMENT_HEAD = re.compile(r"^\d{3}\b")
WEB_CODE = re.compile(
    r"\b(?:function|var|window|setTimeout|addClass|removeClass|url|href|"
    r"isLoading|pagethree|newslist|p\.|\.js)\b")


def is_fragment(written):
    """Câu không đứng được một mình: bị cắt giữa số, hoặc là mã trang web."""
    if FRAGMENT_HEAD.match(written):
        return True
    if written[:1].islower():
        return True
    return len(WEB_CODE.findall(written)) >= 2


def lost_boundary_count(text):
    """Đếm chỗ nghi bị mất ranh giới câu.

    Chunk mất ranh giới dạy mô hình SAI về dấu câu: nhãn vàng nói "không có dấu
    chấm" ở đúng chỗ phải có. Tệ hơn là không có dữ liệu. Đo được: nhóm chunk
    này chỉ có 2,05 dấu chấm/100 từ so với 3,74 ở phần còn lại.
    """
    tokens = text.split()
    count = 0
    for i in range(1, len(tokens)):
        if tokens[i - 1].endswith(SENTENCE_END_CHARS):
            continue
        if SENTENCE_STARTERS.match(tokens[i]):
            count += 1
    return count


def pick_variant(chunk):
    """Chọn đúng MỘT dạng nói cho một chunk. Trả (spoken, nguồn) hoặc None."""
    written = norm(chunk.get("text", ""))
    if not written:
        return None
    variants = [fix_spurious_linh(norm(r.get("text", "")))
                for r in chunk.get("all_text_results", [])
                if r.get("type") == "pronunciation_variant"]
    variants = [v for v in variants if v and v != written]
    if not variants:
        return written.lower(), "không_biến_thể"
    if len(variants) == 1:
        # Hợp đồng đầu vào là transcript ASR: không có casing. Trước
        # đây nhánh có biến thể giữ casing của bài báo; API vì thế không
        # gán span cho "Norng Chan Phal", sau đó Sample.words lại bị hạ thành
        # "norng chan phal". Kết quả là nhãn O giả ngay trên tên riêng.
        return variants[0].lower(), "một_biến_thể"
    best = min(variants, key=lambda v: (garbage_score(written, v), len(v)))
    return best.lower(), "chọn_từ_nhiều"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_v1/json")
    parser.add_argument("--out", default="datasets_v2/v1_pairs.jsonl")
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--max-garbage", type=int, default=0,
                        help="số vùng đánh vần bậy tối đa còn chấp nhận")
    args = parser.parse_args(argv)

    stats, seen, pairs = Counter(), set(), []
    for path in sorted((ROOT / args.input).rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            stats["JSON hỏng"] += 1
            continue
        topic = doc.get("article", {}).get("topic", "?")
        for chunk in doc.get("chunks", []):
            written = norm(chunk.get("text", ""))
            n_words = len(written.split())
            if not (args.min_words <= n_words <= args.max_words):
                stats["độ dài ngoài khoảng"] += 1
                continue
            issues = set(chunk.get("validation_issues") or [])
            if "over_max_words_sentence_kept" in issues:
                stats["cờ over_max: nhiều câu bị dán, mất dấu"] += 1
                continue
            if lost_boundary_count(written) > 0:
                stats["nghi mất ranh giới câu"] += 1
                continue
            if is_fragment(written):
                stats["mảnh vụn: cắt giữa số hoặc mã trang web"] += 1
                continue
            picked = pick_variant(chunk)
            if picked is None:
                stats["không dựng được cặp"] += 1
                continue
            spoken, kind = picked
            if any(ch.isdigit() for ch in spoken):
                stats["dạng nói còn chữ số"] += 1
                continue
            bad = garbage_score(written, spoken)
            if bad > args.max_garbage:
                stats[f"đánh vần bậy ({kind})"] += 1
                continue
            key = hashlib.sha1(written.encode()).hexdigest()
            if key in seen:
                stats["trùng lặp"] += 1
                continue
            seen.add(key)
            stats[f"giữ: {kind}"] += 1
            pairs.append({"id": key[:16], "written": written, "spoken": spoken,
                          "topic": topic, "kind": kind})

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"{len(pairs)} cặp 1:1 -> {out_path}")
    n_diff = sum(1 for p in pairs if p["kind"] != "không_biến_thể")
    print(f"  có thay đổi cách đọc: {n_diff} · chỉ khác hoa-thường/dấu câu: "
          f"{len(pairs) - n_diff}")
    print("\n--- thống kê ---")
    for why, n in stats.most_common():
        print(f"  {n:5d}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
