"""Thu hồi corpus V1 (244.688 câu) về taxonomy V2 — qua HAI cổng tất định.

V1 có 23,5 triệu token văn bản quân sự thật, gấp 350 lần số span CARDINAL hiện
có. Đó là lý do V1 khôi phục số tốt hơn V2 dù kiến trúc thô sơ hơn. Nhưng nhãn
thượng nguồn của nó hỏng ở BIÊN:

    NUM      "trăm"                       span chỉ có mỗi từ chỉ hàng
    DATE     "tháng ba tháng hai mươi"    vô nghĩa
    MEASURE  "mét"                        thiếu hẳn phần số

Nên không thể nối thẳng. Hai cổng:

  1. Chạy normalizer của ta lên từng span. Trượt thì bỏ CẢ CÂU.
  2. Biên phải cực đại: từ sát hai bên span không được là từ số. Cổng này bắt
     span bị cắt cụt mà normalizer vẫn đọc trôi ("hai" cắt từ "hai ngàn").

Bỏ span thì phải bỏ cả câu — giữ lại nghĩa là gán O cho chỗ đó, dạy mô hình
điều ngược với hàng trăm câu khác.

Câu chứa CASE/TITLE/UNIT bị bỏ hẳn: taxonomy V2 không có lớp tương đương, mà
gán chúng thành O sẽ dạy mô hình đừng viết hoa tên riêng.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from ..normalizers.number import FILLERS, STRUCTURE_WORDS, UNIT_DIGITS
from ..registry import get_normalizer
from .schema import Sample, validate_sample, write_jsonl

ROOT = Path(__file__).resolve().parents[2]

# Chỉ ánh xạ cái CHẮC CHẮN tương đương. Không đoán.
LABEL_MAP = {
    "NUM": "CARDINAL", "DATE": "DATE", "COORD": "COORD", "MEASURE": "MEASURE",
    "FOREIGN_NAME": "FOREIGN_NAME", "LEGAL_DOC": "LEGAL_DOC_ID",
    "DOC": "DOCUMENT_ID", "DATE_QUARTER": "QUARTER", "RANK": "RANK",
    "WEAPON_NAME": "EQUIPMENT_NAME", "EQUIP": "EQUIPMENT_ID",
}
# CASE = cần viết hoa, TITLE = chức danh, UNIT = phiên hiệu đơn vị.
UNMAPPABLE = {"CASE", "TITLE", "UNIT"}
NUMBER_WORDS = set(UNIT_DIGITS) | STRUCTURE_WORDS | FILLERS | {"mười"}
PUNCT_OK = {"O", "COMMA", "PERIOD", "QUESTION"}


def decode_spans(tags):
    """V1 dùng nhãn phẳng (không B-/I-) nên span = dãy nhãn giống nhau liền kề."""
    out, cur, start = [], None, 0
    for i, tag in enumerate(tags):
        if tag != "O":
            if cur == tag:
                continue
            if cur is not None:
                out.append((start, i - 1, cur))
            cur, start = tag, i
        elif cur is not None:
            out.append((start, i - 1, cur))
            cur = None
    if cur is not None:
        out.append((start, len(tags) - 1, cur))
    return out


def convert(record, stats):
    tokens = record.get("tokens") or []
    tags = record.get("itn_tags") or []
    punct = record.get("punc_tags") or []
    if not tokens or len(tokens) != len(tags) or len(tokens) != len(punct):
        stats["lệch độ dài"] += 1
        return None
    present = {t for t in tags if t != "O"}
    if present & UNMAPPABLE:
        stats["có CASE/TITLE/UNIT"] += 1
        return None
    if not present:
        stats["không span nào"] += 1
        return None

    words = [w.lower() for w in tokens]
    if any(ch.isdigit() for ch in " ".join(words)):
        stats["dạng nói còn chữ số"] += 1
        return None

    spans = []
    for start, end, tag in decode_spans(tags):
        type_name = LABEL_MAP.get(tag)
        if type_name is None:
            stats[f"nhãn không ánh xạ được: {tag}"] += 1
            return None
        if start > 0 and words[start - 1] in NUMBER_WORDS:
            stats[f"biên cụt trái: {type_name}"] += 1
            return None
        if end + 1 < len(words) and words[end + 1] in NUMBER_WORDS:
            stats[f"biên cụt phải: {type_name}"] += 1
            return None
        text = " ".join(words[start:end + 1])
        normalizer = get_normalizer(type_name)
        if normalizer is None:
            stats[f"không có normalizer: {type_name}"] += 1
            return None
        result = normalizer(text, type_name)
        if not result.valid:
            stats[f"normalizer trượt: {type_name}"] += 1
            return None
        if result.normalized == text:
            stats[f"span đồng nhất: {type_name}"] += 1
            return None
        spans.append((start, end, type_name))

    punct = [p if p in PUNCT_OK else "O" for p in punct]
    spoken = " ".join(words)
    sample = Sample(
        id="v1c" + hashlib.sha1(spoken.encode()).hexdigest()[:13],
        spoken=spoken, written="", words=words, punct=punct, spans=spans,
        types=sorted({t for _, _, t in spans}),
        leak_key="|".join(sorted(" ".join(words[s:e + 1]) for s, e, _ in spans)),
        source="v1corpus", written_source="")
    errors = validate_sample(sample)
    if errors:
        stats[f"bản ghi không hợp lệ: {errors[0][:40]}"] += 1
        return None
    return sample


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets_v11_casefix/mil_state_train_v11_case_filtered.jsonl")
    parser.add_argument("--out", default="datasets_v2/v1_corpus.jsonl")
    parser.add_argument("--limit", type=int, default=0,
                        help="số câu GIỮ tối đa (0 = không giới hạn)")
    parser.add_argument("--max-words", type=int, default=60,
                        help="câu V1 dài trung bình 96 từ; cắt cho gần phân bố thật")
    args = parser.parse_args(argv)

    stats, kept, seen = Counter(), [], set()
    with (ROOT / args.input).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if len(record.get("tokens") or []) > args.max_words:
                stats["dài quá ngưỡng"] += 1
                continue
            sample = convert(record, stats)
            if sample is None:
                continue
            if sample.id in seen:
                stats["trùng câu"] += 1
                continue
            seen.add(sample.id)
            kept.append(sample)
            stats["GIỮ"] += 1
            if args.limit and len(kept) >= args.limit:
                break

    write_jsonl(ROOT / args.out, kept)
    print(f"{len(kept)} câu -> {args.out}")
    types = Counter(t for s in kept for _, _, t in s.spans)
    print("span theo kiểu:", ", ".join(f"{k} {v}" for k, v in types.most_common()))
    print("\n--- lý do loại ---")
    for why, n in stats.most_common(14):
        print(f"  {n:7}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
