"""Chuyển dữ liệu thật ở data_v1/ sang định dạng gán nhãn V2.

data_v1 là văn bản nghiệp vụ THẬT (báo Quân đội / Hàng hải) kèm biến thể phát
âm do một pipeline Gemma v14 sinh ra. Mỗi chunk có cặp:

    text  : dạng viết  ("Huy hiệu 55 năm tuổi Đảng")
    variant: dạng nói  ("Huy hiệu năm mươi lăm năm tuổi Đảng")

**Không dùng nhãn của pipeline thượng nguồn.** Trường ``case`` của nó rất nhiễu:
``EQUIPMENT_CODE`` phần lớn là từ tiếng Việt thường bị đánh vần bậy ("qua 80" ->
"Q U A tám mươi"), và 542/3633 mục không có nhãn, lẫn cả rác do cắt chunk sai.

Thay vào đó: căn hai chuỗi để tìm vùng khác nhau, đoán kiểu từ HÌNH DẠNG của
dạng viết, rồi **xác thực bằng chính bộ chuẩn hoá** — chỉ giữ nhãn nào mà
normalizer chạy trên dạng nói tái tạo đúng dạng viết. Nhãn sai không thể lọt.
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

from ..inference import normalize_with_gold_spans
from ..registry import get_normalizer
from ..validators import validate
from .schema import Sample, validate_sample, write_jsonl

ROOT = Path(__file__).resolve().parents[2]

PUNCT_MAP = {",": "COMMA", ".": "PERIOD", "?": "QUESTION", "!": "PERIOD",
             ";": "COMMA", ":": "COMMA"}
STRIP_CHARS = ''.join(PUNCT_MAP) + '"\'()[]{}…'

# Ứng viên kiểu suy từ hình dạng DẠNG VIẾT. Dạng viết là chân lý, hình dạng của
# nó cho biết lớp; thứ tự trong mỗi nhánh là thứ tự thử.
SHAPE_RULES = [
    (re.compile(r"^-?\d{1,3}(\.\d{3})+$"), ["CARDINAL"]),
    (re.compile(r"^\d+,\d+%$"), ["PERCENT"]),
    (re.compile(r"^\d+%$"), ["PERCENT"]),
    (re.compile(r"^-?\d+,\d+$"), ["DECIMAL"]),
    (re.compile(r"^\d+/\d+$"), ["DATE", "FRACTION"]),
    (re.compile(r"^\d+/\d+/\d+$"), ["DATE"]),
    (re.compile(r"^\d+:\d+$"), ["TIME", "RATIO"]),
    (re.compile(r"^\d+°.*$"), ["COORD", "HEADING", "BEARING"]),
    (re.compile(r"^\d+\.\d+(\.\d+)+$"), ["VERSION"]),
    (re.compile(r"^0\d{8,10}$"), ["TELEPHONE", "DIGIT_SEQ"]),
    (re.compile(r"^\d{9}$"), ["MMSI_ID", "DIGIT_SEQ"]),
    (re.compile(r"^0\d+$"), ["DIGIT_SEQ"]),
    (re.compile(r"^\d+\s*[A-Za-zµ°²³/%]+$"), ["MEASURE", "SPEED", "DISTANCE",
                                              "DEPTH", "DRAFT", "FREQUENCY"]),
    (re.compile(r"^thứ \d+$", re.I), ["ORDINAL"]),
    (re.compile(r"^Quý [IVX]+$", re.I), ["QUARTER"]),
    (re.compile(r"^[A-ZĐ][a-zA-Z]*-\d+.*$"), ["EQUIPMENT_ID"]),
    (re.compile(r"^[A-ZĐ]{2,}$"), ["ACRONYM"]),
    (re.compile(r"^\d+$"), ["CARDINAL", "DIGIT_SEQ"]),
]

# Năm bốn chữ số phải thử DATE TRƯỚC: CardinalParser chèn dấu phân nhóm nên
# "một nghìn chín trăm linh sáu mươi tám" ra "1.968" chứ không phải "1968".
YEAR_RE = re.compile(r"^\d{4}$")
FALLBACK_TYPES = ["CARDINAL", "DIGIT_SEQ", "MEASURE", "DATE", "ACRONYM"]


def candidate_types(written):
    if YEAR_RE.match(written) and 1000 <= int(written) <= 2999:
        return ["DATE", "CARDINAL", "DIGIT_SEQ"]
    for pattern, types in SHAPE_RULES:
        if pattern.match(written):
            return types
    return FALLBACK_TYPES


def split_tokens(text):
    """Tách token, bóc dấu câu ra thành nhãn riêng. Trả (từ, nhãn dấu câu)."""
    words, puncts = [], []
    for raw in unicodedata.normalize("NFC", text).split():
        tail = "O"
        token = raw
        while token and token[-1] in STRIP_CHARS:
            if token[-1] in PUNCT_MAP:
                tail = PUNCT_MAP[token[-1]]
            token = token[:-1]
        token = token.strip(STRIP_CHARS)
        if not token:
            if words:
                puncts[-1] = tail if tail != "O" else puncts[-1]
            continue
        words.append(token)
        puncts.append(tail)
    return words, puncts


def label_span(spoken_words, written_text):
    """Thử từng kiểu ứng viên; giữ kiểu nào tái tạo ĐÚNG dạng viết."""
    target = written_text.strip().strip(STRIP_CHARS)
    if not target:
        return None
    spoken = " ".join(spoken_words)
    for type_name in candidate_types(target):
        normalizer = get_normalizer(type_name)
        if normalizer is None:
            continue
        result = normalizer(spoken, type_name)
        if not result.valid or result.normalized != target:
            continue
        ok, _ = validate(type_name, result.normalized)
        if ok:
            return type_name
    return None


def chunk_to_sample(chunk, variant_text, source):
    written_words, _ = split_tokens(chunk["text"])
    spoken_words, spoken_punct = split_tokens(variant_text)
    if not spoken_words or not written_words:
        return None, "rỗng"

    matcher = SequenceMatcher(
        None, [w.lower() for w in written_words], [w.lower() for w in spoken_words])
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        written_chunk = " ".join(written_words[i1:i2])
        type_name = label_span(spoken_words[j1:j2], written_chunk)
        if type_name:
            spans.append((j1, j2 - 1, type_name))
    if not spans:
        return None, "không nhãn nào được xác thực"

    lowered = [w.lower() for w in spoken_words]
    spoken_text = " ".join(lowered)
    try:
        written, outputs, _al, conflicts = normalize_with_gold_spans(
            spoken_text, spans, spoken_punct)
    except Exception as exc:
        return None, f"pipeline lỗi: {type(exc).__name__}"
    if conflicts:
        return None, "từ ghép cắt ngang span"
    if not all(o.emitted for o in outputs):
        bad = next(o for o in outputs if not o.emitted)
        return None, f"{bad.predicted_type}: {bad.reason[:50]}"

    sample = Sample(
        id=hashlib.sha1(spoken_text.encode()).hexdigest()[:16],
        spoken=spoken_text, written=written, words=lowered,
        spans=spans, punct=spoken_punct,
        types=sorted({t for _, _, t in spans}),
        leak_key="|".join(sorted(o.normalized for o in outputs)),
        source=source,
        written_source=" ".join(chunk["text"].split()),
    )
    errors = validate_sample(sample)
    if errors:
        return None, "schema: " + errors[0]
    return sample, None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_v1/json")
    parser.add_argument("--out", default="datasets_v2/real_from_v1.jsonl")
    args = parser.parse_args(argv)

    files = sorted((ROOT / args.input).rglob("*.json"))
    print(f"{len(files)} file JSON")
    kept, reasons, types = {}, Counter(), Counter()
    n_chunk = n_variant = 0

    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            reasons["JSON hỏng"] += 1
            continue
        topic = doc.get("article", {}).get("topic", "?")
        for chunk in doc.get("chunks", []):
            n_chunk += 1
            for result in chunk.get("all_text_results", []):
                if result.get("type") != "pronunciation_variant":
                    continue
                n_variant += 1
                sample, why = chunk_to_sample(
                    chunk, result["text"], f"data_v1:{topic}")
                if sample is None:
                    reasons[why] += 1
                    continue
                kept.setdefault(sample.id, sample)
                types.update(sample.types)

    samples = list(kept.values())
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, samples)

    print(f"{n_chunk} chunk, {n_variant} biến thể phát âm")
    print(f"-> giữ {len(samples)} câu vào {out_path}\n")
    print("--- kiểu được xác thực ---")
    for t, n in types.most_common():
        print(f"  {t:16s} {n}")
    print("\n--- lý do loại (10 đầu) ---")
    for why, n in reasons.most_common(10):
        print(f"  {n:5d}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
