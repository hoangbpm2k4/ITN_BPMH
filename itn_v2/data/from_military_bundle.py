"""Chuyển bundle quân sự (BIO) sang định dạng Sample của itn_v2.

Bundle dùng nhãn BIO một trục (B-/I-<KIỂU>), còn mô hình của ta tách hai trục:
biên (BIOES, do CRF lo) và kiểu (softmax). Ở đây chỉ cần đổi biểu diễn span,
phần mã hoá BIOES do `encode_spans` lo khi dựng dataset.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from ..labels import SEMANTIC_TYPES
from .schema import Sample, validate_sample, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
PUNCT_OF = {",": "COMMA", ".": "PERIOD", "?": "QUESTION",
            ";": "COMMA", ":": "COMMA"}


def decode_bio(tags):
    out, cur = [], None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur:
                out.append(cur)
            cur = [i, i, tag[2:]]
        elif tag.startswith("I-") and cur:
            cur[1] = i
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return [tuple(x) for x in out]


def unpack(record):
    """v2 để tokens/bio_tags thẳng, v3 đóng gói thành chuỗi JSON."""
    tokens = record.get("tokens")
    tags = record.get("bio_tags")
    if tokens is None:
        tokens = json.loads(record["tokens_json"])
    if tags is None:
        tags = json.loads(record["bio_tags_json"])
    return tokens, tags


def convert(record):
    """Trả Sample, hoặc None kèm lý do bỏ."""
    tokens, tags = unpack(record)
    if len(tokens) != len(tags):
        return None, "lệch số token/tag"

    spans_raw = decode_bio(tags)

    # Dấu câu là token riêng trong bundle; mô hình của ta gắn dấu vào TỪ trước
    # nó, nên phải bỏ token dấu câu và dời chỉ số span theo.
    words, punct, shift = [], [], []
    for tok in tokens:
        if tok in PUNCT_OF and words:
            punct[-1] = PUNCT_OF[tok]
            shift.append(None)
        else:
            shift.append(len(words))
            # v3 còn để "MMSI"/"AIS" nguyên chữ hoa trong câu nền; ASR không
            # bao giờ xuất ra như vậy nên hạ hết về chữ thường.
            words.append(tok.lower())
            punct.append("O")

    spans = []
    for s, e, t in spans_raw:
        if t not in SEMANTIC_TYPES:
            return None, f"kiểu ngoài taxonomy: {t}"
        ns, ne = shift[s], shift[e]
        if ns is None or ne is None:
            return None, "span trùm lên token dấu câu"
        spans.append((ns, ne, t))
    spans.sort()

    spoken = " ".join(words)
    sample = Sample(
        id=hashlib.sha1((record["id"] + spoken).encode()).hexdigest()[:16],
        spoken=spoken,
        written=record["normalized_text"],
        words=words,
        punct=punct,
        spans=spans,
        types=sorted({t for _, _, t in spans}),
        leak_key="|".join(sorted(" ".join(words[s:e + 1]) for s, e, _ in spans)),
        source=f"military_v2:{record['split']}",
        written_source=record["normalized_text"],
    )
    errors = validate_sample(sample)
    if errors:
        return None, errors[0]
    return sample, record["split"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="datasets_v2")
    parser.add_argument("--prefix", default="mil_v2")
    args = parser.parse_args(argv)

    by_split, stats = {"train": [], "dev": [], "test": []}, Counter()
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            sample, why = convert(json.loads(line))
            if sample is None:
                stats[f"bỏ: {why}"] += 1
                continue
            by_split[why].append(sample)
            stats[f"giữ: {why}"] += 1

    out_dir = ROOT / args.out_dir
    for split, rows in by_split.items():
        if rows:
            write_jsonl(out_dir / f"{args.prefix}_{split}.jsonl", rows)
            print(f"  {len(rows):6} câu -> {args.prefix}_{split}.jsonl")
    print("\n--- thống kê ---")
    for k, v in stats.most_common():
        print(f"  {v:6}  {k}")
    types = Counter(t for rows in by_split.values() for s in rows for t in s.types)
    print(f"\n{len(types)} kiểu, ít nhất {min(types.values())} câu/kiểu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
