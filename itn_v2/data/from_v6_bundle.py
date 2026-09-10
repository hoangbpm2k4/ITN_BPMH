"""Chuyển mẻ v6 (reading-variant) về hợp đồng dữ liệu V2 — qua ba cổng.

Mẻ v6 dùng lược đồ riêng và mang ba khuyết tật chặn, đo trên 4.341 câu:

  1. `spoken_text` CHỨA DẤU CÂU ở 4341/4341 câu, và ở 51,3% số câu dấu câu đó
     trùng khít đáp án. Đầu vào thật của ta là văn bản ASR không dấu câu, mà một
     trong ba đầu ra của mô hình lại chính là dấu câu — huấn luyện như vậy là
     dạy mô hình chép đáp án từ đầu vào.
  2. `normalized_text` KHÔNG VIẾT HOA đầu câu ở 4341/4341 câu, ngược với bản gốc
     thật (viết hoa là 6,3% khoảng cách hiện tại).
  3. Toạ độ phút thập phân dùng dấu PHẨY một chữ số (21°42,9′N) trong khi bản
     gốc dùng dấu CHẤM ba chữ số (10°25.500′N).

Khuyết tật 1 và 2 sửa được tất định ở đây. Khuyết tật 3 không sửa được — nó là
một khuôn khác hẳn, và cổng tái dựng sẽ tự loại những câu đó.

Ba cổng, trượt cổng nào thì bỏ CẢ CÂU (giữ lại nghĩa là dạy nhãn O ở đúng chỗ
lẽ ra phải có span):

  1. Hình thức: độ dài khớp, nhãn thuộc taxonomy, span không chồng lấn.
  2. Biên cực đại: từ sát hai bên span không được là từ chỉ số.
  3. Tái dựng: chạy pipeline ở chế độ oracle với span vàng phải cho ra ĐÚNG
     `normalized_text` (sau khi khôi phục viết hoa đầu câu). Cổng này bắt mọi
     lệch khuôn giữa hai bên mà không cần liệt kê trước từng loại.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..labels import SEMANTIC_TYPES
from ..normalizers.number import FILLERS, STRUCTURE_WORDS, UNIT_DIGITS
from ..pipeline import normalize_utterance
from .schema import Sample, validate_sample, write_jsonl

# Dấu câu trong `tokens` được bóc ra khỏi chuỗi từ và chuyển thành NHÃN gắn vào
# từ đứng trước. Dấu chấm phẩy gộp vào COMMA vì taxonomy chỉ có bốn nhãn.
PUNCT_TO_LABEL = {",": "COMMA", ";": "COMMA", ".": "PERIOD",
                  "!": "PERIOD", "?": "QUESTION"}
SENTENCE_END = {"PERIOD", "QUESTION"}
NUMBER_WORDS = set(UNIT_DIGITS) | STRUCTURE_WORDS | FILLERS | {"mười"}


def strip_punctuation(tokens, bio_tags):
    """Bóc token dấu câu thành nhãn, trả (words, punct, bio đã dồn chỉ số).

    Dấu câu đứng đầu câu hoặc đứng liền nhau thì bị bỏ hẳn: không có từ nào
    trước nó để gắn nhãn.
    """
    words, punct, tags = [], [], []
    for token, tag in zip(tokens, bio_tags):
        label = PUNCT_TO_LABEL.get(token)
        if label is not None:
            if words:
                punct[-1] = label
            continue
        words.append(token.lower())
        punct.append("O")
        tags.append(tag)
    return words, punct, tags


def decode_bio(tags):
    """B-/I- -> danh sách (đầu, cuối, KIỂU), cuối là chỉ số BAO GỒM."""
    spans, start, current = [], 0, None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current is not None:
                spans.append((start, i - 1, current))
            start, current = i, tag[2:]
        elif tag.startswith("I-"):
            if current is None:          # I- không có B- đứng trước
                return None
            if tag[2:] != current:
                return None
        else:
            if current is not None:
                spans.append((start, i - 1, current))
            current = None
    if current is not None:
        spans.append((start, len(tags) - 1, current))
    return spans


def restore_sentence_case(text, reference):
    """Khôi phục viết hoa đầu câu cho `normalized_text` vốn toàn chữ thường.

    Chỉ nâng chữ cái ĐẦU của câu; mọi chữ hoa bên trong (Su-22, IPv6) đã đúng
    sẵn trong bản v6 nên không đụng tới.

    Dấu chấm chỉ kết thúc câu khi SAU nó là khoảng trắng hoặc hết chuỗi. Không
    có điều kiện này thì dấu chấm bên trong 188.0.104.23, 1.253.616 và
    74G-997.36 cũng bị coi là hết câu, và từ kế tiếp bị viết hoa oan.
    """
    out, at_start = [], True
    for i, ch in enumerate(text):
        if at_start and ch.isalpha():
            out.append(ch.upper())
            at_start = False
            continue
        out.append(ch)
        if ch in ".?!" and (i + 1 >= len(text) or text[i + 1].isspace()):
            at_start = True
    return "".join(out)


def convert(record, stats, config=None):
    tokens = record.get("tokens") or []
    bio = record.get("bio_tags") or []
    target = (record.get("normalized_text") or "").strip()
    if not tokens or len(tokens) != len(bio) or not target:
        stats["lệch độ dài / thiếu đích"] += 1
        return None

    words, punct, tags = strip_punctuation(tokens, bio)
    if not words:
        stats["rỗng sau khi bóc dấu câu"] += 1
        return None
    if any(ch.isdigit() for w in words for ch in w):
        stats["dạng nói còn chữ số"] += 1
        return None

    spans = decode_bio(tags)
    if spans is None:
        stats["chuỗi BIO không hợp lệ"] += 1
        return None
    if not spans:
        stats["không span nào"] += 1
        return None

    for start, end, type_name in spans:
        if type_name not in SEMANTIC_TYPES:
            stats[f"kiểu ngoài taxonomy: {type_name}"] += 1
            return None
        if start > 0 and words[start - 1] in NUMBER_WORDS:
            stats[f"biên cụt trái: {type_name}"] += 1
            return None
        if end + 1 < len(words) and words[end + 1] in NUMBER_WORDS:
            stats[f"biên cụt phải: {type_name}"] += 1
            return None

    boundaries = ["O"] * len(words)
    for start, end, _ in spans:
        if start == end:
            boundaries[start] = "S"
        else:
            boundaries[start] = "B"
            boundaries[end] = "E"
            for i in range(start + 1, end):
                boundaries[i] = "I"

    try:
        rendered, outputs = normalize_utterance(
            words, boundaries, [t for _, _, t in spans], punct, config=config)
    except Exception as exc:
        stats[f"oracle lỗi: {type(exc).__name__}"] += 1
        return None

    failed = [o for o in outputs if not o.emitted]
    if failed:
        stats[f"normalizer trượt: {failed[0].predicted_type}"] += 1
        return None

    expected = restore_sentence_case(target, rendered)
    if rendered != expected:
        stats[f"tái dựng lệch: {spans[0][2]}"] += 1
        return None

    spoken = " ".join(words)
    sample = Sample(
        id="v6" + hashlib.sha1(spoken.encode()).hexdigest()[:14],
        spoken=spoken, written=rendered, words=words, punct=punct,
        spans=[tuple(s) for s in spans],
        types=[t for _, _, t in spans],
        leak_key="|".join(sorted(o.normalized or "" for o in outputs)),
        source="v6:" + str(record.get("priority_group", "")),
        written_source=target)
    errors = validate_sample(sample)
    if errors:
        stats[f"bản ghi không hợp lệ: {errors[0][:40]}"] += 1
        return None
    return sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="v6_variant_train.jsonl của mẻ v6")
    parser.add_argument("--out", required=True)
    parser.add_argument("--leak-against", default="datasets_v2/dv1_eval.jsonl",
                        help="bộ phải chống rò rỉ (mặc định: TEST CUỐI)")
    args = parser.parse_args(argv)

    forbidden = set()
    leak_path = Path(args.leak_against)
    if leak_path.exists():
        with leak_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    forbidden.add(json.loads(line)["spoken"].strip().lower())

    stats, kept = Counter(), []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            stats["đọc vào"] += 1
            sample = convert(json.loads(line), stats)
            if sample is None:
                continue
            if sample.spoken in forbidden:
                stats["trùng bộ chống rò rỉ"] += 1
                continue
            kept.append(sample)
            stats["giữ lại"] += 1

    write_jsonl(args.out, kept)
    print(f"đọc {stats['đọc vào']} · giữ {stats['giữ lại']} "
          f"({stats['giữ lại'] / max(stats['đọc vào'], 1):.1%}) -> {args.out}")
    print("\nlý do loại:")
    for reason, n in stats.most_common():
        if reason not in {"đọc vào", "giữ lại"}:
            print(f"  {n:6d}  {reason}")
    by_type = Counter(t for s in kept for _, _, t in s.spans)
    print("\nspan giữ lại theo kiểu:")
    for t, n in by_type.most_common():
        print(f"  {t:16s} {n:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
