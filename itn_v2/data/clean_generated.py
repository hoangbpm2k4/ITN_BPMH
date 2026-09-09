"""Lọc chất lượng cho dữ liệu sinh máy.

Pipeline tất định đã bảo đảm mọi span *chuẩn hoá được*. Nhưng nó không nói gì
về việc câu có phải câu thật hay không. Ba loại rác lọt qua được:

  1. ký tự thừa của markdown còn dính lại  ("xtê đi át si gâu ` - ok")
  2. câu chỉ gồm đúng một span, không có ngữ cảnh -> không dạy mô hình được gì
  3. câu quá ngắn

Chạy:
    python -m itn_v2.data.clean_generated --input datasets_v2/seed_generated.jsonl
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from .schema import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]

from ..labels import SEMANTIC_TYPES

# Tên kiểu bị lọt vào chính câu văn ("cardinal biên phòng đã triển khai...") —
# mô hình copy nhãn từ prompt vào nội dung. Bộ lọc cũ không bắt được vì các từ
# này toàn chữ cái ASCII thường nên vẫn khớp khuôn từ tiếng Việt.
TYPE_NAME_TOKENS = {t.lower() for t in SEMANTIC_TYPES}
TYPE_NAME_TOKENS |= {part for t in TYPE_NAME_TOKENS for part in t.split("_")}

JUNK_TOKENS = {"`", "-", "*", ">", "|", "ok", "wait", "note", "line", "sentence",
               "spoken", "output", "example", "câu", "```"} | TYPE_NAME_TOKENS
# Từ tiếng Việt hợp lệ ở dạng nói: chỉ chữ cái (có dấu), không ký tự lạ.
VN_WORD = re.compile(r"^[a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếề"
                     r"ểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+$")
# Span ELECTRONIC được phép chứa ký tự ASCII của địa chỉ.
FREEFORM_TYPES = {"ELECTRONIC"}

MIN_WORDS = 5
MIN_CONTEXT_WORDS = 2


def check(sample):
    """Trả lý do loại, hoặc None nếu câu dùng được."""
    if len(sample.words) < MIN_WORDS:
        return f"quá ngắn ({len(sample.words)} từ)"

    in_span = set()
    freeform = set()
    for start, end, type_name in sample.spans:
        for i in range(start, end + 1):
            in_span.add(i)
            if type_name in FREEFORM_TYPES:
                freeform.add(i)

    context = len(sample.words) - len(in_span)
    if context < MIN_CONTEXT_WORDS:
        return f"thiếu ngữ cảnh ({context} từ ngoài span)"

    for i, word in enumerate(sample.words):
        if word in JUNK_TOKENS and i not in freeform:
            return f"token rác {word!r}"
        if i in freeform:
            continue
        if not VN_WORD.match(word):
            return f"từ có ký tự lạ {word!r}"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets_v2/seed_generated.jsonl")
    parser.add_argument("--output", default="datasets_v2/seed_clean.jsonl")
    args = parser.parse_args(argv)

    samples = list(read_jsonl(ROOT / args.input))
    kept, reasons = [], Counter()
    for s in samples:
        why = check(s)
        if why:
            reasons[why.split("(")[0].strip()] += 1
        else:
            kept.append(s)

    write_jsonl(ROOT / args.output, kept)
    print(f"vào {len(samples)} câu -> giữ {len(kept)} ({len(kept)/max(1,len(samples)):.0%})")
    for why, n in reasons.most_common(10):
        print(f"  loại {n:4d}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
