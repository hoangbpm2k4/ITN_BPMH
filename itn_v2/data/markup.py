"""Phân tích cú pháp đánh dấu [[nội dung|KIỂU]] do mô hình sinh ra.

Mô hình sinh dữ liệu nói rất nhiều và không tuân thủ định dạng, nên bộ phân
tích ở đây **thu hoạch phòng thủ**: quét lấy mọi dòng hợp lệ và bỏ qua phần
văn vẻ xung quanh, thay vì đòi mô hình in ra đúng khuôn.
"""

import re
import unicodedata

MARK = re.compile(r"\[\[([^\[\]|]+)\|([A-Z_]+)\]\]")
PUNCT_MAP = {",": "COMMA", ".": "PERIOD", "?": "QUESTION"}
LEAD_JUNK = re.compile(r"^(?:[\s*\-•>`]|\d+[\.\)]|line\s*\d+\s*:|spoken\s*:|câu\s*\d+\s*:)+",
                       re.IGNORECASE)


def harvest_lines(text):
    """Lấy ra các dòng có khả năng là câu đã gán nhãn."""
    out = []
    for raw in text.splitlines():
        line = LEAD_JUNK.sub("", raw.strip()).strip().strip("`").strip()
        if not MARK.search(line):
            continue
        if line.count("[[") != line.count("]]"):
            continue
        out.append(line)
    return out


def parse_markup(line):
    """Trả (words, spans, punct) hoặc None nếu dòng không dùng được.

    words   : token ASR gốc, chữ thường
    spans   : [(start, end, TYPE)] bao gồm hai đầu
    punct   : nhãn dấu câu cho từng token
    """
    line = unicodedata.normalize("NFC", line).strip()
    words, spans, punct = [], [], []

    def add_plain(chunk):
        for token in chunk.split():
            if token in PUNCT_MAP:
                if not punct:
                    return False
                punct[-1] = PUNCT_MAP[token]
                continue
            trailing = ""
            while token and token[-1] in PUNCT_MAP:
                trailing = PUNCT_MAP[token[-1]]
                token = token[:-1]
            token = token.strip("\"'()")
            if not token:
                if trailing and punct:
                    punct[-1] = trailing
                continue
            words.append(token.lower())
            punct.append(trailing or "O")
        return True

    pos = 0
    for m in MARK.finditer(line):
        if not add_plain(line[pos:m.start()]):
            return None
        span_words = m.group(1).split()
        if not span_words:
            return None
        start = len(words)
        for w in span_words:
            words.append(w.lower())
            punct.append("O")
        spans.append((start, len(words) - 1, m.group(2)))
        pos = m.end()
    if not add_plain(line[pos:]):
        return None

    if not words or not spans:
        return None
    return words, spans, punct
