"""Kiểm tra oracle (spec §29).

Cấp nhãn ranh giới và kiểu VÀNG cho pipeline tất định. Với các lớp tất định,
độ chính xác oracle phải xấp xỉ 100%. Chưa đạt thì huấn luyện mô hình chưa có
ý nghĩa, vì mô hình dự đoán đúng vẫn cho ra kết quả sai.
"""

from .labels import encode_spans
from .pipeline import normalize_utterance


def run_oracle(words, gold_spans, gold_types, punct_labels=None, config=None):
    """gold_spans: [(start, end)] · gold_types: kiểu tương ứng từng span."""
    boundaries = encode_spans(gold_spans, len(words))
    return normalize_utterance(words, boundaries, gold_types, punct_labels, config=config)


def oracle_from_annotation(text, annotations, punct_labels=None, config=None):
    """``annotations``: [(start, end, type)] theo chỉ số từ trong ``text``."""
    words = text.split()
    spans = [(a[0], a[1]) for a in annotations]
    types = [a[2] for a in annotations]
    return run_oracle(words, spans, types, punct_labels, config=config)
