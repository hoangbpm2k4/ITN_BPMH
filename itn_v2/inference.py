"""Đường chạy hoàn chỉnh của nửa tất định (Phase 1 + 2).

Nhận văn bản ASR thô + chú thích span trên token GỐC, trả văn bản dạng viết
kèm vết gỡ lỗi. Khi nhánh mô hình (Phase 3) sẵn sàng, chỉ cần thay nguồn của
``boundaries``/``types``; phần còn lại giữ nguyên.
"""

from .config import Config
from .labels import encode_spans
from .pipeline import normalize_utterance
from .segmentation_alignment import WordSegmenter, project_raw_spans_to_model

_SEGMENTER = None


def get_segmenter():
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = WordSegmenter()
    return _SEGMENTER


def normalize_with_gold_spans(raw_text, annotations, punct_labels=None, config=None):
    """``annotations``: [(raw_start, raw_end, TYPE)] theo chỉ số token ASR gốc.

    Trả (văn bản, danh sách SpanOutput, alignment, xung đột tách từ).
    """
    config = config or Config()
    alignment = get_segmenter().segment(raw_text)
    raw_spans = [(a[0], a[1]) for a in annotations]
    types = [a[2] for a in annotations]
    model_spans, conflicts = project_raw_spans_to_model(alignment, raw_spans)

    n_words = len(alignment.model_words)
    boundaries = encode_spans(model_spans, n_words)

    model_punct = ["O"] * n_words
    if punct_labels:
        # nhãn dấu câu cho trên token gốc -> dồn về từ chứa token cuối
        for widx, (s, e) in enumerate(alignment.spans):
            for r in range(s, e + 1):
                if punct_labels[r] != "O":
                    model_punct[widx] = punct_labels[r]

    text, outputs = normalize_utterance(
        alignment.model_words, boundaries, types, model_punct,
        config=config, alignment=alignment)
    return text, outputs, alignment, conflicts
