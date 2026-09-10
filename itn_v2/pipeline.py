"""Ghép span -> chuẩn hoá -> kiểm tra -> kết xuất văn bản (spec §16, §17, §18, §28).

Ba nguyên tắc bị V1 vi phạm và được cưỡng chế ở đây:
  1. Dấu câu KHÔNG bao giờ quyết định ranh giới span ITN.
  2. Span đã chuẩn hoá được BẢO VỆ: không luật dọn nào sửa được nội dung nó.
  3. Không đạt ngưỡng / parser trượt / validator trượt -> giữ nguyên dạng nói.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .labels import decode_spans
from .registry import get_normalizer
from .validators import has_validator, validate

SENTENCE_END = {".", "?", "!"}
PUNCT_CHAR = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}

# Kiểu anh em: cùng HÌNH DẠNG dạng nói nên mô hình hay lẫn, khác nhau ở khuôn
# viết. Khi validator của kiểu dự đoán bác bỏ, thử lần lượt các kiểu này.
#
# Đo được: 19 số điện thoại của bộ test bị mất vì mô hình gán MMSI_ID cho
# "không chín không tám một hai ba bốn năm sáu". Validator bác đúng (MMSI phải
# 9 chữ số) rồi pipeline giữ dạng nói — mất trắng, dù TelephoneParser đọc được.
# TELEPHONE có 369 span huấn luyện, MMSI_ID có 381, cả hai đều là chuỗi chữ số
# trần nên không có gì phân biệt ngoài ngữ cảnh.
#
# Điều kiện an toàn: CHỈ lùi sang kiểu CÓ validator riêng. Kiểu không validator
# sẽ nhận mọi thứ, biến cổng dự phòng thành đường bịa giá trị — đúng thứ spec
# §16 cấm. Vì vậy DIGIT_SEQ, VESSEL_ID, CALLSIGN không nằm trong bảng này.
FALLBACK_TYPES = {
    "MMSI_ID": ("TELEPHONE", "IMO_ID"),
    "TELEPHONE": ("MMSI_ID", "IMO_ID"),
    "IMO_ID": ("MMSI_ID", "TELEPHONE"),
}


@dataclass
class SpanOutput:
    """Vết gỡ lỗi có cấu trúc cho MỘT span (spec §28) — bắt buộc để quy trách lỗi."""
    raw_span: str
    raw_token_start: int
    raw_token_end: int
    model_word_start: int
    model_word_end: int
    predicted_boundaries: List[str]
    predicted_type: str
    boundary_confidence: float
    type_confidence: float
    model_confidence: float
    threshold: float
    normalized: Optional[str]
    valid: bool
    normalizer: str
    reason: str = ""
    emitted: bool = False
    resolved_type: str = ""      # kiểu thực sự dùng, khác predicted_type nếu đã lùi

    def as_dict(self):
        return {
            "raw_span": self.raw_span,
            "raw_token_start": self.raw_token_start,
            "raw_token_end": self.raw_token_end,
            "model_word_start": self.model_word_start,
            "model_word_end": self.model_word_end,
            "predicted_boundaries": self.predicted_boundaries,
            "predicted_type": self.predicted_type,
            "boundary_confidence": round(self.boundary_confidence, 4),
            "type_confidence": round(self.type_confidence, 4),
            "model_confidence": round(self.model_confidence, 4),
            "threshold": self.threshold,
            "normalized": self.normalized,
            "valid": self.valid,
            "normalizer": self.normalizer,
            "reason": self.reason,
            "emitted": self.emitted,
            "resolved_type": self.resolved_type or self.predicted_type,
        }


def process_span(raw_words, type_name, start, end, boundaries,
                 boundary_confidence=1.0, type_confidence=1.0,
                 raw_start=None, raw_end=None, config=None):
    """Chạy một span qua normalizer + validator + cổng tin cậy."""
    config = config or Config()
    raw_span = " ".join(raw_words)
    threshold = config.threshold_for(type_name)
    model_confidence = min(boundary_confidence, type_confidence)

    out = SpanOutput(
        raw_span=raw_span,
        raw_token_start=raw_start if raw_start is not None else start,
        raw_token_end=raw_end if raw_end is not None else end,
        model_word_start=start, model_word_end=end,
        predicted_boundaries=boundaries, predicted_type=type_name,
        boundary_confidence=boundary_confidence, type_confidence=type_confidence,
        model_confidence=model_confidence, threshold=threshold,
        normalized=None, valid=False, normalizer="",
    )

    attempts = [type_name] + [
        t for t in FALLBACK_TYPES.get(type_name, ()) if has_validator(t)]
    first_reason = ""
    resolved = None
    for k, candidate in enumerate(attempts):
        normalizer = get_normalizer(candidate)
        if normalizer is None:
            reason = f"không có normalizer cho kiểu {candidate!r}"
        else:
            if k == 0:
                out.normalizer = normalizer.name
            result = normalizer(raw_span, candidate)
            if not result.valid:
                reason = result.reason
            else:
                ok, why = validate(candidate, result.normalized)
                if ok:
                    resolved = (candidate, normalizer.name, result.normalized)
                    break
                if k == 0:
                    out.normalized = result.normalized
                reason = f"validator: {why}"
        if k == 0:
            first_reason = reason

    if resolved is None:
        out.reason = first_reason
        return out

    candidate, normalizer_name, normalized = resolved
    out.normalizer = normalizer_name
    out.resolved_type = candidate
    if candidate != type_name:
        # Giữ nguyên predicted_type để vết gỡ lỗi vẫn quy được trách nhiệm cho
        # mô hình; ghi rõ đã lùi sang kiểu nào và vì sao.
        out.reason = f"lùi {type_name} -> {candidate} ({first_reason})"

    out.normalized = normalized
    out.valid = True
    if model_confidence < threshold:
        out.reason = f"tin cậy {model_confidence:.2f} < ngưỡng {threshold:.2f}"
        return out

    out.emitted = True
    return out


def render(words, span_outputs, punct_labels=None):
    """Ghép văn bản cuối. Span đã phát ra được đánh dấu bảo vệ."""
    n = len(words)
    punct_labels = punct_labels or ["O"] * n
    covered = {}
    for so in span_outputs:
        for i in range(so.model_word_start, so.model_word_end + 1):
            covered[i] = so

    pieces = []          # (text, protected)
    i = 0
    while i < n:
        so = covered.get(i)
        if so is not None and so.model_word_start == i:
            text = so.normalized if so.emitted else " ".join(words[so.model_word_start:so.model_word_end + 1])
            pieces.append([text.replace("_", " "), so.emitted])
            tail_punct = PUNCT_CHAR.get(punct_labels[so.model_word_end], "")
            if tail_punct:
                pieces.append([tail_punct, True])
            i = so.model_word_end + 1
        else:
            pieces.append([words[i].replace("_", " "), False])
            tail_punct = PUNCT_CHAR.get(punct_labels[i], "")
            if tail_punct:
                pieces.append([tail_punct, True])
            i += 1

    # Viết hoa đầu câu chạy SAU khi đã khôi phục dấu câu (spec §17 quy tắc 4).
    # V1 thiếu hẳn bước này nên câu sau dấu chấm không được viết hoa.
    at_sentence_start = True
    for piece in pieces:
        text, protected = piece
        if text in {",", ".", "?", "!"}:
            at_sentence_start = text in SENTENCE_END
            continue
        if at_sentence_start and not protected and text[:1].isalpha():
            piece[0] = text[0].upper() + text[1:]
        at_sentence_start = False

    out = ""
    for text, _ in pieces:
        if text in {",", ".", "?", "!"} or not out:
            out += text
        else:
            out += " " + text
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def normalize_utterance(words, boundaries, types, punct_labels=None,
                        boundary_confidences=None, type_confidences=None,
                        raw_spans=None, config=None, alignment=None):
    """Đường chạy chung cho cả oracle (nhãn vàng) lẫn suy luận (nhãn mô hình).

    ``types`` là danh sách kiểu THEO SPAN, cùng thứ tự với span giải mã được từ
    ``boundaries`` — hai trục tách rời đúng như spec §6.
    """
    config = config or Config()
    spans = decode_spans(boundaries)
    if len(types) != len(spans):
        raise ValueError(f"có {len(spans)} span nhưng nhận {len(types)} kiểu")
    bc = boundary_confidences or [1.0] * len(spans)
    tc = type_confidences or [1.0] * len(spans)

    outputs = []
    for k, (start, end) in enumerate(spans):
        if alignment is not None:
            # Bộ chuẩn hoá LUÔN nhận văn bản ASR gốc, không nhận dạng đã ghép
            # từ (spec §3) — "hải_lý" phải tới parser dưới dạng "hải lý".
            raw_start, raw_end = alignment.raw_span(start, end)
            raw_words = alignment.raw_tokens[raw_start:raw_end + 1]
        elif raw_spans:
            raw_start, raw_end = raw_spans[k]
            raw_words = words[start:end + 1]
        else:
            raw_start, raw_end = start, end
            raw_words = words[start:end + 1]
        outputs.append(process_span(
            raw_words, types[k], start, end,
            boundaries[start:end + 1], bc[k], tc[k],
            raw_start, raw_end, config))
    return render(words, outputs, punct_labels), outputs
