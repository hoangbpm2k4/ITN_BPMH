"""Định nghĩa độ tin cậy (spec §9).

Cấm dùng log-likelihood thô của cả chuỗi CRF làm độ tin cậy của một span —
nó là đại lượng của cả câu, không phải của span, và không so sánh được giữa
các câu dài ngắn khác nhau.
"""

import torch


def boundary_confidence(marginals, predicted_tags, start, end):
    """min P(nhãn ranh giới đã chọn | x) trên mọi từ trong span (spec §9.1)."""
    values = [marginals[i, predicted_tags[i]] for i in range(start, end + 1)]
    return float(min(float(v) for v in values))


def span_type_logits(type_logits, start, end):
    """Gộp trung bình logit kiểu trên span — baseline V2 (spec §6.2)."""
    return type_logits[start:end + 1].mean(dim=0)


def type_confidence(type_logits, start, end):
    """(kiểu dự đoán, xác suất) sau khi gộp trung bình logit (spec §9.2)."""
    pooled = span_type_logits(type_logits, start, end)
    probs = torch.softmax(pooled, dim=-1)
    idx = int(probs.argmax())
    return idx, float(probs[idx])


def model_confidence(boundary_conf, type_conf):
    """Baseline: lấy giá trị nhỏ hơn (spec §9.3)."""
    return min(boundary_conf, type_conf)
