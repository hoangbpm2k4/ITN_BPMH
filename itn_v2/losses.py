"""Hàm mất mát (spec §8).

L_total = 1.00 * L_BOUNDARY_CRF + 1.00 * L_TYPE + 0.30 * L_PUNCT

Cố ý KHÔNG dùng trọng số lớp cực đoan 100x-150x như V1: spec §8.2 cấm, và
V1 phải dùng chúng chỉ vì một softmax phẳng gánh ba loại việc khác nhau.
"""

import torch
import torch.nn.functional as F


def focal_loss(logits, targets, class_weights=None, gamma=2.0, ignore_index=-100):
    """Focal loss cho đầu ra dấu câu (94% là nhãn 'không có dấu')."""
    ce = F.cross_entropy(logits, targets, weight=class_weights,
                         reduction="none", ignore_index=ignore_index)
    valid = targets != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce)[valid].mean()


def type_loss(logits, targets, ignore_index=-100):
    """Cross-entropy cho đầu ra kiểu. Từ ngoài span mang nhãn O."""
    return F.cross_entropy(logits, targets, reduction="mean", ignore_index=ignore_index)


def total_loss(boundary_nll, type_ce, punct_focal, config):
    return (config.w_boundary * boundary_nll
            + config.w_type * type_ce
            + config.w_punct * punct_focal)
