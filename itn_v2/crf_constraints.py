"""Ràng buộc chuyển trạng thái cho CRF ranh giới BIOES (spec §6.1).

Cấu trúc hợp lệ duy nhất: O | S | B -> I* -> E. Mọi chuyển trạng thái khác bị
phạt bằng một hằng số âm lớn, nên CRF gần như không bao giờ giải mã ra chuỗi
sai cấu trúc — thay vì phải vá bằng heuristic ở bước hậu xử lý như V1.
"""

import torch

from .labels import (BOUNDARY_LABELS, boundary_end_mask, boundary_start_mask,
                     boundary_transition_mask)

NEG = -1e4      # dùng số hữu hạn thay -inf để tránh NaN khi backward


def build_constraint_tensors(device=None, dtype=torch.float32):
    """Trả (transition_penalty, start_penalty, end_penalty) cộng thẳng vào score."""
    n = len(BOUNDARY_LABELS)
    trans = torch.zeros(n, n, dtype=dtype, device=device)
    allowed = boundary_transition_mask()
    for i in range(n):
        for j in range(n):
            if not allowed[i][j]:
                trans[i, j] = NEG
    start = torch.zeros(n, dtype=dtype, device=device)
    for i, ok in enumerate(boundary_start_mask()):
        if not ok:
            start[i] = NEG
    end = torch.zeros(n, dtype=dtype, device=device)
    for i, ok in enumerate(boundary_end_mask()):
        if not ok:
            end[i] = NEG
    return trans, start, end
