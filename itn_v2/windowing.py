"""Cửa sổ trượt có chồng lấn cho câu dài (spec §19).

V1 cắt cứng ở 256 subword rồi vứt phần đuôi, không báo gì. V2 chia cửa sổ theo
ranh giới TỪ, ghép lại theo chỉ số từ gốc, và khi một từ nằm trong hai cửa sổ
thì ưu tiên dự đoán từ cửa sổ mà từ đó nằm gần tâm hơn.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Window:
    word_start: int          # bao gồm
    word_end: int            # không bao gồm
    subword_count: int

    def center(self):
        return (self.word_start + self.word_end - 1) / 2.0


def plan_windows(word_subword_lengths, useful=224, overlap=40) -> List[Window]:
    """Chia theo ranh giới từ sao cho mỗi cửa sổ <= ``useful`` subword.

    ``useful`` đã trừ chỗ cho CLS/SEP ở phía gọi.
    """
    n = len(word_subword_lengths)
    if n == 0:
        return []
    windows = []
    start = 0
    while start < n:
        total, end = 0, start
        while end < n and total + word_subword_lengths[end] <= useful:
            total += word_subword_lengths[end]
            end += 1
        if end == start:            # một từ dài hơn cả cửa sổ -> vẫn phải nhận
            end = start + 1
            total = word_subword_lengths[start]
        windows.append(Window(start, end, total))
        if end >= n:
            break
        # lùi lại theo số subword chồng lấn, tính bằng từ
        back, k = 0, end
        while k > start + 1 and back < overlap:
            k -= 1
            back += word_subword_lengths[k]
        start = max(k, start + 1)
    return windows


def stitch(window_predictions, windows, num_words):
    """Ghép dự đoán theo từ. ``window_predictions[i]`` là list dài
    ``windows[i].word_end - windows[i].word_start``.

    Với từ xuất hiện ở nhiều cửa sổ, giữ dự đoán của cửa sổ có từ đó gần tâm nhất.
    """
    chosen = [None] * num_words
    best_score = [float("inf")] * num_words
    for preds, win in zip(window_predictions, windows):
        center = win.center()
        for offset, value in enumerate(preds):
            w = win.word_start + offset
            if w >= num_words:
                break
            distance = abs(w - center)
            if distance < best_score[w]:
                best_score[w] = distance
                chosen[w] = value
    missing = [i for i, v in enumerate(chosen) if v is None]
    if missing:
        raise ValueError(f"các từ sau không được cửa sổ nào phủ: {missing[:10]}")
    return chosen
