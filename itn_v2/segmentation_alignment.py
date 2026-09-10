"""Tách từ tiếng Việt + ánh xạ NGƯỢC ĐƯỢC về token ASR gốc (spec §3).

PhoBERT được huấn luyện trên văn bản đã tách từ ghép, nên phải có tầng này.
Nhưng bộ chuẩn hoá thì luôn nhận **văn bản ASR gốc**, không nhận dạng đã ghép —
vì vậy ánh xạ hai chiều là bắt buộc, không phải tuỳ chọn.

Không dùng underthesea (chưa cài trên máy này). Từ điển lấy thẳng từ vốn từ
PhoBERT: 32.724 token chứa dấu ``_`` chính là các từ ghép mà PhoBERT biết —
tách theo từ điển này đảm bảo kết quả luôn nằm trong vốn từ của mô hình.
"""

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from .config import PHOBERT_PATH

MAX_WORD_SYLLABLES = 4


@lru_cache(maxsize=None)
def load_compound_lexicon(phobert_path: str = PHOBERT_PATH):
    """Tập các từ ghép (dạng tuple âm tiết, chữ thường) có trong vốn từ PhoBERT."""
    tok_path = Path(phobert_path) / "tokenizer.json"
    with tok_path.open(encoding="utf-8") as fh:
        vocab = json.load(fh)["model"]["vocab"]
    lexicon = set()
    for token in vocab:
        word = token[:-4] if token.endswith("</w>") else token
        if "_" not in word:
            continue
        syllables = tuple(unicodedata.normalize("NFC", word).lower().split("_"))
        if 2 <= len(syllables) <= MAX_WORD_SYLLABLES and all(syllables):
            lexicon.add(syllables)
    return frozenset(lexicon)


@dataclass
class Alignment:
    """Giữ cả hai tầng token và ánh xạ giữa chúng."""
    raw_tokens: List[str]
    model_words: List[str]
    spans: List[Tuple[int, int]]      # chỉ số model word -> (raw_start, raw_end) bao gồm hai đầu

    def raw_span(self, model_start, model_end):
        return self.spans[model_start][0], self.spans[model_end][1]

    def raw_text(self, model_start, model_end):
        s, e = self.raw_span(model_start, model_end)
        return " ".join(self.raw_tokens[s:e + 1])

    def is_reversible(self) -> bool:
        """Ghép lại các đoạn raw phải dựng đúng chuỗi token ASR ban đầu."""
        rebuilt = []
        cursor = 0
        for s, e in self.spans:
            if s != cursor:
                return False
            rebuilt.extend(self.raw_tokens[s:e + 1])
            cursor = e + 1
        return cursor == len(self.raw_tokens) and rebuilt == self.raw_tokens


class WordSegmenter:
    """Khớp dài nhất từ trái sang phải theo từ điển từ ghép của PhoBERT."""

    def __init__(self, phobert_path: str = PHOBERT_PATH,
                 max_syllables: int = MAX_WORD_SYLLABLES, enabled: bool = True):
        # Từ điển rỗng = mỗi token ASR là một model word, tức tắt hẳn tách từ.
        # Ánh xạ hai chiều vẫn giữ nguyên nên phần còn lại của pipeline không đổi.
        self.lexicon = load_compound_lexicon(phobert_path) if enabled else frozenset()
        self.max_syllables = max_syllables

    def segment(self, text_or_tokens) -> Alignment:
        if isinstance(text_or_tokens, str):
            raw_tokens = unicodedata.normalize("NFC", text_or_tokens).split()
        else:
            raw_tokens = [unicodedata.normalize("NFC", t) for t in text_or_tokens]

        lowered = [t.lower() for t in raw_tokens]
        model_words, spans = [], []
        i, n = 0, len(raw_tokens)
        while i < n:
            for size in range(min(self.max_syllables, n - i), 1, -1):
                if tuple(lowered[i:i + size]) in self.lexicon:
                    model_words.append("_".join(raw_tokens[i:i + size]))
                    spans.append((i, i + size - 1))
                    i += size
                    break
            else:
                model_words.append(raw_tokens[i])
                spans.append((i, i))
                i += 1
        return Alignment(raw_tokens, model_words, spans)


def project_raw_spans_to_model(alignment: Alignment, raw_spans):
    """Đưa nhãn gắn trên token ASR gốc về chỉ số model word.

    Trả (model_spans, conflicts). ``conflicts`` liệt kê những span bị một từ ghép
    cắt ngang — phải xử lý khi sinh dữ liệu, không được im lặng bỏ qua.
    """
    start_of = {}
    end_of = {}
    for widx, (s, e) in enumerate(alignment.spans):
        for r in range(s, e + 1):
            start_of.setdefault(r, widx)
            end_of[r] = widx

    model_spans, conflicts = [], []
    for rs, re_ in raw_spans:
        ws, we = start_of[rs], end_of[re_]
        exact = alignment.spans[ws][0] == rs and alignment.spans[we][1] == re_
        model_spans.append((ws, we))
        if not exact:
            conflicts.append({
                "raw_span": (rs, re_),
                "model_span": (ws, we),
                "raw_text": " ".join(alignment.raw_tokens[rs:re_ + 1]),
                "model_text": " ".join(alignment.model_words[ws:we + 1]),
            })
    return model_spans, conflicts
