"""Gióng model word <-> subword PhoBERT (spec §3.1, §3.2).

Hai khác biệt bắt buộc so với V1:
  1. KHÔNG ghi đè position id. V1 gán mọi subword của một từ cùng một vị trí;
     V2 dùng vị trí gốc của PhoBERT, tức là không truyền position_ids.
  2. Biểu diễn của một từ là TRUNG BÌNH mọi subword của nó, không phải subword
     đầu tiên.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from .config import PHOBERT_PATH


@lru_cache(maxsize=None)
def get_tokenizer(path: str = PHOBERT_PATH):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, use_fast=False)


@dataclass
class EncodedWords:
    input_ids: List[int]
    attention_mask: List[int]
    word_subword_spans: List[Tuple[int, int]]   # từ -> (sub_start, sub_end) bao gồm hai đầu
    model_words: List[str]

    def __len__(self):
        return len(self.input_ids)

    def num_words(self):
        return len(self.word_subword_spans)


def encode_words(model_words, tokenizer=None, add_special_tokens=True) -> EncodedWords:
    """Mã hoá danh sách model word, giữ ranh giới subword của từng từ.

    Không cắt ngắn ở đây: việc chia cửa sổ do ``windowing`` lo, để không bao giờ
    im lặng mất phần đuôi câu như V1.
    """
    tokenizer = tokenizer or get_tokenizer()
    ids, spans = [], []
    if add_special_tokens:
        ids.append(tokenizer.cls_token_id)
    for word in model_words:
        sub_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        if not sub_ids:
            sub_ids = [tokenizer.unk_token_id]
        start = len(ids)
        ids.extend(sub_ids)
        spans.append((start, len(ids) - 1))
    if add_special_tokens:
        ids.append(tokenizer.sep_token_id)
    return EncodedWords(ids, [1] * len(ids), spans, list(model_words))


def build_pooling_matrix(word_subword_spans, num_subwords):
    """Ma trận [số_từ, số_subword] để gộp trung bình. Dùng cho torch.matmul."""
    matrix = [[0.0] * num_subwords for _ in word_subword_spans]
    for w, (s, e) in enumerate(word_subword_spans):
        width = e - s + 1
        for j in range(s, e + 1):
            matrix[w][j] = 1.0 / width
    return matrix


def mean_pool(hidden_states, word_subword_spans):
    """Gộp trung bình bằng torch. hidden_states: [T, H] -> [số_từ, H]."""
    import torch
    outs = []
    for s, e in word_subword_spans:
        outs.append(hidden_states[s:e + 1].mean(dim=0))
    return torch.stack(outs) if outs else hidden_states.new_zeros((0, hidden_states.shape[-1]))
