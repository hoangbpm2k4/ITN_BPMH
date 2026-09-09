"""Dataset V2: từ bản ghi gán nhãn trên token ASR gốc -> tensor huấn luyện.

Việc tách từ được làm Ở ĐÂY, bằng đúng bộ tách từ mà lúc suy luận sẽ dùng, nên
train và inference không bao giờ lệch nhau về ranh giới model word.
"""

import torch
from torch.utils.data import Dataset

from .config import Config
from .data.schema import PUNCT_LABELS, read_jsonl
from .labels import BOUNDARY2ID, TYPE2ID, encode_spans
from .segmentation_alignment import WordSegmenter, project_raw_spans_to_model
from .tokenizer_alignment import build_pooling_matrix, encode_words, get_tokenizer

PUNCT2ID = {l: i for i, l in enumerate(PUNCT_LABELS)}


class ITNv2Dataset(Dataset):
    def __init__(self, path, config=None, tokenizer=None, segmenter=None,
                 skip_conflicts=True, max_subwords=None):
        self.config = config or Config()
        self.tokenizer = tokenizer or get_tokenizer(self.config.phobert_path)
        self.segmenter = segmenter or WordSegmenter(self.config.phobert_path)
        self.max_subwords = max_subwords or self.config.window_useful
        self.items = []
        self.skipped = []
        for sample in read_jsonl(path):
            item = self._prepare(sample, skip_conflicts)
            if item is None:
                continue
            self.items.append(item)

    def _prepare(self, sample, skip_conflicts):
        alignment = self.segmenter.segment(sample.words)
        raw_spans = [(s, e) for s, e, _ in sample.spans]
        model_spans, conflicts = project_raw_spans_to_model(alignment, raw_spans)
        if conflicts and skip_conflicts:
            self.skipped.append((sample.id, "từ ghép cắt ngang span"))
            return None

        n_words = len(alignment.model_words)
        boundary = [BOUNDARY2ID[t] for t in encode_spans(model_spans, n_words)]

        types = [0] * n_words
        for (ws, we), (_, _, type_name) in zip(model_spans, sample.spans):
            for w in range(ws, we + 1):
                types[w] = TYPE2ID[type_name]

        punct = [0] * n_words
        for widx, (rs, re_) in enumerate(alignment.spans):
            for r in range(rs, re_ + 1):
                if sample.punct[r] != "O":
                    punct[widx] = PUNCT2ID[sample.punct[r]]

        enc = encode_words(alignment.model_words, self.tokenizer)
        if len(enc) > self.max_subwords:
            self.skipped.append((sample.id, f"dài {len(enc)} subword"))
            return None

        return {
            "sample": sample,
            "alignment": alignment,
            "model_spans": model_spans,
            "input_ids": enc.input_ids,
            "word_subword_spans": enc.word_subword_spans,
            "boundary": boundary,
            "types": types,
            "punct": punct,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    """Đệm theo cả trục subword lẫn trục từ; ma trận gộp cũng được đệm theo."""
    max_sub = max(len(b["input_ids"]) for b in batch)
    max_word = max(len(b["boundary"]) for b in batch)
    bs = len(batch)

    input_ids = torch.ones(bs, max_sub, dtype=torch.long)     # 1 = pad của PhoBERT
    attention = torch.zeros(bs, max_sub, dtype=torch.long)
    pooling = torch.zeros(bs, max_word, max_sub)
    word_mask = torch.zeros(bs, max_word, dtype=torch.bool)
    boundary = torch.zeros(bs, max_word, dtype=torch.long)
    types = torch.full((bs, max_word), -100, dtype=torch.long)
    punct = torch.full((bs, max_word), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        n_sub, n_word = len(b["input_ids"]), len(b["boundary"])
        input_ids[i, :n_sub] = torch.tensor(b["input_ids"])
        attention[i, :n_sub] = 1
        matrix = build_pooling_matrix(b["word_subword_spans"], n_sub)
        pooling[i, :n_word, :n_sub] = torch.tensor(matrix)
        word_mask[i, :n_word] = True
        boundary[i, :n_word] = torch.tensor(b["boundary"])
        types[i, :n_word] = torch.tensor(b["types"])
        punct[i, :n_word] = torch.tensor(b["punct"])

    return {
        "input_ids": input_ids, "attention_mask": attention,
        "pooling_matrix": pooling, "word_mask": word_mask,
        "boundary_tags": boundary, "type_tags": types, "punct_tags": punct,
        "batch": batch,
    }
