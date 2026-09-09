"""Mô hình V2: PhoBERT + CRF ranh giới + đầu ra kiểu + đầu ra dấu câu (spec §7).

Ba khác biệt bắt buộc so với V1:
  1. KHÔNG ghi đè position id — dùng vị trí gốc của PhoBERT.
  2. Biểu diễn từ = TRUNG BÌNH mọi subword, không phải subword đầu.
  3. Ranh giới và kiểu là HAI trục tách rời, không phải nhãn ghép B-COORD.
"""

import torch
import torch.nn as nn

from .confidence import (boundary_confidence, model_confidence,
                         span_type_logits, type_confidence)
from .config import Config
from .crf import ConstrainedCRF
from .labels import (BOUNDARY_LABELS, ID2BOUNDARY, ID2TYPE, NUM_TYPES,
                     decode_spans)
from .losses import focal_loss, total_loss, type_loss

NUM_BOUNDARY = len(BOUNDARY_LABELS)
NUM_PUNCT = 4


class ITNv2Model(nn.Module):
    def __init__(self, config: Config = None, encoder=None, hidden_size=None):
        super().__init__()
        self.config = config or Config()
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(self.config.phobert_path)
        self.encoder = encoder
        hidden = hidden_size or encoder.config.hidden_size

        self.dropout = nn.Dropout(self.config.dropout)
        self.boundary_head = nn.Linear(hidden, NUM_BOUNDARY)
        self.crf = ConstrainedCRF(NUM_BOUNDARY)
        self.type_head = nn.Linear(hidden, NUM_TYPES)      # 46 kiểu + O
        self.punct_head = nn.Linear(hidden, NUM_PUNCT)
        self.register_buffer(
            "punct_weights", torch.tensor(self.config.punct_class_weights, dtype=torch.float))

    def encode_words(self, input_ids, attention_mask, pooling_matrix):
        """pooling_matrix [B, W, T] — gộp trung bình subword về từ."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        word_hidden = torch.bmm(pooling_matrix.to(hidden.dtype), hidden)
        return self.dropout(word_hidden)

    def forward(self, input_ids, attention_mask, pooling_matrix):
        word_hidden = self.encode_words(input_ids, attention_mask, pooling_matrix)
        return {
            "word_hidden": word_hidden,
            "boundary_logits": self.boundary_head(word_hidden),
            "type_logits": self.type_head(word_hidden),
            "punct_logits": self.punct_head(word_hidden),
        }

    def compute_loss(self, outputs, boundary_tags, type_tags, punct_tags, word_mask):
        b_nll = self.crf.neg_log_likelihood(outputs["boundary_logits"], boundary_tags, word_mask)
        t_ce = type_loss(outputs["type_logits"].reshape(-1, NUM_TYPES), type_tags.reshape(-1))
        p_fl = focal_loss(outputs["punct_logits"].reshape(-1, NUM_PUNCT),
                          punct_tags.reshape(-1), self.punct_weights,
                          self.config.punct_focal_gamma)
        return {
            "loss": total_loss(b_nll, t_ce, p_fl, self.config),
            "boundary_nll": b_nll, "type_ce": t_ce, "punct_focal": p_fl,
        }

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, pooling_matrix, word_mask):
        """Giải mã ranh giới -> dựng span -> gộp logit kiểu -> một kiểu mỗi span.

        Trả cả độ tin cậy của từng span để cổng ngưỡng ở ``pipeline`` dùng.
        """
        out = self.forward(input_ids, attention_mask, pooling_matrix)
        boundary_paths = self.crf.decode(out["boundary_logits"], word_mask)
        marginals = self.crf.marginals(out["boundary_logits"], word_mask)

        results = []
        for b, path in enumerate(boundary_paths):
            tags = [ID2BOUNDARY[t] for t in path]
            spans = decode_spans(tags)
            entry = {"boundaries": tags, "spans": [], "types": [],
                     "boundary_confidences": [], "type_confidences": [],
                     "punct": [int(x) for x in out["punct_logits"][b][:len(path)].argmax(-1)]}
            for start, end in spans:
                type_idx, t_conf = type_confidence(out["type_logits"][b], start, end)
                b_conf = boundary_confidence(marginals[b], path, start, end)
                entry["spans"].append((start, end))
                entry["types"].append(ID2TYPE[type_idx])
                entry["boundary_confidences"].append(b_conf)
                entry["type_confidences"].append(t_conf)
            results.append(entry)
        return results, out

    def parameter_groups(self):
        """Learning rate phân tầng: encoder chậm, các đầu ra nhanh (spec §24)."""
        head_params = list(self.boundary_head.parameters()) + list(self.crf.parameters()) + \
            list(self.type_head.parameters()) + list(self.punct_head.parameters())
        return [
            {"params": self.encoder.parameters(), "lr": self.config.encoder_lr},
            {"params": head_params, "lr": self.config.head_lr},
        ]
