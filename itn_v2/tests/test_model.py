"""Kiểm tra mô hình bằng encoder giả — không nạp PhoBERT thật cho nhanh."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from itn_v2.config import Config
from itn_v2.labels import NUM_TYPES, TYPE2ID
from itn_v2.model import NUM_BOUNDARY, NUM_PUNCT, ITNv2Model
from itn_v2.tokenizer_alignment import build_pooling_matrix


class StubEncoder(nn.Module):
    def __init__(self, vocab=200, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.config = SimpleNamespace(hidden_size=hidden)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def make_batch(word_spans, num_subwords, batch=1):
    matrix = torch.tensor([build_pooling_matrix(word_spans, num_subwords)] * batch)
    input_ids = torch.randint(0, 200, (batch, num_subwords))
    attn = torch.ones(batch, num_subwords, dtype=torch.long)
    word_mask = torch.ones(batch, len(word_spans), dtype=torch.bool)
    return input_ids, attn, matrix, word_mask


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ITNv2Model(Config(), encoder=StubEncoder(), hidden_size=32)
        self.spans = [(1, 1), (2, 3), (4, 4), (5, 6)]
        self.batch = make_batch(self.spans, 8)

    def test_kich_thuoc_dau_ra(self):
        ids, attn, mat, _ = self.batch
        out = self.model(ids, attn, mat)
        W = len(self.spans)
        self.assertEqual(out["word_hidden"].shape, (1, W, 32))
        self.assertEqual(out["boundary_logits"].shape, (1, W, NUM_BOUNDARY))
        self.assertEqual(out["type_logits"].shape, (1, W, NUM_TYPES))
        self.assertEqual(out["punct_logits"].shape, (1, W, NUM_PUNCT))

    def test_gop_trung_binh_dung(self):
        """Từ có 2 subword phải cho vector đúng bằng trung bình hai vector đó."""
        ids, attn, mat, _ = self.batch
        hidden = self.model.encoder(input_ids=ids).last_hidden_state
        pooled = torch.bmm(mat, hidden)
        expected = hidden[0, 2:4].mean(dim=0)
        self.assertTrue(torch.allclose(pooled[0, 1], expected, atol=1e-6))

    def test_ba_thanh_phan_loss_va_backward(self):
        ids, attn, mat, wmask = self.batch
        out = self.model(ids, attn, mat)
        W = len(self.spans)
        b_tags = torch.tensor([[1, 3, 0, 4]])          # B E O S
        t_tags = torch.tensor([[TYPE2ID["DATE"]] * 2 + [0, TYPE2ID["HEADING"]]])
        p_tags = torch.zeros(1, W, dtype=torch.long)
        losses = self.model.compute_loss(out, b_tags, t_tags, p_tags, wmask)
        for key in ("loss", "boundary_nll", "type_ce", "punct_focal"):
            self.assertTrue(torch.isfinite(losses[key]), key)
        losses["loss"].backward()
        self.assertIsNotNone(self.model.boundary_head.weight.grad)

    def test_predict_tra_ve_span_kem_tin_cay(self):
        ids, attn, mat, wmask = self.batch
        results, _ = self.model.predict(ids, attn, mat, wmask)
        r = results[0]
        self.assertEqual(len(r["boundaries"]), len(self.spans))
        self.assertEqual(len(r["spans"]), len(r["types"]))
        self.assertEqual(len(r["spans"]), len(r["boundary_confidences"]))
        for c in r["boundary_confidences"] + r["type_confidences"]:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_learning_rate_phan_tang(self):
        groups = self.model.parameter_groups()
        self.assertEqual(len(groups), 2)
        self.assertLess(groups[0]["lr"], groups[1]["lr"])

    def test_khong_truyen_position_ids_cho_encoder(self):
        """V2 dùng vị trí gốc của PhoBERT (spec §3.1)."""
        seen = {}

        class Spy(StubEncoder):
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                seen.update(kwargs)
                return super().forward(input_ids, attention_mask)

        model = ITNv2Model(Config(), encoder=Spy(), hidden_size=32)
        ids, attn, mat, _ = self.batch
        model(ids, attn, mat)
        self.assertNotIn("position_ids", seen)


class TestModelToPipeline(unittest.TestCase):
    def test_noi_duoc_vao_duong_chay_chuan_hoa(self):
        """Đầu ra mô hình cắm thẳng vào pipeline tất định."""
        from itn_v2.pipeline import normalize_utterance
        torch.manual_seed(3)
        words = "phát_hiện xu ba lăm ở hướng không chín không".split()
        spans = [(i + 1, i + 1) for i in range(len(words))]
        model = ITNv2Model(Config(), encoder=StubEncoder(), hidden_size=32)
        ids, attn, mat, wmask = make_batch(spans, len(words) + 2)
        results, _ = model.predict(ids, attn, mat, wmask)
        r = results[0]
        text, outputs = normalize_utterance(
            words, r["boundaries"], r["types"],
            boundary_confidences=r["boundary_confidences"],
            type_confidences=r["type_confidences"])
        self.assertIsInstance(text, str)
        self.assertEqual(len(outputs), len(r["spans"]))
        # mô hình chưa huấn luyện -> tin cậy thấp -> phải giữ dạng nói, không bịa
        for o in outputs:
            if o.emitted:
                self.assertGreaterEqual(o.model_confidence, o.threshold)


if __name__ == "__main__":
    unittest.main()
