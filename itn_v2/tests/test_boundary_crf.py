import itertools
import unittest

import torch

from itn_v2.confidence import boundary_confidence, model_confidence, type_confidence
from itn_v2.crf import ConstrainedCRF
from itn_v2.labels import BOUNDARY_LABELS, ID2BOUNDARY

K = len(BOUNDARY_LABELS)


def all_scores(crf, emissions, seq_len):
    trans = crf.transitions + crf.trans_penalty
    start = crf.start_transitions + crf.start_penalty
    end = crf.end_transitions + crf.end_penalty
    scores, seqs = [], []
    for seq in itertools.product(range(K), repeat=seq_len):
        s = start[seq[0]] + emissions[0, 0, seq[0]]
        for i in range(1, seq_len):
            s = s + trans[seq[i - 1], seq[i]] + emissions[0, i, seq[i]]
        scores.append(s + end[seq[-1]])
        seqs.append(seq)
    return torch.stack(scores), seqs


class TestCRF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.crf = ConstrainedCRF(K)
        self.T = 5
        self.em = torch.randn(1, self.T, K)
        self.mask = torch.ones(1, self.T, dtype=torch.bool)

    def test_ham_phan_hoach_khop_vet_can(self):
        scores, _ = all_scores(self.crf, self.em, self.T)
        self.assertTrue(torch.allclose(torch.logsumexp(scores, 0),
                                       self.crf.partition(self.em, self.mask)[0], atol=1e-4))

    def test_bien_duyen_khop_vet_can(self):
        scores, seqs = all_scores(self.crf, self.em, self.T)
        probs = torch.softmax(scores, 0)
        brute = torch.zeros(self.T, K)
        for p, seq in zip(probs, seqs):
            for i, k in enumerate(seq):
                brute[i, k] += p
        self.assertTrue(torch.allclose(brute, self.crf.marginals(self.em, self.mask)[0],
                                       atol=1e-4))

    def test_bien_duyen_tong_bang_mot(self):
        m = self.crf.marginals(self.em, self.mask)[0]
        self.assertTrue(torch.allclose(m.sum(-1), torch.ones(self.T), atol=1e-4))

    def test_viterbi_khop_vet_can(self):
        scores, seqs = all_scores(self.crf, self.em, self.T)
        best = seqs[int(scores.argmax())]
        self.assertEqual(self.crf.decode(self.em, self.mask)[0], list(best))

    def test_nll_khong_am(self):
        tags = torch.tensor([self.crf.decode(self.em, self.mask)[0]])
        nll = self.crf.neg_log_likelihood(self.em, tags, self.mask)
        self.assertGreaterEqual(float(nll), -1e-4)

    def test_giai_ma_luon_hop_le_ve_cau_truc(self):
        """Ràng buộc chuyển trạng thái phải loại sạch chuỗi BIOES sai."""
        torch.manual_seed(123)
        bad = 0
        for _ in range(200):
            em = torch.randn(1, 7, K)
            tags = [ID2BOUNDARY[t] for t in self.crf.decode(em, torch.ones(1, 7, dtype=torch.bool))[0]]
            prev = "O"
            for t in tags:
                if prev in ("O", "S", "E") and t in ("I", "E"):
                    bad += 1
                if prev in ("B", "I") and t in ("O", "B", "S"):
                    bad += 1
                prev = t
            if prev in ("B", "I"):
                bad += 1
        self.assertEqual(bad, 0)

    def test_ton_trong_mask_padding(self):
        em = torch.randn(2, 6, K)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[1, 4:] = False
        paths = self.crf.decode(em, mask)
        self.assertEqual(len(paths[0]), 6)
        self.assertEqual(len(paths[1]), 4)


class TestConfidence(unittest.TestCase):
    def test_boundary_confidence_lay_gia_tri_nho_nhat(self):
        marg = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.7, 0.3, 0.0],
                             [0.0, 0.0, 0.0, 0.95, 0.05]])
        conf = boundary_confidence(marg, [0, 2, 3], 0, 2)
        self.assertAlmostEqual(conf, 0.7, places=5)

    def test_type_confidence_gop_trung_binh_logit(self):
        logits = torch.zeros(3, 47)
        logits[0, 5] = 10.0
        logits[1, 5] = 10.0
        logits[2, 5] = 10.0
        idx, conf = type_confidence(logits, 0, 2)
        self.assertEqual(idx, 5)
        self.assertGreater(conf, 0.99)

    def test_model_confidence_la_min(self):
        self.assertAlmostEqual(model_confidence(0.7, 0.95), 0.7)


if __name__ == "__main__":
    unittest.main()
