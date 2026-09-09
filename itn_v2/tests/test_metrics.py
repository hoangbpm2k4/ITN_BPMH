import unittest

from itn_v2.evaluate import (boundary_strict_f1, checkpoint_score,
                             critical_category_accuracy,
                             critical_utterance_exact_match,
                             exact_normalization_accuracy,
                             false_normalization_rate, invalid_transition_rate,
                             type_strict_f1)


class TestSpanMetrics(unittest.TestCase):
    def test_boundary_khop_tuyet_doi(self):
        gold = [["B", "I", "E", "O", "S"]]
        r = boundary_strict_f1(gold, gold)
        self.assertEqual(r["f1"], 1.0)

    def test_lech_mot_tu_la_sai_ca_span(self):
        gold = [["B", "I", "E", "O", "O"]]
        pred = [["B", "E", "O", "O", "O"]]
        r = boundary_strict_f1(gold, pred)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 1)

    def test_type_f1_tach_theo_lop(self):
        gold = [[(0, 2, "COORD"), (4, 4, "CARDINAL")]]
        pred = [[(0, 2, "COORD"), (4, 4, "DIGIT_SEQ")]]
        r = type_strict_f1(gold, pred)
        self.assertEqual(r["per_type"]["COORD"]["f1"], 1.0)
        self.assertEqual(r["per_type"]["CARDINAL"]["fn"], 1)
        self.assertEqual(r["per_type"]["DIGIT_SEQ"]["fp"], 1)

    def test_ty_le_chuyen_trang_thai_sai(self):
        self.assertEqual(invalid_transition_rate([["B", "I", "E", "O"]]), 0.0)
        self.assertGreater(invalid_transition_rate([["I", "O", "E"]]), 0.0)
        self.assertGreater(invalid_transition_rate([["B", "I"]]), 0.0)   # span chưa đóng


class TestValueMetrics(unittest.TestCase):
    def test_do_chinh_xac_tuyet_doi_theo_lop(self):
        recs = [{"type": "COORD", "gold": "10°25'N", "pred": "10°25'N"},
                {"type": "COORD", "gold": "10°25'N", "pred": "10°26'N"},
                {"type": "DATE", "gold": "15/08", "pred": "15/08"}]
        r = exact_normalization_accuracy(recs)
        self.assertAlmostEqual(r["per_type"]["COORD"]["accuracy"], 0.5)
        self.assertAlmostEqual(r["per_type"]["DATE"]["accuracy"], 1.0)

    def test_lop_quan_trong(self):
        recs = [{"type": "HEADING", "gold": "090°", "pred": "090°"},
                {"type": "CARDINAL", "gold": "25", "pred": "26"}]
        self.assertAlmostEqual(critical_category_accuracy(recs), 1.0)

    def test_false_normalization_rate(self):
        recs = [{"type": "O", "raw": "con mèo", "pred": "con mèo"},
                {"type": "O", "raw": "ba con", "pred": "3 con"}]
        self.assertAlmostEqual(false_normalization_rate(recs), 0.5)

    def test_phat_ngon_quan_trong_sai_mot_chu_so_la_hong_ca_cau(self):
        utts = [{"records": [{"type": "COORD", "gold": "10°25'N", "pred": "10°25'N"}]},
                {"records": [{"type": "COORD", "gold": "10°25'N", "pred": "10°26'N"},
                             {"type": "DATE", "gold": "15/08", "pred": "15/08"}]}]
        self.assertAlmostEqual(critical_utterance_exact_match(utts), 0.5)

    def test_diem_chon_checkpoint(self):
        self.assertAlmostEqual(checkpoint_score(1, 1, 1, 1, 1), 1.0)
        self.assertAlmostEqual(checkpoint_score(0, 0, 0, 0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
