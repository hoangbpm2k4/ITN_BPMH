import unittest

from itn_v2.segmentation_alignment import (WordSegmenter,
                                           project_raw_spans_to_model)


class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seg = WordSegmenter()

    def test_tu_dien_du_lon(self):
        self.assertGreater(len(self.seg.lexicon), 10_000)

    def test_ghep_tu_ghep(self):
        a = self.seg.segment("tàu chở hàng đang neo đậu tại cảng hải phòng")
        self.assertIn("neo_đậu", a.model_words)
        self.assertIn("hải_phòng", a.model_words)

    def test_anh_xa_nguoc_duoc(self):
        for text in ["phát hiện xu ba lăm ở hướng không chín không",
                     "tốc độ mười hai hải lý một giờ",
                     "một hai ba bốn năm"]:
            a = self.seg.segment(text)
            self.assertTrue(a.is_reversible(), text)
            self.assertEqual(a.raw_tokens, text.split())

    def test_raw_text_tra_ve_dang_ASR_goc(self):
        a = self.seg.segment("tốc độ mười hai hải lý một giờ")
        idx = a.model_words.index("hải_lý")
        self.assertEqual(a.raw_text(idx, idx), "hải lý")

    def test_chieu_span_raw_sang_model(self):
        a = self.seg.segment("phát hiện xu ba lăm ở hướng không chín không")
        spans, conflicts = project_raw_spans_to_model(a, [(2, 4), (7, 9)])
        self.assertEqual(conflicts, [])
        self.assertEqual(a.raw_text(*spans[0]), "xu ba lăm")
        self.assertEqual(a.raw_text(*spans[1]), "không chín không")

    def test_bao_xung_dot_khi_tu_ghep_cat_ngang_span(self):
        a = self.seg.segment("tốc độ mười hai hải lý một giờ")
        # span chỉ lấy "hai hải" -> cắt ngang cả mười_hai lẫn hải_lý
        _, conflicts = project_raw_spans_to_model(a, [(3, 4)])
        self.assertTrue(conflicts, "phải báo xung đột thay vì im lặng")

    def test_cau_rong(self):
        a = self.seg.segment("")
        self.assertEqual(a.model_words, [])
        self.assertTrue(a.is_reversible())


if __name__ == "__main__":
    unittest.main()
