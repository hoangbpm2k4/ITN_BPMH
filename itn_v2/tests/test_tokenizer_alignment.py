import unittest

from itn_v2.segmentation_alignment import WordSegmenter
from itn_v2.tokenizer_alignment import (build_pooling_matrix, encode_words,
                                        get_tokenizer)


class TestTokenizerAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = get_tokenizer()
        cls.seg = WordSegmenter()

    def test_moi_tu_co_it_nhat_mot_subword(self):
        a = self.seg.segment("phát hiện xu ba lăm ở hướng không chín không")
        enc = encode_words(a.model_words, self.tok)
        self.assertEqual(enc.num_words(), len(a.model_words))
        for s, e in enc.word_subword_spans:
            self.assertLessEqual(s, e)

    def test_span_subword_lien_tuc_va_khong_chong_lan(self):
        a = self.seg.segment("tàu chở hàng đang neo đậu tại cảng hải phòng")
        enc = encode_words(a.model_words, self.tok)
        prev_end = 0    # 0 là CLS
        for s, e in enc.word_subword_spans:
            self.assertEqual(s, prev_end + 1)
            prev_end = e
        self.assertEqual(prev_end + 1, len(enc) - 1)   # phần tử cuối là SEP

    def test_ma_tran_gop_trung_binh(self):
        a = self.seg.segment("phát hiện tàu lạ")
        enc = encode_words(a.model_words, self.tok)
        m = build_pooling_matrix(enc.word_subword_spans, len(enc))
        self.assertEqual(len(m), enc.num_words())
        for row in m:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_khong_truyen_position_ids(self):
        """V2 dùng vị trí gốc của PhoBERT; EncodedWords không mang position_ids."""
        enc = encode_words(["tàu"], self.tok)
        self.assertFalse(hasattr(enc, "position_ids"))

    def test_khong_cat_ngan_o_day(self):
        words = ["tàu"] * 400
        enc = encode_words(words, self.tok)
        self.assertEqual(enc.num_words(), 400)


if __name__ == "__main__":
    unittest.main()
