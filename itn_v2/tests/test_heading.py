import unittest

from itn_v2.normalizers.coordinate import BearingParser, HeadingParser
from itn_v2.validators.heading import validate_heading


class TestHeading(unittest.TestCase):
    def test_giu_so_khong_dau_ba_chu_so(self):
        self.assertEqual(HeadingParser()("không chín không").normalized, "090°")
        self.assertEqual(HeadingParser()("không không chín").normalized, "009°")
        self.assertEqual(HeadingParser()("một hai không").normalized, "120°")

    def test_doc_theo_so_dem_cung_ra_dung(self):
        self.assertEqual(HeadingParser()("chín mươi").normalized, "090°")

    def test_bien(self):
        self.assertEqual(HeadingParser()("không không không").normalized, "000°")
        self.assertEqual(HeadingParser()("ba năm chín").normalized, "359°")

    def test_ngoai_mien_thi_truot(self):
        r = HeadingParser()("ba bảy không")
        self.assertFalse(r.valid)
        self.assertIn("359", r.reason)

    def test_bearing_dung_chung_khuon(self):
        self.assertEqual(BearingParser()("không chín không").normalized, "090°")

    def test_validator(self):
        self.assertTrue(validate_heading("090°")[0])
        self.assertFalse(validate_heading("90°")[0])      # thiếu số 0 đầu
        self.assertFalse(validate_heading("370°")[0])


if __name__ == "__main__":
    unittest.main()
