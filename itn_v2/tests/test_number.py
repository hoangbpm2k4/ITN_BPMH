import unittest

from itn_v2.normalizers.number import (CardinalParser, DecimalParser,
                                       DigitSequenceParser, FractionParser,
                                       OrdinalParser, PercentParser,
                                       RangeParser, RatioParser, VersionParser,
                                       read_cardinal, read_digit_string, read_year)


class TestDigitSequence(unittest.TestCase):
    """Chế độ đọc dãy chữ số — chỗ V1 phá huỷ dữ liệu."""

    def test_doc_roi_tung_chu(self):
        self.assertEqual(read_digit_string("không ba một hai".split()), "0312")
        self.assertEqual(read_digit_string("không tám tám hai một ba bốn năm sáu".split()),
                         "088213456")

    def test_doc_theo_nhom_cho_cung_ket_qua(self):
        """Cùng một số, đọc rời hay đọc nhóm đều phải ra như nhau.

        V1 trả về '6' cho cách đọc nhóm — mất 8/9 chữ số, im lặng.
        """
        roi = read_digit_string("không tám tám hai một ba bốn năm sáu".split())
        nhom = read_digit_string("không tám tám hai mươi mốt ba bốn năm sáu".split())
        self.assertEqual(roi, nhom)
        self.assertEqual(nhom, "088213456")

    def test_giu_so_khong_dau(self):
        self.assertEqual(read_digit_string("không chín không".split()), "090")

    def test_bien_the_phat_am(self):
        self.assertEqual(read_digit_string("ba lăm".split()), "35")
        self.assertEqual(read_digit_string("ba nhăm".split()), "35")
        self.assertEqual(read_digit_string("hai mốt".split()), "21")
        self.assertEqual(read_digit_string("hai tư".split()), "24")

    def test_tu_la_thi_bao_loi(self):
        r = DigitSequenceParser()("không ba xyz")
        self.assertFalse(r.valid)
        self.assertIn("xyz", r.reason)


class TestCardinal(unittest.TestCase):
    def test_co_ban(self):
        self.assertEqual(read_cardinal("hai mươi lăm".split()), 25)
        self.assertEqual(read_cardinal("một trăm hai mươi ba".split()), 123)
        self.assertEqual(read_cardinal("một trăm lẻ bảy".split()), 107)
        self.assertEqual(read_cardinal("tám mươi lăm".split()), 85)

    def test_khong_tram_khong_bi_thanh_mot_tram(self):
        """V1 coi 'không trăm' là 100 nên 1020 thành 1120."""
        self.assertEqual(read_cardinal("một nghìn không trăm hai mươi".split()), 1020)

    def test_thang_lon(self):
        self.assertEqual(read_cardinal("hai trăm năm mươi triệu".split()), 250_000_000)
        self.assertEqual(read_cardinal("hai tỷ ba trăm triệu".split()), 2_300_000_000)

    def test_phan_nhom_nghin(self):
        self.assertEqual(CardinalParser()("hai trăm năm mươi triệu").normalized, "250.000.000")

    def test_so_am_la_modifier(self):
        self.assertEqual(CardinalParser()("âm mười").normalized, "-10")
        self.assertEqual(CardinalParser()("trừ hai mươi lăm").normalized, "-25")


class TestYear(unittest.TestCase):
    def test_cac_cach_doc_nam(self):
        self.assertEqual(read_year("hai không hai sáu".split()), 2026)
        self.assertEqual(read_year("hai không hai lăm".split()), 2025)
        self.assertEqual(read_year("một chín tám mươi".split()), 1980)
        self.assertEqual(read_year("hai nghìn không trăm hai mươi tư".split()), 2024)


class TestOtherNumericTypes(unittest.TestCase):
    def test_decimal(self):
        self.assertEqual(DecimalParser()("ba phẩy năm").normalized, "3,5")
        self.assertEqual(DecimalParser()("âm mười phẩy năm").normalized, "-10,5")

    def test_decimal_phan_nguyen_doc_roi_khong_chen_dau_nghin(self):
        self.assertEqual(DecimalParser()("một hai một phẩy năm").normalized, "121,5")

    def test_percent(self):
        self.assertEqual(PercentParser()("hai mươi lăm phần trăm").normalized, "25%")
        self.assertEqual(PercentParser()("ba phẩy năm phần trăm").normalized, "3,5%")

    def test_fraction_ratio(self):
        self.assertEqual(FractionParser()("một phần hai").normalized, "1/2")
        self.assertEqual(RatioParser()("một trên mười").normalized, "1:10")

    def test_ordinal(self):
        self.assertEqual(OrdinalParser()("thứ hai").normalized, "thứ 2")
        self.assertEqual(OrdinalParser()("thứ nhất").normalized, "thứ 1")

    def test_range(self):
        self.assertEqual(RangeParser()("từ mười đến hai mươi").normalized, "10-20")

    def test_version(self):
        self.assertEqual(VersionParser()("phiên bản một chấm hai chấm ba").normalized, "1.2.3")
        self.assertEqual(VersionParser()("một chấm hai").normalized, "1.2")

    def test_version_thieu_thanh_phan_thi_truot(self):
        self.assertFalse(VersionParser()("một").valid)


if __name__ == "__main__":
    unittest.main()
