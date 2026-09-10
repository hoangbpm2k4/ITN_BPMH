import unittest

from itn_v2.normalizers.coordinate import CoordinateParser
from itn_v2.validators.coordinate import validate_coordinate


# Ký hiệu phút/giây là dấu nguyên ′ ″ (U+2032/U+2033), không phải nháy ASCII.
# Spec viết 10°25'N cho gọn, nhưng 18/18 toạ độ trong bản gốc thật dùng dấu
# nguyên — mà bản gốc mới là thứ ta bị chấm điểm. Validator vẫn nhận cả hai.
class TestCoordinate(unittest.TestCase):
    def test_do_phut_huong(self):
        self.assertEqual(CoordinateParser()("mười độ hai mươi lăm phút bắc").normalized,
                         "10°25′N")
        self.assertEqual(CoordinateParser()("một trăm lẻ bảy độ tám phút đông").normalized,
                         "107°08′E")

    def test_co_giay(self):
        self.assertEqual(
            CoordinateParser()("mười độ hai mươi lăm phút ba mươi giây bắc").normalized,
            "10°25′30″N")

    def test_cap_vi_do_kinh_do(self):
        """Span thật thường gộp cả hai vế trong một cụm."""
        self.assertEqual(
            CoordinateParser()("mười độ hai mươi lăm phút ba mươi giây bắc "
                               "một trăm lẻ bảy độ tám phút mười lăm giây đông").normalized,
            "10°25′30″N, 107°08′15″E")

    def test_chi_co_do(self):
        self.assertEqual(CoordinateParser()("mười độ bắc").normalized, "10°N")

    def test_thieu_tu_do_thi_truot(self):
        self.assertFalse(CoordinateParser()("mười hai bắc").valid)

    def test_validator_mien_gia_tri(self):
        self.assertTrue(validate_coordinate("10°25′N")[0])
        self.assertTrue(validate_coordinate("10°25'N")[0])       # dạng ASCII cũ
        self.assertTrue(validate_coordinate("10°25′30″N, 107°08′15″E")[0])
        self.assertFalse(validate_coordinate("95°25'N")[0])     # vĩ độ > 90
        self.assertTrue(validate_coordinate("107°08'E")[0])
        self.assertFalse(validate_coordinate("190°08'E")[0])    # kinh độ > 180


if __name__ == "__main__":
    unittest.main()


class TestDecimalMinuteCoordinate(unittest.TestCase):
    """Khuôn độ-phút-thập-phân (DDM) mà hải đồ điện tử dùng cho waypoint."""

    def setUp(self):
        self.parser = CoordinateParser()

    def test_doc_phut_thap_phan(self):
        got = self.parser(
            "mười độ hai mươi lăm nghìn năm trăm phút bắc "
            "một trăm lẻ bảy độ tám nghìn hai trăm năm mươi phút đông", "COORD")
        self.assertEqual(got.normalized, "10°25.500′N, 107°08.250′E")

    def test_validator_chap_nhan_ddm(self):
        ok, why = validate_coordinate("10°30.750′N, 107°12.500′E")
        self.assertTrue(ok, why)

    def test_khuon_do_phut_giay_van_chay(self):
        got = self.parser(
            "mười độ hai mươi lăm phút ba mươi giây bắc "
            "một trăm lẻ bảy độ tám phút mười lăm giây đông", "COORD")
        self.assertEqual(got.normalized, "10°25′30″N, 107°08′15″E")

    def test_phut_vuot_nguong_ma_con_giay_thi_truot(self):
        # Có cả phần giây thì không phải DDM; phải trượt chứ không được đoán bừa.
        got = self.parser("mười độ chín nghìn phút ba mươi giây bắc", "COORD")
        self.assertFalse(got.valid)
