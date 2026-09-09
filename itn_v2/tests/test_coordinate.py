import unittest

from itn_v2.normalizers.coordinate import CoordinateParser
from itn_v2.validators.coordinate import validate_coordinate


class TestCoordinate(unittest.TestCase):
    def test_do_phut_huong(self):
        self.assertEqual(CoordinateParser()("mười độ hai mươi lăm phút bắc").normalized,
                         "10°25'N")
        self.assertEqual(CoordinateParser()("một trăm lẻ bảy độ tám phút đông").normalized,
                         "107°08'E")

    def test_co_giay(self):
        self.assertEqual(
            CoordinateParser()("mười độ hai mươi lăm phút ba mươi giây bắc").normalized,
            "10°25'30\"N")

    def test_chi_co_do(self):
        self.assertEqual(CoordinateParser()("mười độ bắc").normalized, "10°N")

    def test_thieu_tu_do_thi_truot(self):
        self.assertFalse(CoordinateParser()("mười hai bắc").valid)

    def test_validator_mien_gia_tri(self):
        self.assertTrue(validate_coordinate("10°25'N")[0])
        self.assertFalse(validate_coordinate("95°25'N")[0])     # vĩ độ > 90
        self.assertTrue(validate_coordinate("107°08'E")[0])
        self.assertFalse(validate_coordinate("190°08'E")[0])    # kinh độ > 180


if __name__ == "__main__":
    unittest.main()
