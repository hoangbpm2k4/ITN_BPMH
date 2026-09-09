import unittest

from itn_v2.normalizers.equipment import EquipmentIDParser


class TestEquipmentID(unittest.TestCase):
    def setUp(self):
        self.p = EquipmentIDParser()

    def test_giu_nguyen_hoa_thuong_chuan(self):
        """Lỗi V1: viết hoa toàn bộ tiền tố nên Su-30 thành SU-30."""
        self.assertEqual(self.p("su ba mươi").normalized, "Su-30")
        self.assertEqual(self.p("míc hai mốt").normalized, "MiG-21")
        self.assertEqual(self.p("mi mười bảy").normalized, "Mi-17")
        self.assertEqual(self.p("ka hát ba lăm").normalized, "Kh-35")

    def test_bien_the_phat_am_cho_cung_ket_qua(self):
        for spoken in ["xu ba lăm", "su ba lăm", "su ba mươi lăm", "xu ba mươi lăm"]:
            self.assertEqual(self.p(spoken).normalized, "Su-35", spoken)

    def test_cac_ho_van_pham_bat_buoc(self):
        cases = {
            "a ka bốn bảy": "AK-47",
            "ép mười sáu": "F-16",
            "bê năm hai": "B-52",
            "em mười sáu": "M16",              # không gạch nối
            "em bốn a một": "M4A1",            # chữ + số xen kẽ
            "rờ pê giê bảy": "RPG-7",          # tiền tố nhiều chữ
            "tê chín mươi": "T-90",
            "su ba mươi em ka hai": "Su-30MK2",  # có hậu tố
            "ka hai tám": "Ka-28",
            "pê tám trăm": "P-800",
            "em u chín mươi": "MU90",
            "em ca bốn sáu": "MK 46",           # có dấu cách
            "ét e tê sáu lăm": "SET-65",
            "năm ba sáu lăm ka e": "53-65KE",
            "a ka sáu trăm ba mươi": "AK-630",
        }
        for spoken, expected in cases.items():
            self.assertEqual(self.p(spoken).normalized, expected, spoken)

    def test_model_chua_gap_van_dung_van_pham(self):
        self.assertEqual(self.p("su chín chín").normalized, "Su-99")
        self.assertEqual(self.p("ép hai hai").normalized, "F-22")

    def test_khong_khop_gi_thi_truot(self):
        r = self.p("con mèo")
        self.assertFalse(r.valid)


if __name__ == "__main__":
    unittest.main()
