import unittest

from itn_v2.normalizers.electronic import ElectronicParser, detect_subtype


class TestElectronic(unittest.TestCase):
    def test_nhan_dang_phan_loai_con(self):
        self.assertEqual(detect_subtype("vts@portcontrol.vn"), "EMAIL")
        self.assertEqual(detect_subtype("www.vinamarine.gov.vn"), "URL")
        self.assertEqual(detect_subtype("192.168.10.25"), "IPV4")
        self.assertEqual(detect_subtype("2001:db8:100::25"), "IPV6")
        self.assertEqual(detect_subtype("00:1A:2B:3C:4D:5E"), "MAC")

    def test_dang_viet_san_di_qua_nguyen_ven(self):
        for text in ["192.168.10.25", "2001:db8:100::25", "operations@coastguard.vn"]:
            self.assertEqual(ElectronicParser()(text).normalized, text.lower()
                             if "@" in text else text)

    def test_mac_viet_hoa(self):
        self.assertEqual(ElectronicParser()("00:1a:2b:3c:4d:5e").normalized,
                         "00:1A:2B:3C:4D:5E")

    def test_dung_tu_cach_doc(self):
        self.assertEqual(
            ElectronicParser()("vê kép vê kép vê kép chấm vinamarine chấm gov chấm vê en").normalized,
            "www.vinamarine.gov.vn")

    def test_ipv4_doc_thanh_tieng(self):
        self.assertEqual(
            ElectronicParser()("một chín hai chấm một sáu tám chấm mười chấm hai lăm").normalized,
            "192.168.10.25")

    def test_khong_khop_khuon_nao_thi_truot(self):
        self.assertFalse(ElectronicParser()("xin chào các bạn").valid)


if __name__ == "__main__":
    unittest.main()
