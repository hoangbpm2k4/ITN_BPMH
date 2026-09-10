import unittest

from itn_v2.normalizers.entity import (AcronymResolver, EquipmentNameResolver,
                                       ForeignNameResolver, LocationFormatter,
                                       MaritimeTermResolver,
                                       PersonNameFormatter, RankFormatter)


class TestCatalogEntities(unittest.TestCase):
    def test_tra_dung(self):
        self.assertEqual(EquipmentNameResolver()("pa tri ốt").normalized, "Patriot")
        self.assertEqual(ForeignNameResolver()("oa sinh tơn").normalized, "Washington")
        self.assertEqual(AcronymResolver()("vê hát ép").normalized, "VHF")
        # Dạng chuẩn để chữ THƯỜNG: bản gốc thật viết "main engine", "slow
        # astern", "full ahead" thường khi nằm giữa câu (76 hoa / 124 thường).
        # Bước viết hoa đầu câu ở render() lo phần vị trí.
        self.assertEqual(MaritimeTermResolver()("đét xờ lâu a hét").normalized,
                         "dead slow ahead")

    def test_bien_the_phat_am(self):
        for spoken in ["pa tri ốt", "pat ri ốt", "ba tri ốt"]:
            self.assertEqual(EquipmentNameResolver()(spoken).normalized, "Patriot", spoken)

    def test_ngoai_danh_muc_thi_truot_de_giu_dang_noi(self):
        r = ForeignNameResolver()("cờ lô ri ban đờ")
        self.assertFalse(r.valid)
        self.assertIn("danh mục", r.reason)

    def test_khop_gan_dung_duoi_nguong_thi_truot(self):
        r = ForeignNameResolver(threshold=0.99)("oa sinh tân")
        self.assertFalse(r.valid)


class TestNameFormatters(unittest.TestCase):
    def test_ten_nguoi_hoa_moi_tu(self):
        self.assertEqual(PersonNameFormatter()("nguyễn văn hùng").normalized,
                         "Nguyễn Văn Hùng")

    def test_dia_danh(self):
        self.assertEqual(LocationFormatter()("hải phòng").normalized, "Hải Phòng")

    def test_quan_ham_hoa_chu_dau_cum(self):
        self.assertEqual(RankFormatter()("đại tá").normalized, "Đại tá")
        self.assertEqual(RankFormatter()("trung úy").normalized, "Trung úy")
        self.assertEqual(RankFormatter()("thiếu tá").normalized, "Thiếu tá")


if __name__ == "__main__":
    unittest.main()
