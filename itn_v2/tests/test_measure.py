import unittest

from itn_v2.normalizers.measure import (ChannelParser, DepthParser,
                                        DistanceParser, DraftParser,
                                        FrequencyParser, MeasureParser,
                                        MoneyParser, SpeedParser)


class TestMeasure(unittest.TestCase):
    def test_cac_vi_du_bat_buoc_cua_spec(self):
        cases = {
            "bốn phẩy năm bar": "4,5 bar",
            "âm mười độ xê": "-10°C",
            "tám mươi lăm tấn một giờ": "85 t/h",
            "một trăm hai mươi mét khối một giờ": "120 m³/h",
            "hai mươi bốn vôn": "24 V",
        }
        for spoken, expected in cases.items():
            self.assertEqual(MeasureParser()(spoken).normalized, expected, spoken)

    def test_don_vi_dai(self):
        self.assertEqual(MeasureParser()("hai mươi lăm ki lô mét").normalized, "25 km")
        self.assertEqual(MeasureParser()("năm mét vuông").normalized, "5 m²")

    def test_don_vi_khong_nhan_ra_thi_truot(self):
        r = MeasureParser()("hai mươi lăm quả")
        self.assertFalse(r.valid)
        self.assertIn("đơn vị", r.reason)

    def test_thieu_phan_luong(self):
        self.assertFalse(MeasureParser()("ki lô mét").valid)


class TestMaritimeMeasure(unittest.TestCase):
    def test_speed_doi_hai_ly_mot_gio_thanh_nut(self):
        self.assertEqual(SpeedParser()("mười hai hải lý một giờ").normalized, "12 kn")

    def test_distance_depth_draft(self):
        self.assertEqual(DistanceParser()("mười hải lý").normalized, "10 NM")
        self.assertEqual(DraftParser()("tám phẩy hai mét").normalized, "8,2 m")
        self.assertEqual(DepthParser()("ba mươi mét").normalized, "30 m")

    def test_frequency(self):
        self.assertEqual(FrequencyParser()("một hai một phẩy năm mê ga héc").normalized,
                         "121,5 MHz")

    def test_don_vi_sai_kieu_thi_truot(self):
        """SPEED không được nhận đơn vị khối lượng."""
        self.assertFalse(SpeedParser()("mười hai ki lô gam").valid)

    def test_channel(self):
        self.assertEqual(ChannelParser()("kênh mười sáu").normalized, "kênh 16")
        self.assertEqual(ChannelParser()("mười hai").normalized, "12")


class TestMoney(unittest.TestCase):
    def test_tien(self):
        self.assertEqual(MoneyParser()("hai triệu đồng").normalized, "2.000.000 đồng")
        self.assertEqual(MoneyParser()("năm trăm nghìn đồng").normalized, "500.000 đồng")

    def test_khong_co_don_vi_tien_thi_truot(self):
        self.assertFalse(MoneyParser()("hai triệu").valid)


if __name__ == "__main__":
    unittest.main()
