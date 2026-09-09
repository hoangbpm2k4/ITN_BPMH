import unittest

from itn_v2.normalizers.date import (DateParser, DurationParser, ETAParser,
                                     ETDParser, QuarterParser, TimeParser,
                                     TimezoneParser)


class TestDate(unittest.TestCase):
    def test_khong_doi_tu_ngay_trong_span(self):
        """Lỗi V1 nặng nhất: span thiếu từ 'ngày' thì mất luôn phần ngày.

        Đo trên 60.000 câu huấn luyện V1: 23,9% số cụm DATE rơi vào đây.
        """
        self.assertEqual(DateParser()("hai mươi tám tháng bốn").normalized, "28/04")
        self.assertEqual(DateParser()("mười lăm tháng tám").normalized, "15/08")
        self.assertEqual(DateParser()("mười sáu tháng chín").normalized, "16/09")

    def test_co_tu_ngay_van_dung(self):
        self.assertEqual(DateParser()("ngày hai mươi tám tháng bốn").normalized, "28/04")
        self.assertEqual(DateParser()("ngày mùng một tháng năm").normalized, "01/05")
        self.assertEqual(DateParser()("mồng chín tháng chín").normalized, "09/09")

    def test_day_du_ngay_thang_nam(self):
        self.assertEqual(
            DateParser()("mười lăm tháng tám năm hai không hai sáu").normalized,
            "15/08/2026")

    def test_thang_nam_khong_bi_nham_la_moc_nam(self):
        self.assertEqual(DateParser()("ngày mùng một tháng năm").normalized, "01/05")

    def test_chi_thang_va_nam(self):
        self.assertEqual(
            DateParser()("tháng mười hai năm một nghìn chín trăm tám mươi").normalized,
            "12/1980")

    def test_gia_tri_ngoai_mien_thi_truot(self):
        self.assertFalse(DateParser()("bốn mươi tháng bốn").valid)
        self.assertFalse(DateParser()("mười lăm tháng mười ba").valid)


class TestTime(unittest.TestCase):
    def test_gio_phut(self):
        self.assertEqual(TimeParser()("sáu giờ bốn mươi lăm").normalized, "06:45")
        self.assertEqual(TimeParser()("mười bốn giờ ba mươi").normalized, "14:30")

    def test_chi_co_gio(self):
        self.assertEqual(TimeParser()("sáu giờ").normalized, "06:00")

    def test_ruoi(self):
        self.assertEqual(TimeParser()("sáu giờ rưỡi").normalized, "06:30")

    def test_bien(self):
        self.assertEqual(TimeParser()("không giờ").normalized, "00:00")
        self.assertFalse(TimeParser()("hai mươi lăm giờ").valid)
        self.assertFalse(TimeParser()("sáu giờ bảy mươi").valid)


class TestOtherDatetime(unittest.TestCase):
    def test_timezone(self):
        self.assertEqual(TimezoneParser()("u tê xê cộng bảy").normalized, "UTC+7")
        self.assertEqual(TimezoneParser()("u tê xê trừ năm").normalized, "UTC-5")

    def test_eta_etd(self):
        self.assertEqual(ETAParser()("ê tê a tám giờ mười lăm").normalized, "ETA 08:15")
        self.assertEqual(ETDParser()("ê tê đê mười sáu giờ bốn mươi").normalized, "ETD 16:40")

    def test_quarter(self):
        self.assertEqual(QuarterParser()("quý ba").normalized, "Quý III")
        self.assertEqual(QuarterParser()("quý tư").normalized, "Quý IV")
        self.assertFalse(QuarterParser()("quý năm").valid)

    def test_duration(self):
        self.assertEqual(DurationParser()("hai giờ ba mươi phút").normalized, "2 giờ 30 phút")


if __name__ == "__main__":
    unittest.main()


class TestMonthYearNoMarker(unittest.TestCase):
    """'tháng <tháng> <năm>' không có từ dẫn 'năm' — dạng nói phổ biến nhất.

    Trước khi vá, 239 cụm DATE của corpus v1 bị ParseError rồi loại âm thầm.
    """

    def setUp(self):
        self.p = DateParser()

    def test_thang_nam_bon_chu_so(self):
        self.assertEqual(self.p("tháng sáu một nghìn chín trăm sáu mươi tám").normalized, "06/1968")
        self.assertEqual(self.p("tháng ba một nghìn chín trăm bảy mươi lăm").normalized, "03/1975")

    def test_thang_nam_doc_tung_chu_so(self):
        self.assertEqual(self.p("tháng mười hai một chín bảy hai").normalized, "12/1972")
        self.assertEqual(self.p("tháng năm hai không hai sáu").normalized, "05/2026")

    def test_khong_tach_nham(self):
        # 'tháng năm' đứng một mình vẫn là tháng 5, không phải mốc năm.
        self.assertEqual(self.p("tháng năm").normalized, "05")
        # Không có đuôi năm hợp lệ thì phải hỏng, không được đoán bừa.
        self.assertFalse(self.p("tháng sáu mươi tám").valid)

    def test_dang_day_du_khong_doi(self):
        self.assertEqual(
            self.p("ngày ba mươi tháng tư năm một nghìn chín trăm bảy mươi lăm").normalized,
            "30/04/1975")
        self.assertEqual(self.p("hai mươi tám tháng bốn").normalized, "28/04")


class TestNamMuoiKhongPhaiMocNam(unittest.TestCase):
    """'năm' trong 'năm mươi' là chữ số 5, không phải từ dẫn mốc năm.

    Trước khi vá, DateParser cắt câu tại đó và hỏng với mọi năm thập niên 50.
    """

    def setUp(self):
        self.p = DateParser()

    def test_thap_nien_50(self):
        self.assertEqual(self.p("một nghìn chín trăm năm mươi ba").normalized, "1953")
        self.assertEqual(self.p("một nghìn chín trăm năm mươi lăm").normalized, "1955")
        self.assertEqual(self.p("năm hai nghìn không trăm năm mươi").normalized, "2050")

    def test_van_nhan_moc_nam_that(self):
        self.assertEqual(self.p("năm một nghìn chín trăm sáu mươi chín").normalized, "1969")
        self.assertEqual(self.p("tháng ba năm một nghìn chín trăm năm mươi tư").normalized, "03/1954")
        self.assertEqual(
            self.p("ngày hai tháng chín năm một nghìn chín trăm bốn mươi lăm").normalized,
            "02/09/1945")


class TestNgayThangNamVietLien(unittest.TestCase):
    """'ngày tháng năm' viết liền, không có từ dẫn nào.

    read_year lặng lẽ bỏ qua phần đầu và chỉ trả về năm, nên 100 cụm DATE của
    corpus mất hẳn ngày lẫn tháng mà không báo lỗi.
    """

    def setUp(self):
        self.p = DateParser()

    def test_khong_tu_dan(self):
        self.assertEqual(self.p("tám chín một nghìn chín trăm sáu mươi chín").normalized,
                         "08/09/1969")
        self.assertEqual(self.p("mười một bảy một nghìn chín trăm bảy mươi lăm").normalized,
                         "11/07/1975")
        self.assertEqual(self.p("một bảy hai nghìn không trăm hai mươi lăm").normalized,
                         "01/07/2025")

    def test_uu_tien_phan_ngay_dai_nhat(self):
        # "hai mươi mười" là 20/10; cắt ngắn hơn sẽ xé đôi "hai mươi" thành 2.
        self.assertEqual(
            self.p("ngày hai mươi mười một nghìn chín trăm tám mươi tư").normalized,
            "20/10/1984")
        self.assertEqual(
            self.p("ngày ba mươi tám một nghìn chín trăm tám mươi tư").normalized,
            "30/08/1984")

    def test_khong_pha_nam_dung_mot_minh(self):
        self.assertEqual(self.p("một nghìn chín trăm sáu mươi tám").normalized, "1968")
        self.assertEqual(self.p("hai nghìn không trăm hai mươi lăm").normalized, "2025")


class TestDocSoPhaiTieuThuHetTu(unittest.TestCase):
    """Phép cắt ngày/tháng không được để sót từ.

    read_number_auto đọc ["sáu","mười"] ra 10 và ["ba","mươi","mốt","mười"]
    cũng ra 10 — âm thầm bỏ từ thừa. Dựa vào nó thì phép cắt chọn nhầm mốc mà
    không hề báo lỗi, cho 10/01/1978 thay vì 06/11/1978.
    """

    def setUp(self):
        self.p = DateParser()

    def test_khong_nham_moc(self):
        self.assertEqual(self.p("sáu mười một một nghìn chín trăm bảy mươi tám").normalized,
                         "06/11/1978")
        self.assertEqual(
            self.p("ba mươi mốt mười hai một nghìn chín trăm sáu mươi bảy").normalized,
            "31/12/1967")

    def test_thu_nhieu_cach_doc(self):
        # "ba mươi tám" vừa là 38 vừa có thể là 30 rồi 8; chỉ cách sau ghép
        # được với phần năm còn lại.
        self.assertEqual(
            self.p("ngày ba mươi tám một nghìn chín trăm tám mươi tư").normalized,
            "30/08/1984")

    def test_thang_hai_tu_khong_bi_cat_ngan(self):
        self.assertEqual(self.p("tháng mười hai một nghìn chín trăm bảy mươi tám").normalized,
                         "12/1978")
