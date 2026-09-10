"""Các cách đọc mà bộ v6 và bản gốc thật dùng, trước đây parser đều trượt.

Mỗi ca ở đây từng làm hỏng cả một chủ đề trong bộ test: hai chủ đề vũ khí hỏng
94-97%, văn bản pháp lý hỏng 100%, địa chỉ mạng hỏng 80%.
"""

import unittest

from itn_v2.registry import get_normalizer


def norm(type_name, text):
    return get_normalizer(type_name)(text, type_name)


class TestEquipmentSpokenDash(unittest.TestCase):
    """Số hiệu khí tài đọc CẢ dấu gạch, và tiền tố đọc theo tên chữ cái."""

    CASES = [
        ("ét u gạch ngang hai hai", "Su-22"),
        ("ét u gạch hai mươi hai", "Su-22"),
        ("su gạch nối hai mươi hai", "Su-22"),
        ("a ca gạch ngang bốn bảy", "AK-47"),
        ("a ca gạch ngang sáu ba không", "AK-630"),
        ("ép gạch ngang một sáu", "F-16"),
        ("ca hát gạch ngang ba năm", "Kh-35"),
        ("em i gạch ngang một bảy", "Mi-17"),
        ("pê gạch ngang tám không không", "P-800"),
        ("tê gạch ngang chín không", "T-90"),
        ("ét u gạch ngang ba không em ca hai", "Su-30MK2"),
    ]

    def test_doc_ca_dau_gach(self):
        for spoken, want in self.CASES:
            with self.subTest(spoken=spoken):
                self.assertEqual(norm("EQUIPMENT_ID", spoken).normalized, want)

    def test_khong_doc_dau_gach_van_chay(self):
        self.assertEqual(norm("EQUIPMENT_ID", "su ba mươi").normalized, "Su-30")


class TestLegalDocumentReadings(unittest.TestCase):
    """Ký hiệu đọc theo NGHĨA, dấu / đọc là "năm", số hiệu có hàng trăm."""

    CASES = [
        ("số hai mươi hai năm hai nghìn không trăm hai mươi tư nghị định chính phủ",
         "Số 22/2024/NĐ-CP"),
        ("số mười lăm năm hai nghìn không trăm hai mươi lăm thông tư bộ tài chính",
         "Số 15/2025/TT-BTC"),
        ("số chín năm hai nghìn không trăm hai mươi lăm thông tư bộ quốc phòng",
         "Số 9/2025/TT-BQP"),
        ("số một trăm sáu mươi lăm xẹt hai nghìn mười chín en quy cê pê",
         "Số 165/2019/NQ-CP"),
        # lối đánh vần cũ phải giữ nguyên hành vi
        ("số mười hai xẹt hai không hai tư nờ đê cê pê", "Số 12/2024/NĐ-CP"),
    ]

    def test_cac_loi_doc(self):
        for spoken, want in self.CASES:
            with self.subTest(spoken=spoken):
                self.assertEqual(norm("LEGAL_DOC_ID", spoken).normalized, want)

    def test_nam_la_chu_so_khi_khong_o_giua_hai_cum_so(self):
        # "năm" đứng đầu cụm là chữ số 5, không phải dấu phân cách.
        self.assertEqual(norm("DOCUMENT_ID", "số năm xẹt hai không hai tư").normalized,
                         "Số 5/2024")


class TestProtocolLabels(unittest.TestCase):
    def test_ten_giao_thuc_dung_mot_minh(self):
        for spoken, want in [("i p v bốn", "IPv4"), ("i p v sáu", "IPv6"),
                             ("i p phiên bản sáu", "IPv6"),
                             ("giao thức i p phiên bản sáu", "IPv6"),
                             ("i p vê bốn", "IPv4")]:
            with self.subTest(spoken=spoken):
                self.assertEqual(norm("ELECTRONIC", spoken).normalized, want)

    def test_dia_chi_that_van_chay(self):
        got = norm("ELECTRONIC", "một trăm chín mươi hai chấm một trăm sáu mươi "
                                 "tám chấm mười chấm hai mươi lăm")
        self.assertEqual(got.normalized, "192.168.10.25")


class TestDurationNamAmbiguity(unittest.TestCase):
    """"năm" vừa là đơn vị (year) vừa là chữ số (5)."""

    def test_nam_la_chu_so_trong_phut(self):
        self.assertEqual(norm("DURATION", "mười một giờ năm mươi lăm phút").normalized,
                         "11 giờ 55 phút")
        self.assertEqual(norm("DURATION", "mười hai giờ năm phút").normalized,
                         "12 giờ 5 phút")

    def test_nam_van_la_don_vi_khi_dung_sau_so(self):
        self.assertEqual(norm("DURATION", "hai năm ba tháng").normalized,
                         "2 năm 3 tháng")


class TestPlateSeparatorWithDau(unittest.TestCase):
    def test_dau_cham(self):
        self.assertEqual(
            norm("VEHICLE_PLATE", "bảy bốn gờ gạch chín chín bảy dấu chấm ba sáu").normalized,
            "74G-997.36")


if __name__ == "__main__":
    unittest.main()
