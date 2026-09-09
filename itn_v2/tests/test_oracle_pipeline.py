"""Kiểm tra oracle (spec §29) + các quy tắc kết xuất (spec §16, §17, §18).

Sáu lỗi V1 mà spec §29 liệt kê là BẮT BUỘC phải biến mất đều có test ở đây.
"""

import unittest

from .helpers import run


class TestV1FailuresMustDisappear(unittest.TestCase):
    def test_date_khong_mat_ngay_khi_thieu_tu_khoa(self):
        out, spans, _, _ = run("đến hai mươi tám tháng bốn",
                               ("hai mươi tám tháng bốn", "DATE"))
        self.assertEqual(spans[0].normalized, "28/04")
        self.assertEqual(out, "Đến 28/04.")

    def test_nhan_coord_moi_quyet_dinh_chuyen_doi(self):
        """Có nhãn thì chuyển; KHÔNG có nhãn thì giữ nguyên.

        V1 dựng toạ độ bằng regex quét toàn câu nên nhãn COORD vô tác dụng và
        văn bản không được gắn nhãn vẫn bị sửa.
        """
        co_nhan, _, _, _ = run("vị trí mười chín độ hai mươi phút bắc",
                               ("mười chín độ hai mươi phút bắc", "COORD"))
        self.assertEqual(co_nhan, "Vị trí 19°20'N.")
        khong_nhan, _, _, _ = run("vị trí mười chín độ hai mươi phút bắc")
        self.assertEqual(khong_nhan, "Vị trí mười chín độ hai mươi phút bắc.")

    def test_quan_ham_khong_bi_ha_chu(self):
        for rank in ["đại tá", "trung úy", "thiếu tá", "trung tướng"]:
            out, _, _, _ = run(f"gặp {rank}", (rank, "RANK"))
            self.assertEqual(out, f"Gặp {rank[0].upper()}{rank[1:]}.", rank)

    def test_khi_tai_giu_hoa_thuong_chuan(self):
        out, _, _, _ = run("biên đội su ba mươi xuất kích", ("su ba mươi", "EQUIPMENT_ID"))
        self.assertIn("Su-30", out)
        self.assertNotIn("SU-30", out)

    def test_dau_cau_khong_quyet_dinh_ranh_gioi_span(self):
        """Hai cụm DATE liền nhau, KHÔNG có dấu câu ngăn cách, vẫn phải tách."""
        text = "từ mười hai tháng năm đến mười ba tháng năm"
        out, spans, _, _ = run(text,
                               ("mười hai tháng năm", "DATE"),
                               ("mười ba tháng năm", "DATE"))
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].normalized, "12/05")
        self.assertEqual(spans[1].normalized, "13/05")
        self.assertEqual(out, "Từ 12/05 đến 13/05.")

    def test_viet_hoa_dau_cau_sau_dau_cham_giua_doan(self):
        out, _, _, _ = run("hôm nay trời đẹp chúng tôi đi học",
                           punct={3: "PERIOD"})
        self.assertEqual(out, "Hôm nay trời đẹp. Chúng tôi đi học.")


class TestSafeFallback(unittest.TestCase):
    def test_validator_truot_thi_giu_dang_noi(self):
        out, spans, _, _ = run("tàu đi hướng ba bảy không", ("ba bảy không", "HEADING"))
        self.assertFalse(spans[0].emitted)
        self.assertEqual(out, "Tàu đi hướng ba bảy không.")

    def test_duoi_nguong_tin_cay_thi_giu_dang_noi(self):
        from itn_v2.config import Config
        from itn_v2.pipeline import process_span
        so = process_span("không chín không".split(), "HEADING", 0, 2, ["B", "I", "E"],
                          boundary_confidence=0.5, type_confidence=0.99, config=Config())
        self.assertTrue(so.valid)          # parser + validator đều đạt
        self.assertFalse(so.emitted)       # nhưng tin cậy dưới ngưỡng lớp
        self.assertIn("ngưỡng", so.reason)

    def test_ngoai_danh_muc_thi_giu_dang_noi(self):
        out, spans, _, _ = run("hệ thống cờ lô ri ban đờ sẵn sàng",
                               ("cờ lô ri ban đờ", "FOREIGN_NAME"))
        self.assertFalse(spans[0].emitted)
        self.assertIn("cờ lô ri ban đờ", out)


class TestRenderingPolicy(unittest.TestCase):
    def test_span_da_chuan_hoa_duoc_bao_ve(self):
        """Không có luật dọn nào được sửa nội dung span đã chuẩn hoá (spec §18)."""
        out, spans, _, _ = run("tàu mang số hiệu em ca bốn sáu",
                               ("em ca bốn sáu", "EQUIPMENT_ID"))
        self.assertIn("MK 46", out)

    def test_khong_co_khoang_trang_truoc_dau_cau(self):
        out, _, _, _ = run("đạt hai mươi lăm phần trăm",
                           ("hai mươi lăm phần trăm", "PERCENT"))
        self.assertTrue(out.endswith("25%."))
        self.assertNotIn(" .", out)

    def test_span_ngay_truoc_dau_cau(self):
        out, spans, _, _ = run("hướng không chín không tốc độ tăng",
                               ("không chín không", "HEADING"),
                               punct={3: "COMMA"})
        self.assertEqual(out, "Hướng 090°, tốc độ tăng.")

    def test_gach_duoi_cua_tu_ghep_khong_lot_ra_dau_ra(self):
        out, _, _, _ = run("tàu chở hàng đang neo đậu tại cảng hải phòng",
                           ("hải phòng", "LOCATION_NAME"))
        self.assertNotIn("_", out)
        self.assertEqual(out, "Tàu chở hàng đang neo đậu tại cảng Hải Phòng.")


class TestOracleDeterministicTypes(unittest.TestCase):
    def test_oracle_dat_100_phan_tram_tren_lop_tat_dinh(self):
        """Với nhãn vàng, các lớp tất định phải ra đúng tuyệt đối (spec §29)."""
        from .test_catalog_33_groups import GROUPS
        fails = []
        for number, name, text, pairs, expected in GROUPS:
            out, _, _, _ = run(text, *pairs)
            if out != expected:
                fails.append(f"nhóm {number} ({name}): {out!r} != {expected!r}")
        self.assertEqual(fails, [], "\n".join(fails))


if __name__ == "__main__":
    unittest.main()
