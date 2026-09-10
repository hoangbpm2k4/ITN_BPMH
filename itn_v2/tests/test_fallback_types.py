"""Cổng dự phòng giữa các kiểu cùng hình dạng dạng nói (pipeline.FALLBACK_TYPES).

Mô hình hay lẫn MMSI_ID với TELEPHONE vì cả hai đều là chuỗi chữ số trần. Trước
đây validator bác đúng rồi pipeline giữ dạng nói — mất trắng 19 số trong bộ test
dù parser đọc được. Cổng dự phòng chỉ được lùi sang kiểu CÓ validator riêng, nếu
không nó sẽ thành đường bịa giá trị.
"""

import unittest

from itn_v2.pipeline import FALLBACK_TYPES, process_span
from itn_v2.validators import has_validator


def run(text, type_name):
    words = text.split()
    boundaries = ["S"] if len(words) == 1 else ["B"] + ["I"] * (len(words) - 2) + ["E"]
    return process_span(words, type_name, 0, len(words) - 1, boundaries)


class TestFallbackTypes(unittest.TestCase):
    def test_chi_lui_sang_kieu_co_validator(self):
        for source, targets in FALLBACK_TYPES.items():
            for target in targets:
                self.assertTrue(has_validator(target),
                                f"{source} -> {target}: kiểu đích không có validator")

    def test_mmsi_nham_thanh_dien_thoai_noi_dia(self):
        out = run("không chín không tám một hai ba bốn năm sáu", "MMSI_ID")
        self.assertTrue(out.emitted)
        self.assertEqual(out.normalized, "0908123456")
        self.assertEqual(out.resolved_type, "TELEPHONE")
        self.assertEqual(out.predicted_type, "MMSI_ID")  # vết gỡ lỗi giữ nguyên

    def test_mmsi_that_van_la_mmsi(self):
        out = run("hai bảy ba ba năm bốn ba hai một", "MMSI_ID")
        self.assertTrue(out.emitted)
        self.assertEqual(out.normalized, "273354321")
        self.assertEqual(out.resolved_type, "MMSI_ID")

    def test_khong_kieu_nao_hop_le_thi_giu_dang_noi(self):
        out = run("bảy tám chín", "MMSI_ID")
        self.assertFalse(out.emitted)


class TestTelephoneShapes(unittest.TestCase):
    def test_quoc_te_doc_ca_tu_cong(self):
        out = run("cộng tám bốn chín không tám một hai ba bốn năm sáu", "TELEPHONE")
        self.assertTrue(out.emitted)
        self.assertEqual(out.normalized, "+84 908 123 456")

    def test_dau_so_dich_vu(self):
        out = run("một chín không không một hai ba bốn", "TELEPHONE")
        self.assertTrue(out.emitted)
        self.assertEqual(out.normalized, "1900 1234")

    def test_noi_dia_muoi_mot_chu_so(self):
        out = run("không hai tám ba tám hai ba bốn năm sáu bảy", "TELEPHONE")
        self.assertTrue(out.emitted)
        self.assertEqual(out.normalized, "02838234567")

    def test_thieu_tu_cong_thi_khong_bia_ra_dau_cong(self):
        # Biên span cắt mất "cộng" là lỗi của mô hình; parser không được đoán bù.
        out = run("tám bốn chín không tám một hai ba bốn năm sáu", "TELEPHONE")
        self.assertFalse(out.emitted)


if __name__ == "__main__":
    unittest.main()
