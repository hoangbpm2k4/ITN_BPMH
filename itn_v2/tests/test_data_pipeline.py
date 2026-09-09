"""Test tầng dữ liệu: phân tích đánh dấu, lược đồ, chia tập, Dataset."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from itn_v2.data.build_splits import build_splits, check_leakage, template_signature
from itn_v2.data.build_v1_pairs import pick_variant
from itn_v2.data.label_pairs_llm_align import (comparison_form,
                                                parse_spans, span_is_sane)
from itn_v2.data.markup import harvest_lines, parse_markup
from itn_v2.data.schema import Sample, read_jsonl, validate_sample, write_jsonl
from itn_v2.dataset import ITNv2Dataset, collate
from itn_v2.labels import BOUNDARY2ID, TYPE2ID


def make_sample(idx, spoken, spans, punct=None, leak_key="Su-35"):
    words = spoken.split()
    return Sample(id=str(idx), spoken=spoken, written="", words=words,
                  spans=spans, punct=punct or ["O"] * len(words),
                  types=sorted({t for _, _, t in spans}), leak_key=leak_key)


class TestMarkup(unittest.TestCase):
    def test_thu_hoach_bo_qua_van_ve_xung_quanh(self):
        text = ("*   Sentence 1:\n"
                "    Line 1: tàu đi hướng [[không chín không|HEADING]] .\n"
                "    dòng không có đánh dấu\n"
                "1) biên đội [[su ba mươi|EQUIPMENT_ID]] xuất kích .\n")
        self.assertEqual(len(harvest_lines(text)), 2)

    def test_phan_tich_dung_chi_so(self):
        words, spans, punct = parse_markup(
            "tàu đi hướng [[không chín không|HEADING]] , tốc độ tăng .")
        self.assertEqual(words[3:6], ["không", "chín", "không"])
        self.assertEqual(spans, [(3, 5, "HEADING")])
        self.assertEqual(punct[5], "COMMA")
        self.assertEqual(punct[-1], "PERIOD")

    def test_dau_cau_dinh_lien_tu(self):
        words, _, punct = parse_markup("[[quý ba|QUARTER]] đã xong.")
        self.assertEqual(punct[-1], "PERIOD")
        self.assertEqual(words[-1], "xong")

    def test_danh_dau_lech_thi_bo(self):
        self.assertEqual(harvest_lines("tàu [[hỏng|TYPE] ."), [])


class TestSchema(unittest.TestCase):
    def test_bat_chu_so_trong_dang_noi(self):
        s = make_sample(1, "tàu đi hướng 090", [(3, 3, "HEADING")])
        self.assertIn("chữ số", " ".join(validate_sample(s)))

    def test_bat_span_chong_lan(self):
        s = make_sample(2, "một hai ba bốn", [(0, 2, "CARDINAL"), (1, 3, "CARDINAL")])
        self.assertTrue(any("chồng lấn" in e for e in validate_sample(s)))

    def test_bat_kieu_ngoai_taxonomy(self):
        s = make_sample(3, "một hai", [(0, 1, "KHONG_CO_KIEU_NAY")])
        self.assertTrue(any("taxonomy" in e for e in validate_sample(s)))

    def test_ghi_va_doc_lai_khop(self):
        s = make_sample(4, "tàu đi hướng không chín không", [(3, 5, "HEADING")])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.jsonl"
            write_jsonl(path, [s])
            back = list(read_jsonl(path))[0]
        self.assertEqual(back.spans, s.spans)
        self.assertEqual(back.words, s.words)


class TestV1Alignment(unittest.TestCase):
    def test_bien_the_cung_phai_ha_chng_nhu_asr(self):
        chunk = {
            "text": "Norng Chan Phal sinh năm 1979.",
            "all_text_results": [{
                "type": "pronunciation_variant",
                "text": "Norng Chan Phal sinh năm một nghìn chín trăm bảy mươi chín.",
            }],
        }
        spoken, _ = pick_variant(chunk)
        self.assertEqual(spoken, spoken.lower())
        self.assertIn("norng chan phal", spoken)

    def test_so_sanh_khong_duoc_nuot_case_ten_rieng(self):
        self.assertNotEqual(comparison_form("việt nam."),
                            comparison_form("Việt Nam."))
        self.assertNotEqual(comparison_form("bà chan phal."),
                            comparison_form("Bà Chan Phal."))

    def test_so_sanh_cho_phep_viet_hoa_dau_cau(self):
        self.assertEqual(comparison_form("bà đang nói."),
                         comparison_form("Bà đang nói."))

    def test_loc_ba_nhung_giu_chan_phal_o_dau_cau(self):
        stats = Counter()
        spans, _ = parse_spans(
            [{"noi": "bà", "viet": "Bà", "kieu": "PERSON_NAME"}],
            ["bà", "kể"], ["O", "PERIOD"], stats)
        self.assertEqual(spans, [])

        spans, _ = parse_spans(
            [{"noi": "chan phal", "viet": "Chan Phal", "kieu": "PERSON_NAME"}],
            ["chan", "phal", "kể"], ["O", "O", "PERIOD"], Counter())
        self.assertEqual(spans, [(0, 1, "PERSON_NAME")])

    def test_span_giong_het_nhau_phai_bi_loai(self):
        ok, why = span_is_sane("bà", "bà", "PERSON_NAME")
        self.assertFalse(ok)
        self.assertIn("giống hệt", why)


class TestSplits(unittest.TestCase):
    def setUp(self):
        # Dữ liệu giả phải ĐA DẠNG cả giá trị chuẩn lẫn khung câu, nếu không mọi
        # câu dính vào một thành phần liên thông và phép chia mất ý nghĩa.
        self.samples = []
        frames = ["biên đội {} xuất kích", "phát hiện {} tại khu vực",
                  "đơn vị tiếp nhận {} hôm nay", "báo cáo về {} đã gửi",
                  "quan sát thấy {} ngoài khơi", "ghi nhận {} lúc rạng sáng"]
        for i in range(60):
            frame = frames[i % len(frames)]
            words = frame.format("su ba mươi").split()
            start = words.index("su")
            self.samples.append(Sample(
                id=str(i), spoken=" ".join(words), written="", words=words,
                spans=[(start, start + 2, "EQUIPMENT_ID")],
                punct=["O"] * len(words), types=["EQUIPMENT_ID"],
                leak_key=f"Su-{30 + i % 10}"))

    def test_khong_ro_ri_bien_the_giua_cac_tap(self):
        splits = build_splits(self.samples)
        self.assertEqual(check_leakage(splits), [])

    def test_entity_holdout_co_thuc_the_khong_thay_o_train(self):
        splits = build_splits(self.samples)
        train_keys = {s.leak_key for s in splits["train"]}
        for s in splits["entity_holdout_dev"]:
            self.assertNotIn(s.leak_key, train_keys)

    def test_template_holdout_co_khung_cau_khong_thay_o_train(self):
        splits = build_splits(self.samples)
        train_sigs = {template_signature(s) for s in splits["train"]}
        for s in splits["template_holdout_dev"]:
            self.assertNotIn(template_signature(s), train_sigs)

    def test_du_bon_tap_sinh_tu_dong(self):
        splits = build_splits(self.samples)
        self.assertEqual(set(splits), {"train", "random_dev",
                                       "entity_holdout_dev", "template_holdout_dev"})
        for name, items in splits.items():
            self.assertGreater(len(items), 0, name)

    def test_chu_ky_khung_cau_bo_noi_dung_span(self):
        s = make_sample(1, "biên đội su ba mươi xuất kích", [(2, 4, "EQUIPMENT_ID")])
        self.assertEqual(template_signature(s), "biên đội <EQUIPMENT_ID> xuất kích")


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "d.jsonl"
        samples = [
            make_sample(1, "tàu đi hướng không chín không", [(3, 5, "HEADING")]),
            make_sample(2, "biên đội xu ba lăm xuất kích", [(2, 4, "EQUIPMENT_ID")]),
        ]
        samples[0].punct[-1] = "PERIOD"
        write_jsonl(path, samples)
        cls.ds = ITNv2Dataset(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_nap_duoc(self):
        self.assertEqual(len(self.ds), 2)

    def test_nhan_ranh_gioi_dung_BIOES(self):
        item = self.ds[0]
        tags = item["boundary"]
        span = item["model_spans"][0]
        self.assertEqual(tags[span[0]], BOUNDARY2ID["B"])
        self.assertEqual(tags[span[1]], BOUNDARY2ID["E"])
        self.assertEqual(tags[0], BOUNDARY2ID["O"])

    def test_nhan_kieu_phu_het_span(self):
        item = self.ds[0]
        ws, we = item["model_spans"][0]
        for w in range(ws, we + 1):
            self.assertEqual(item["types"][w], TYPE2ID["HEADING"])
        self.assertEqual(item["types"][0], 0)

    def test_collate_dem_dung_hai_truc(self):
        batch = collate([self.ds[0], self.ds[1]])
        bs = 2
        self.assertEqual(batch["input_ids"].shape[0], bs)
        self.assertEqual(batch["pooling_matrix"].shape[0], bs)
        self.assertEqual(batch["pooling_matrix"].shape[1], batch["boundary_tags"].shape[1])
        self.assertEqual(batch["pooling_matrix"].shape[2], batch["input_ids"].shape[1])
        # vị trí đệm phải bị che
        self.assertTrue((batch["type_tags"][~batch["word_mask"]] == -100).all())

    def test_ma_tran_gop_cua_cau_ngan_hon_van_dung(self):
        batch = collate([self.ds[0], self.ds[1]])
        for i, item in enumerate(batch["batch"]):
            n_word = len(item["boundary"])
            rows = batch["pooling_matrix"][i, :n_word].sum(dim=-1)
            self.assertTrue(torch.allclose(rows, torch.ones(n_word), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
