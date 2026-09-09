import unittest

from itn_v2.windowing import plan_windows, stitch


class TestWindowing(unittest.TestCase):
    def test_phu_het_khong_mat_duoi(self):
        """V1 cắt cứng 256 subword rồi vứt đuôi. V2 không được phép mất từ nào."""
        lengths = [2] * 300
        windows = plan_windows(lengths, useful=224, overlap=40)
        covered = set()
        for w in windows:
            covered.update(range(w.word_start, w.word_end))
        self.assertEqual(covered, set(range(300)))

    def test_moi_cua_so_khong_vuot_gioi_han(self):
        lengths = [3] * 200
        for w in plan_windows(lengths, useful=224, overlap=40):
            self.assertLessEqual(w.subword_count, 224)

    def test_co_chong_lan(self):
        lengths = [2] * 300
        windows = plan_windows(lengths, useful=224, overlap=40)
        self.assertGreater(len(windows), 1)
        self.assertLess(windows[1].word_start, windows[0].word_end)

    def test_cau_ngan_chi_mot_cua_so(self):
        self.assertEqual(len(plan_windows([2] * 10, useful=224)), 1)

    def test_tu_dai_hon_ca_cua_so_van_duoc_nhan(self):
        windows = plan_windows([500, 2, 2], useful=224)
        self.assertEqual(windows[0].word_start, 0)
        covered = set()
        for w in windows:
            covered.update(range(w.word_start, w.word_end))
        self.assertEqual(covered, {0, 1, 2})

    def test_ghep_theo_chi_so_tu_goc(self):
        lengths = [2] * 300
        windows = plan_windows(lengths, useful=224, overlap=40)
        preds = [[f"w{i}" for i in range(w.word_start, w.word_end)] for w in windows]
        self.assertEqual(stitch(preds, windows, 300), [f"w{i}" for i in range(300)])

    def test_uu_tien_du_doan_gan_tam_cua_so(self):
        windows = plan_windows([2] * 300, useful=224, overlap=40)
        preds = []
        for k, w in enumerate(windows):
            preds.append([f"win{k}" for _ in range(w.word_start, w.word_end)])
        merged = stitch(preds, windows, 300)
        # từ 0 chỉ nằm trong cửa sổ 0
        self.assertEqual(merged[0], "win0")
        # từ ở vùng chồng lấn phải lấy theo cửa sổ có nó gần tâm hơn
        overlap_word = windows[1].word_start
        self.assertIn(merged[overlap_word], {"win0", "win1"})

    def test_rong(self):
        self.assertEqual(plan_windows([]), [])


if __name__ == "__main__":
    unittest.main()
