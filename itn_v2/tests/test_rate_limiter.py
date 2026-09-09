"""Test bộ tiết chế hạn mức API (không cần mạng)."""

import threading
import time
import unittest

from itn_v2.data.gemma_client import RateLimiter


class TestRateLimiter(unittest.TestCase):
    def test_chan_khi_vuot_so_luot_moi_phut(self):
        lim = RateLimiter(rpm=4, tpm=10 ** 9, safety=1.0)
        for _ in range(4):
            lim.acquire(1)
        start = time.monotonic()
        t = threading.Thread(target=lim.acquire, args=(1,), daemon=True)
        t.start()
        t.join(timeout=0.6)
        self.assertTrue(t.is_alive(), "lượt thứ 5 phải bị chặn lại")
        self.assertGreaterEqual(lim.waited, 0.0)
        self.assertLess(time.monotonic() - start, 2.0)

    def test_chan_khi_vuot_so_token_moi_phut(self):
        lim = RateLimiter(rpm=10 ** 6, tpm=1000, safety=1.0)
        lim.acquire(900)
        t = threading.Thread(target=lim.acquire, args=(900,), daemon=True)
        t.start()
        t.join(timeout=0.6)
        self.assertTrue(t.is_alive(), "vượt trần token thì phải chặn")

    def test_settle_thay_uoc_luong_bang_so_that(self):
        lim = RateLimiter(rpm=10, tpm=10_000, safety=1.0)
        lim.acquire(5000)
        lim.settle(5000, 300)
        used = sum(tok for _, tok in lim.events)
        self.assertEqual(used, 300)
        # sau khi trả lại phần thừa thì lượt tiếp theo phải đi qua ngay
        start = time.monotonic()
        lim.acquire(5000)
        self.assertLess(time.monotonic() - start, 0.5)

    def test_chan_han_muc_ngay(self):
        lim = RateLimiter(rpm=10 ** 6, tpm=10 ** 9, rpd=2, safety=1.0)
        lim.acquire(1)
        lim.acquire(1)
        with self.assertRaises(RuntimeError):
            lim.acquire(1)

    def test_he_so_an_toan_ha_tran_thuc_te(self):
        lim = RateLimiter(rpm=30, tpm=16_000, safety=0.85)
        self.assertEqual(lim.rpm, 25)
        self.assertEqual(lim.tpm, 13_600)

    def test_nhieu_luong_dung_chung_mot_bo_dem(self):
        lim = RateLimiter(rpm=6, tpm=10 ** 9, safety=1.0)
        done = []

        def worker():
            lim.acquire(1)
            done.append(1)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
        self.assertEqual(len(done), 6)
        self.assertEqual(len(lim.events), 6)


if __name__ == "__main__":
    unittest.main()
