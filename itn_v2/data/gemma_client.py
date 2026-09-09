"""Client tối giản cho Google AI Studio (Gemma).

Key LUÔN đọc từ biến môi trường ``GOOGLE_API_KEY`` — không bao giờ ghi vào mã
nguồn hay file trong kho. Endpoint hay trả 500 tạm thời nên bắt buộc có retry.
"""

import json
import os
import re
import random
import threading
import time
import urllib.error
import urllib.request
from collections import deque

DEFAULT_MODEL = "gemma-4-31b-it"

# Hạn mức của Gemma 4 31B trên AI Studio (đo 2026-09-08).
# TPM là trần bị đụng TRƯỚC TIÊN khi chạy nhiều luồng: mỗi phản hồi ~1,5-2K token
# nên chỉ 8-10 lượt gọi trong một phút đã chạm 16K. Endpoint trả 500 chứ không
# trả 429 khi quá tải, nên nếu không tự tiết chế thì mất khoảng nửa số lượt gọi.
LIMIT_RPM = 30
LIMIT_TPM = 16_000
LIMIT_RPD = 14_400
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
RETRY_CODES = {429, 500, 502, 503, 504}


class RateLimiter:
    """Cửa sổ trượt 60 giây cho cả số lượt gọi lẫn số token, dùng chung mọi luồng."""

    def __init__(self, rpm=LIMIT_RPM, tpm=LIMIT_TPM, rpd=LIMIT_RPD, safety=0.85):
        self.rpm = max(1, int(rpm * safety))
        self.tpm = max(1, int(tpm * safety))
        self.rpd = rpd
        self.lock = threading.Lock()
        self.events = deque()          # (thời điểm, số token)
        self.day_count = 0
        self.waited = 0.0

    def reset(self):
        """Xoá cửa sổ đang đếm. Gọi khi đổi sang khoá khác: khoá mới có hạn
        mức riêng, không việc gì phải gánh phần đã tiêu của khoá cũ."""
        with self.lock:
            self.events.clear()

    def _prune(self, now):
        while self.events and now - self.events[0][0] > 60:
            self.events.popleft()

    def acquire(self, estimated_tokens):
        """Chặn cho tới khi gửi thêm một lượt nữa vẫn nằm trong hạn mức."""
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                if self.day_count >= self.rpd:
                    raise RuntimeError("đã chạm hạn mức lượt gọi trong ngày")
                used_tokens = sum(t for _, t in self.events)
                if len(self.events) < self.rpm and used_tokens + estimated_tokens <= self.tpm:
                    self.events.append((now, estimated_tokens))
                    self.day_count += 1
                    return
                oldest = self.events[0][0] if self.events else now
                sleep_for = max(0.2, 60 - (now - oldest) + 0.1)
            self.waited += sleep_for
            time.sleep(sleep_for)

    def settle(self, estimated_tokens, actual_tokens):
        """Thay ước lượng bằng số token thật lấy từ usageMetadata."""
        if actual_tokens is None:
            return
        with self.lock:
            for i in range(len(self.events) - 1, -1, -1):
                if self.events[i][1] == estimated_tokens:
                    self.events[i] = (self.events[i][0], actual_tokens)
                    return


class AllKeysDown(RuntimeError):
    """Mọi khoá đều chết hoặc đang trong thời gian nghỉ."""


def load_keys(path=None):
    """Gom khoá theo thứ tự ưu tiên: biến môi trường trước, rồi tới file.

    Khoá KHÔNG bao giờ được ghi vào mã nguồn. File khoá nằm ngoài quyền quản
    lý của kho mã; nhớ thêm nó vào .gitignore.
    """
    keys, seen = [], set()

    def add(raw):
        for part in re.split(r"[\s,;]+", raw or ""):
            part = part.strip()
            if part and part not in seen:
                seen.add(part)
                keys.append(part)

    add(os.environ.get("GOOGLE_API_KEY", ""))
    path = path or os.environ.get("GOOGLE_API_KEY_FILE")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            add(fh.read())
    return keys


def mask(key):
    """Chỉ lộ 6 ký tự cuối — đủ để phân biệt khoá trong log, không đủ để dùng."""
    return f"…{key[-6:]}" if key and len(key) > 6 else "…"


class KeyPool:
    """Nhiều khoá API dùng luân phiên, tự loại khoá hỏng.

    Ba trạng thái của một khoá:
      sống    dùng bình thường.
      nghỉ    vừa dính 429; nghỉ tới `until` rồi quay lại vòng.
      chết    bị đình chỉ hoặc sai định dạng; không bao giờ dùng lại.
    """

    def __init__(self, keys, verbose=True):
        if not keys:
            raise RuntimeError(
                "không có khoá nào: đặt GOOGLE_API_KEY hoặc GOOGLE_API_KEY_FILE")
        self.keys = list(keys)
        self.dead = set()
        self.until = {}
        self.index = 0
        self.rotations = 0
        self.lock = threading.Lock()
        self.verbose = verbose

    def _alive(self, now):
        return [k for k in self.keys
                if k not in self.dead and self.until.get(k, 0) <= now]

    def acquire(self):
        """Khoá đang dùng. Chờ nếu mọi khoá đều đang nghỉ."""
        while True:
            with self.lock:
                now = time.monotonic()
                alive = self._alive(now)
                if alive:
                    if self.keys[self.index] in alive:
                        return self.keys[self.index]
                    self.index = self.keys.index(alive[0])
                    return alive[0]
                waiting = [t for k, t in self.until.items()
                           if k not in self.dead and t > now]
                if not waiting:
                    raise AllKeysDown(
                        f"cả {len(self.keys)} khoá đều bị đình chỉ hoặc sai")
                sleep_for = min(waiting) - now
            if self.verbose:
                print(f"  [khoá] mọi khoá đang nghỉ, chờ {sleep_for:.0f}s",
                      flush=True)
            time.sleep(min(sleep_for, 30) + 0.5)

    def penalise(self, key, permanent=False, cooldown=60.0, why=""):
        """Đánh dấu khoá hỏng rồi chuyển sang khoá kế. Trả True nếu còn khoá khác."""
        with self.lock:
            if permanent:
                self.dead.add(key)
            else:
                self.until[key] = time.monotonic() + cooldown
            now = time.monotonic()
            alive = self._alive(now)
            if alive:
                self.index = self.keys.index(alive[0])
            self.rotations += 1
            n_left = len(alive)
        if self.verbose:
            trang_thai = "chết" if permanent else f"nghỉ {cooldown:.0f}s"
            print(f"  [khoá] {mask(key)} {trang_thai} ({why}) -> còn {n_left} khoá",
                  flush=True)
        return n_left > 0


# Thông điệp lỗi của Google phân biệt "hết hạn mức" với "khoá hỏng hẳn".
PERMANENT_MARKERS = ("has been suspended", "API_KEY_INVALID",
                     "API key not valid", "API_KEY_SERVICE_BLOCKED")
DAILY_MARKERS = ("PerDay", "per day", "GenerateRequestsPerDayPerProject")


def classify_key_error(code, body):
    """(hỏng vĩnh viễn?, số giây nghỉ) cho một lỗi HTTP, hoặc None nếu không phải lỗi khoá."""
    if code == 403 and any(m in body for m in PERMANENT_MARKERS):
        return True, 0.0
    if code == 400 and any(m in body for m in PERMANENT_MARKERS):
        return True, 0.0
    if code == 429:
        return False, 3600.0 if any(m in body for m in DAILY_MARKERS) else 65.0
    return None


class GemmaClient:
    def __init__(self, model=DEFAULT_MODEL, api_key=None, max_retries=6, timeout=180,
                 limiter=None, keys=None, key_file=None):
        self.model = model
        if keys is None:
            keys = [api_key] if api_key else load_keys(key_file)
        self.pool = KeyPool(keys)
        self.api_key = self.pool.keys[0]
        self.max_retries = max_retries
        self.timeout = timeout
        self.limiter = limiter if limiter is not None else RateLimiter()
        self.calls = 0
        self.retries = 0
        self.tokens = 0

    def generate(self, prompt, temperature=1.0, max_tokens=2048, response_schema=None):
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if response_schema is not None:
            # Gemini hỗ trợ JSON có lược đồ -> không phải thu hoạch phòng thủ
            # như với Gemma nữa.
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = response_schema
        data = json.dumps(body).encode()
        # Ước lượng trước khi gửi: prompt chia 4 ký tự một token, cộng trần đầu ra.
        estimate = len(prompt) // 4 + max_tokens
        attempt = 0
        # Trần số lần đổi khoá cho MỘT lượt gọi: hai vòng quanh cả kho khoá.
        # Không có trần này thì hai khoá cùng dính 429 sẽ đá qua đá lại mãi.
        rotations_left = 2 * len(self.pool.keys) + 2
        while attempt < self.max_retries:
            key = self.pool.acquire()
            url = ENDPOINT.format(model=self.model, key=key)
            if self.limiter:
                self.limiter.acquire(estimate)
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.load(resp)
                self.calls += 1
                usage = payload.get("usageMetadata") or {}
                actual = usage.get("totalTokenCount")
                if actual:
                    self.tokens += actual
                if self.limiter:
                    self.limiter.settle(estimate, actual)
                candidates = payload.get("candidates") or []
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts") or []
                return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:600]
                except Exception:
                    detail = ""
                verdict = classify_key_error(exc.code, detail)
                if verdict is not None:
                    # Hỏng ở KHOÁ chứ không phải ở yêu cầu: đổi khoá rồi thử
                    # lại ngay, và KHÔNG tính vào số lần thử lại — nếu tính,
                    # một khoá chết sẽ ăn hết ngân sách retry của cả lượt gọi.
                    permanent, cooldown = verdict
                    rotations_left -= 1
                    if rotations_left <= 0:
                        raise AllKeysDown(
                            f"đổi khoá quá nhiều lần trong một lượt gọi: {detail[:200]}") from exc
                    if self.pool.penalise(key, permanent, cooldown,
                                          why=f"HTTP {exc.code}"):
                        if self.limiter:
                            self.limiter.reset()
                        continue
                    raise AllKeysDown(f"hết khoá dùng được: {detail[:200]}") from exc
                attempt += 1
                if exc.code in RETRY_CODES and attempt < self.max_retries:
                    self.retries += 1
                    time.sleep(min(2 ** attempt, 20) + random.random())
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                attempt += 1
                if attempt < self.max_retries:
                    self.retries += 1
                    time.sleep(min(2 ** attempt, 20) + random.random())
                    continue
                raise
        return ""
