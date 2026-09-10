"""Nạp và tra danh mục ngoài (spec §14).

Luồng: chuẩn hoá cách đọc -> tra khớp chính xác -> nếu trượt thì tra gần đúng
-> ngưỡng tin cậy -> trả canonical hoặc None để pipeline giữ dạng nói.
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

CATALOG_DIR = Path(__file__).resolve().parent / "catalogs"
DEFAULT_MATCH_THRESHOLD = 0.86

# Từ chỉ số trong dạng nói. Khớp gần đúng được phép sai ở phần CHỮ (cách đánh
# vần tiền tố: "míc"/"mích", "su"/"xu") nhưng TUYỆT ĐỐI không được sai ở phần
# SỐ. Nếu không, "su bảy mươi hai" (Su-72, chưa có trong danh mục) khớp với
# "su ba mươi hai" ở mức 0,8696 và trả về Su-32 — bịa ra một số hiệu khí tài
# khác có thật. Spec cấm đúng chuyện này: giá trị bịa nguy hiểm hơn nhiều so
# với việc giữ nguyên dạng nói.
NUMBER_SPOKEN = ("không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm",
                 "nhăm", "sáu", "bảy", "bẩy", "tám", "chín", "mười", "mươi",
                 "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "lẻ", "linh")


def deaccent(text: str) -> str:
    """Bỏ dấu thanh và dấu phụ, đ -> d.

    Danh mục tên riêng và thuật ngữ tiếng Anh được phiên âm bằng âm tiết Việt,
    mà dấu thanh trên một âm tiết phiên âm là TUỲ TIỆN: cùng một từ HIMARS,
    ASR lúc ra "hi mác" lúc ra "hi mắc". Bỏ dấu rồi mới so thì hai cái là một.
    Đo trên bốn danh mục: 0 va chạm ở foreign_names và equipment, 3 va chạm ở
    maritime_terms đều là hai bản viết hoa/thường của cùng một mục.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")


# Nhóm phụ âm mà người Việt dùng lẫn nhau khi phiên âm tiếng Anh. Đây là cách
# ASR viết CÙNG MỘT ÂM ra nhiều chữ khác nhau, không phải khác biệt nghĩa.
_ONSET_GROUPS = (("ngh", "N"), ("ng", "N"), ("nh", "N"), ("gh", "G"),
                 ("kh", "K"), ("ph", "F"), ("th", "T"), ("tr", "C"),
                 ("ch", "C"), ("gi", "J"), ("qu", "K"))
_LETTER_GROUPS = {"c": "K", "k": "K", "q": "K", "g": "G", "n": "N", "m": "M",
                  "p": "P", "b": "P", "t": "T", "d": "J", "j": "J", "r": "J",
                  "x": "S", "s": "S", "f": "F", "v": "V", "w": "V", "h": "H",
                  "l": "L", "y": "I", "i": "I", "e": "E", "a": "A", "o": "O",
                  "u": "U"}
_REPEATED = re.compile(r"(.)\1+")


def phonetic_key(text: str) -> str:
    """Khoá ngữ âm cho tên riêng phiên âm: "hy mác"/"hi mắc" -> HIMAK.

    Chỉ dùng cho mục KHÔNG có chữ số trong dạng chuẩn. Khoá này bỏ hết chữ số
    nên nếu áp cho số hiệu thì AK-47, AK-101 và AK-630 gộp chung làm một —
    đo được: 35 va chạm ở equipment, 7 ở acronyms, 0 ở foreign_names.
    """
    text = _REPEATED.sub(r"\1", deaccent(text).lower())
    text = re.sub(r"[^a-z]", "", text)
    out, i, n = [], 0, len(text)
    while i < n:
        for pattern, symbol in _ONSET_GROUPS:
            if text.startswith(pattern, i):
                out.append(symbol)
                i += len(pattern)
                break
        else:
            out.append(_LETTER_GROUPS.get(text[i], text[i].upper()))
            i += 1
    return _REPEATED.sub(r"\1", "".join(out))


def normalize_spoken(text: str) -> str:
    """Dạng chuẩn để so khớp: NFC, chữ thường, bỏ mọi khoảng trắng và gạch dưới.

    Bỏ khoảng trắng để chịu được việc ASR cắt cụm thành số lượng từ khác nhau
    ("pa tri ốt" / "pat ri ốt" / "patriốt" đều về một chuỗi).
    """
    text = unicodedata.normalize("NFC", str(text)).lower()
    return "".join(ch for ch in text if not ch.isspace() and ch != "_")


@lru_cache(maxsize=None)
def load_raw(name: str) -> dict:
    path = CATALOG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"không tìm thấy danh mục {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


class Catalog:
    """Danh mục thực thể: nhiều cách đọc -> một dạng chuẩn."""

    def __init__(self, name, threshold=DEFAULT_MATCH_THRESHOLD):
        self.name = name
        self.threshold = threshold
        raw = load_raw(name)
        self.index = {}          # cách đọc đã chuẩn hoá -> canonical
        self.index_flat = {}     # cách đọc đã BỎ DẤU -> canonical
        self.index_phon = {}     # khoá NGỮ ÂM -> canonical (chỉ mục không có số)
        self.canonicals = []
        # equipment.json để danh sách model dưới khoá "models" chứ không phải
        # "entries", nên trước đây EquipmentNameResolver không thấy mục nào.
        for entry in list(raw.get("entries", [])) + list(raw.get("models", [])):
            canonical = entry["canonical"]
            self.canonicals.append(canonical)
            for spoken in entry.get("spoken", []):
                key = normalize_spoken(spoken)
                self.index[key] = canonical
                self.index_flat.setdefault(deaccent(key), canonical)
                if not self._has_digits(canonical):
                    self.index_phon.setdefault(phonetic_key(spoken), canonical)
            # bản thân dạng chuẩn cũng là một cách khớp hợp lệ
            key = normalize_spoken(canonical)
            self.index.setdefault(key, canonical)
            self.index_flat.setdefault(deaccent(key), canonical)
            if not self._has_digits(canonical):
                self.index_phon.setdefault(phonetic_key(canonical), canonical)
        self.raw = raw

    @staticmethod
    def _number_signature(text):
        """Dãy từ chỉ số trong một cách đọc, theo đúng thứ tự."""
        i, out, n = 0, [], len(text)
        while i < n:
            for word in NUMBER_SPOKEN:
                if text.startswith(word, i):
                    out.append(word)
                    i += len(word)
                    break
            else:
                i += 1
        return tuple(out)

    @staticmethod
    def _has_digits(canonical):
        return any(ch.isdigit() for ch in canonical)

    def lookup(self, spoken):
        """Trả (canonical, score). Ba tầng, chặt dần:

          1. khớp chính xác                        -> 1.00
          2. khớp sau khi bỏ dấu                   -> 0.97
          3. khớp theo KHOÁ NGỮ ÂM (chỉ tên riêng) -> 0.94
          4. khớp gần đúng trên dạng bỏ dấu, ngưỡng self.threshold

        Tầng 3 bị chặn nếu phần SỐ khác nhau VÀ dạng chuẩn có chữ số — nếu
        không, "su bảy mươi hai" khớp 0,8696 với "su ba mươi hai" rồi trả về
        Su-32, bịa ra một số hiệu khí tài khác có thật. Với dạng chuẩn KHÔNG có
        chữ số (Himars, Patriot) thì từ chỉ số chỉ là phiên âm, không mang giá
        trị, nên không chặn.
        """
        key = normalize_spoken(spoken)
        if not key:
            return None, 0.0
        if key in self.index:
            return self.index[key], 1.0
        flat = deaccent(key)
        if flat in self.index_flat:
            return self.index_flat[flat], 0.97
        phon = phonetic_key(key)
        if phon and phon in self.index_phon:
            return self.index_phon[phon], 0.94

        key_numbers = self._number_signature(key)
        best, best_score = None, 0.0
        blocked_score = 0.0
        for cand_key, canonical in self.index.items():
            # cắt sớm theo chênh lệch độ dài để khỏi quét toàn bộ vô ích
            if abs(len(cand_key) - len(key)) > max(3, len(key) // 3):
                continue
            score = SequenceMatcher(None, flat, deaccent(cand_key)).ratio()
            if not self._has_digits(canonical):
                # Với TÊN RIÊNG, so thêm trên khoá ngữ âm và lấy điểm cao hơn:
                # "ja vơ lin" vs "ja vê lin" chỉ khác nguyên âm, ASR lẫn thường
                # xuyên. Không áp cho mục có chữ số vì khoá ngữ âm bỏ chữ số.
                score = max(score, SequenceMatcher(
                    None, phon, phonetic_key(cand_key)).ratio())
            if score <= best_score:
                continue
            if (self._has_digits(canonical)
                    and self._number_signature(cand_key) != key_numbers):
                blocked_score = max(blocked_score, score)
                continue
            best, best_score = canonical, score
        if best_score >= self.threshold:
            return best, best_score
        return None, max(best_score, blocked_score)


@lru_cache(maxsize=None)
def get_catalog(name, threshold=DEFAULT_MATCH_THRESHOLD) -> Catalog:
    return Catalog(name, threshold)


@lru_cache(maxsize=None)
def get_units():
    raw = load_raw("units")
    simple = raw["simple"]
    # sắp xếp theo số từ giảm dần: "độ xê" phải thắng "độ"
    ordered = sorted(simple.items(), key=lambda kv: -len(kv[0].split()))
    # per_unit_suffixes ghi rõ MẪU SỐ cho từng cách đọc ("trên giây" -> s).
    # per_hour_suffixes là dạng cũ, chỉ có tử số; giữ để không phá code cũ.
    per_unit = raw.get("per_unit_suffixes")
    if not per_unit:
        per_unit = {k: "h" for k in raw["per_hour_suffixes"]}
    return [(k.split(), v) for k, v in ordered], per_unit, raw["glue_prefixes"]


@lru_cache(maxsize=None)
def get_equipment_prefixes():
    raw = load_raw("equipment")
    return sorted(raw["prefixes"].items(), key=lambda kv: -len(kv[0].split()))


@lru_cache(maxsize=None)
def get_letters():
    """Bảng đọc chữ cái, dùng cho mã cảng, callsign, địa chỉ điện tử."""
    raw = load_raw("port_codes")
    return sorted(raw["letters"].items(), key=lambda kv: -len(kv[0].split()))
