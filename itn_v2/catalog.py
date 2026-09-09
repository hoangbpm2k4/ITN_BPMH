"""Nạp và tra danh mục ngoài (spec §14).

Luồng: chuẩn hoá cách đọc -> tra khớp chính xác -> nếu trượt thì tra gần đúng
-> ngưỡng tin cậy -> trả canonical hoặc None để pipeline giữ dạng nói.
"""

import json
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

CATALOG_DIR = Path(__file__).resolve().parent / "catalogs"
DEFAULT_MATCH_THRESHOLD = 0.86


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
        self.canonicals = []
        # equipment.json để danh sách model dưới khoá "models" chứ không phải
        # "entries", nên trước đây EquipmentNameResolver không thấy mục nào.
        for entry in list(raw.get("entries", [])) + list(raw.get("models", [])):
            canonical = entry["canonical"]
            self.canonicals.append(canonical)
            for spoken in entry.get("spoken", []):
                self.index[normalize_spoken(spoken)] = canonical
            # bản thân dạng chuẩn cũng là một cách khớp hợp lệ
            self.index.setdefault(normalize_spoken(canonical), canonical)
        self.raw = raw

    def lookup(self, spoken):
        """Trả (canonical, score). score = 1.0 khi khớp chính xác."""
        key = normalize_spoken(spoken)
        if not key:
            return None, 0.0
        if key in self.index:
            return self.index[key], 1.0
        best, best_score = None, 0.0
        for cand_key, canonical in self.index.items():
            # cắt sớm theo chênh lệch độ dài để khỏi quét toàn bộ vô ích
            if abs(len(cand_key) - len(key)) > max(3, len(key) // 3):
                continue
            score = SequenceMatcher(None, key, cand_key).ratio()
            if score > best_score:
                best, best_score = canonical, score
        if best_score >= self.threshold:
            return best, best_score
        return None, best_score


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
