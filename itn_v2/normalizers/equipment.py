"""Số hiệu khí tài (spec §13).

Hai tầng: tra danh mục model đã biết trước (giữ đúng dạng chuẩn kể cả khi
không theo khuôn, như M16 không gạch hay MK 46 có dấu cách), sau đó mới tới
văn phạm tổng quát cho số hiệu chưa gặp.

Bắt buộc: GIỮ NGUYÊN hoa-thường của tiền tố. V1 viết hoa toàn bộ nên Su-30
thành SU-30 và MiG-21 thành MIG-21.
"""

from ..catalog import get_catalog, get_equipment_prefixes, normalize_spoken
from .base import Normalizer, ParseError
from .identifier import DIGIT_LIKE, read_alphanumeric
from .number import read_number_auto

MODEL_MATCH_THRESHOLD = 0.90

# DIGIT_LIKE chỉ có chữ số rời. Thiếu "trăm"/"nghìn" nên "ét hai trăm" (S-200)
# và "ca ba trăm pê" (K-300P) đứt ngay tại từ "trăm" rồi ném ParseError.
NUMBER_WORDS = DIGIT_LIKE | {"trăm", "nghìn", "ngàn"}


class EquipmentIDParser(Normalizer):
    name = "EquipmentIDParser"

    def __init__(self):
        raw = get_catalog("equipment").raw
        self.models = {}
        for entry in raw.get("models", []):
            for spoken in entry.get("spoken", []):
                self.models[normalize_spoken(spoken)] = entry["canonical"]
            self.models.setdefault(normalize_spoken(entry["canonical"]), entry["canonical"])

    def parse(self, raw_text, context=None):
        key = normalize_spoken(raw_text)
        if key in self.models:
            return self.models[key]

        words = raw_text.split()
        for spoken, symbol in get_equipment_prefixes():
            sw = spoken.split()
            if words[:len(sw)] != sw:
                continue
            rest = words[len(sw):]
            if not rest:
                continue
            j = 0
            while j < len(rest) and rest[j] in NUMBER_WORDS:
                j += 1
            if j == 0:
                continue
            number = read_number_auto(rest[:j])
            out = f"{symbol}-{number}"
            if rest[j:]:
                out += read_alphanumeric(rest[j:])
            return out
        raise ParseError("không khớp danh mục model lẫn văn phạm tiền tố + số")
