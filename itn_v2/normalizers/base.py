"""Giao diện chung cho mọi bộ chuẩn hoá theo kiểu (spec §10, §15).

Mọi normalizer trả về cùng một cấu trúc, kể cả khi thất bại. Không normalizer
nào được phép ném exception ra ngoài: thất bại là một kết quả hợp lệ và
pipeline sẽ giữ nguyên dạng nói (spec §16).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class NormResult:
    raw_text: str
    type: str
    normalized: Optional[str]
    valid: bool
    reason: str = ""
    normalizer: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


class ParseError(ValueError):
    """Cụm không khớp văn phạm của kiểu đang xét."""


class Normalizer:
    """Lớp cơ sở. Lớp con cài ``parse`` và ném ``ParseError`` khi không khớp."""

    name = "Normalizer"

    def parse(self, raw_text: str, context: Optional[dict] = None) -> str:
        raise NotImplementedError

    def __call__(self, raw_text: str, type_name: str = "", context: Optional[dict] = None) -> NormResult:
        text = " ".join(str(raw_text).split())
        if not text:
            return NormResult(raw_text, type_name or self.name, None, False,
                              "cụm rỗng", self.name)
        try:
            normalized = self.parse(text, context)
        except ParseError as exc:
            return NormResult(text, type_name or self.name, None, False,
                              f"parser: {exc}", self.name)
        except Exception as exc:  # lỗi ngoài dự kiến vẫn phải fallback an toàn
            return NormResult(text, type_name or self.name, None, False,
                              f"parser lỗi bất thường: {type(exc).__name__}: {exc}", self.name)
        if normalized is None or normalized == "":
            return NormResult(text, type_name or self.name, None, False,
                              "parser trả về rỗng", self.name)
        return NormResult(text, type_name or self.name, normalized, True, "", self.name)
