"""Toạ độ, hướng đi, phương vị (spec §4.3, §15.1, §15.2)."""

from .base import Normalizer, ParseError
from .number import read_number_auto

DIRECTIONS = {"bắc": "N", "nam": "S", "đông": "E", "tây": "W",
              "n": "N", "s": "S", "e": "E", "w": "W"}


class CoordinateParser(Normalizer):
    """"mười độ hai mươi lăm phút bắc" -> 10°25'N"""

    name = "CoordinateParser"

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        if not words:
            raise ParseError("cụm rỗng")
        direction = None
        if words[-1] in DIRECTIONS:
            direction = DIRECTIONS[words[-1]]
            words = words[:-1]
        if "độ" not in words:
            raise ParseError("không có từ 'độ'")
        i = words.index("độ")
        degree_words, rest = words[:i], words[i + 1:]
        if not degree_words:
            raise ParseError("thiếu phần độ")
        degree = read_number_auto(degree_words)

        minute = second = None
        if "phút" in rest:
            k = rest.index("phút")
            minute = read_number_auto(rest[:k])
            rest = rest[k + 1:]
        if "giây" in rest:
            k = rest.index("giây")
            second = read_number_auto(rest[:k])
            rest = rest[k + 1:]

        out = f"{degree}°"
        if minute is not None:
            out += f"{minute:02d}'"
        if second is not None:
            out += f'{second:02d}"'
        if direction:
            out += direction
        return out


class _Bearing(Normalizer):
    """Hướng đi / phương vị: luôn ba chữ số, giữ số 0 đầu (spec §15.1)."""

    strip_words = ("hướng", "phương", "vị", "phương_vị", "độ")

    def parse(self, raw_text, context=None):
        words = [w for w in raw_text.split() if w not in self.strip_words]
        if not words:
            raise ParseError("thiếu phần số")
        value = read_number_auto(words)
        if not 0 <= value <= 359:
            raise ParseError(f"giá trị {value} ngoài khoảng 0-359")
        return f"{value:03d}°"


class HeadingParser(_Bearing):
    name = "HeadingParser"


class BearingParser(_Bearing):
    name = "BearingParser"
