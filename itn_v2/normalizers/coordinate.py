"""Toạ độ, hướng đi, phương vị (spec §4.3, §15.1, §15.2)."""

from .base import Normalizer, ParseError
from .number import read_number_auto

DIRECTIONS = {"bắc": "N", "nam": "S", "đông": "E", "tây": "W",
              "n": "N", "s": "S", "e": "E", "w": "W"}
# "vĩ bắc" / "kinh đông": từ chỉ trục đứng ngay trước từ chỉ hướng.
AXIS_WORDS = {"vĩ", "kinh", "vĩ_độ", "kinh_độ"}

# Ký hiệu phút/giây là DẤU NGUYÊN U+2032/U+2033, không phải nháy ASCII. Spec
# mục 288 viết 10°25'N bằng nháy thường, nhưng 18/18 toạ độ trong bản gốc thật
# đều dùng ′ ″ — và dữ liệu tổng hợp mới cũng vậy. Bản gốc thắng: đây là thứ ta
# bị chấm điểm.
PRIME, DOUBLE_PRIME = "′", "″"
# Vĩ độ hai chữ số, kinh độ ba: 08°45′20″S nhưng 107°08′15″E.
DEGREE_WIDTH = {"N": 2, "S": 2, "E": 3, "W": 3}


def _split_coordinates(words):
    """Cắt cụm thành từng toạ độ một, mỗi cái kết thúc ở một từ chỉ hướng.

    Bản trước chỉ đọc MỘT toạ độ: nó lấy `words[-1]` làm hướng rồi tìm "độ" đầu
    tiên, nên với cụm "…giây bắc …độ… giây đông" nó ghép phần số của vĩ độ với
    chữ E của kinh độ và vứt hẳn kinh độ đi. Sai câm — vẫn trả ra chuỗi hợp lệ.
    """
    segments, current = [], []
    for word in words:
        if word in DIRECTIONS and current:
            segments.append((current, DIRECTIONS[word]))
            current = []
        elif word in AXIS_WORDS:
            continue
        else:
            current.append(word)
    if current:
        segments.append((current, None))
    return segments


# Số phần thập phân của phút trong khuôn độ-phút-thập-phân. Bản gốc thật dùng
# đúng ba chữ số: 10°25.500′N.
DECIMAL_MINUTE_DIGITS = 3


def _split_decimal_minute(minute, second):
    """Tách "hai mươi lăm nghìn năm trăm phút" (=25500) thành 25 phút .500.

    Đây là khuôn ĐỘ - PHÚT THẬP PHÂN (DDM) mà hải đồ điện tử dùng cho waypoint:
    10°25.500′N. Người đọc đọc liền cả cụm chữ số nên ra một số lớn hơn 60, và
    bản trước để nguyên thành 10°25500′N rồi bị validator bác — mất 7 toạ độ.

    Chỉ chạy khi phút >= 60, tức là giá trị đã chắc chắn không hợp lệ ở khuôn
    độ-phút-giây, nên nhánh này không cướp mất trường hợp nào đang chạy đúng.
    """
    if second is not None:
        # DDM không có phần giây; có cả hai nghĩa là cụm đã đọc sai chỗ khác.
        raise ParseError(f"phút {minute} vượt 59 mà cụm vẫn có phần giây")
    digits = str(minute)
    if len(digits) <= DECIMAL_MINUTE_DIGITS:
        raise ParseError(f"phút {minute} ngoài khoảng 0-59")
    whole, fraction = digits[:-DECIMAL_MINUTE_DIGITS], digits[-DECIMAL_MINUTE_DIGITS:]
    value = int(whole)
    if not 0 <= value < 60:
        raise ParseError(f"phần phút {value} ngoài khoảng 0-59")
    return value, fraction


def _format_one(words, direction):
    if "độ" not in words:
        raise ParseError("không có từ 'độ'")
    i = words.index("độ")
    degree_words, rest = words[:i], words[i + 1:]
    if not degree_words:
        raise ParseError("thiếu phần độ")
    degree = read_number_auto(degree_words)

    minute = second = None
    minute_fraction = None
    if "phút" in rest:
        k = rest.index("phút")
        head = rest[:k]
        if "phẩy" in head:
            # "bốn mươi hai PHẨY chín phút" = 42,9 phút. Khuôn độ-phút-thập-phân
            # đọc theo lối này thay vì đọc liền cả cụm chữ số.
            j = head.index("phẩy")
            minute = read_number_auto(head[:j])
            minute_fraction = str(read_number_auto(head[j + 1:]))
        else:
            minute = read_number_auto(head)
        rest = rest[k + 1:]
    if "giây" in rest:
        k = rest.index("giây")
        second = read_number_auto(rest[:k])
        rest = rest[k + 1:]

    if minute is not None and minute_fraction is None and minute >= 60:
        minute, minute_fraction = _split_decimal_minute(minute, second)

    width = DEGREE_WIDTH.get(direction, 0)
    out = f"{degree:0{width}d}°" if width else f"{degree}°"
    if minute is not None:
        out += f"{minute:02d}"
        if minute_fraction is not None:
            out += f".{minute_fraction}"
        out += PRIME
    if second is not None:
        out += f"{second:02d}{DOUBLE_PRIME}"
    if direction:
        out += direction
    return out


class CoordinateParser(Normalizer):
    """"mười độ hai mươi lăm phút bắc" -> 10°25′N

    Cụm thật thường là cả CẶP vĩ độ + kinh độ trong một span:
    "mười độ … giây bắc một trăm lẻ bảy độ … giây đông" -> 10°25′30″N, 107°08′15″E
    """

    name = "CoordinateParser"

    def parse(self, raw_text, context=None):
        words = raw_text.replace(",", " ").split()
        if not words:
            raise ParseError("cụm rỗng")
        segments = _split_coordinates(words)
        if not segments:
            raise ParseError("không tách được toạ độ nào")
        return ", ".join(_format_one(w, d) for w, d in segments)


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
