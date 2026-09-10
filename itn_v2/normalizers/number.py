"""Lõi đọc số tiếng Việt + các bộ chuẩn hoá số cơ bản.

Bài học từ V1: bộ số cũ **tự đoán** đang đọc dãy chữ số hay đọc số đếm bằng
heuristic "cụm có chứa từ mười/mươi/trăm không". Heuristic đó phá huỷ số điện
thoại đọc theo nhóm — "không tám tám hai mươi mốt ba bốn năm sáu" trả về "6".

V2 không đoán nữa: **kiểu ngữ nghĩa do mô hình dự đoán quyết định chế độ đọc**.
  - đọc dãy chữ số : DIGIT_SEQ, TELEPHONE, MMSI_ID, IMO_ID, HEADING, ...
  - đọc số đếm      : CARDINAL, MONEY, MEASURE, ...
"""

import re
from .base import Normalizer, ParseError

# --- từ vựng -------------------------------------------------------------
UNIT_DIGITS = {
    "không": 0, "linh": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3,
    "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "nhăm": 5,
    "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9,
}
# Ba từ chỉ xuất hiện ở hàng đơn vị sau hàng chục -> tín hiệu đọc tắt ("ba lăm" = 35).
COLLOQUIAL_UNITS = {"mốt", "lăm", "nhăm", "tư"}
SCALES = {"trăm": 100, "nghìn": 1000, "ngàn": 1000,
          "triệu": 10 ** 6, "tỷ": 10 ** 9, "tỉ": 10 ** 9}
BIG_SCALES = {"nghìn", "ngàn", "triệu", "tỷ", "tỉ"}
FILLERS = {"lẻ", "linh"}
NEGATIVE_WORDS = {"âm", "trừ"}
STRUCTURE_WORDS = {"mười", "mươi"} | set(SCALES)

DECIMAL_SEP_WORDS = {"phẩy", "phảy"}
DOT_SEP_WORDS = {"chấm"}


def strip_negative(words):
    """Tách tiền tố phủ định. Số âm là *modifier*, không phải kiểu riêng (spec §4.1)."""
    if words and words[0] in NEGATIVE_WORDS:
        return True, words[1:]
    return False, words


def has_structure(words):
    return any(w in STRUCTURE_WORDS for w in words)


def format_thousands(value) -> str:
    """Phân nhóm hàng nghìn bằng dấu chấm (chuẩn hiển thị tiếng Việt)."""
    neg = "-" if int(value) < 0 else ""
    return neg + "{:,}".format(abs(int(value))).replace(",", ".")


# --- chế độ 1: đọc dãy chữ số -------------------------------------------
def read_digit_string(words) -> str:
    """Đọc thành chuỗi chữ số, giữ nguyên số 0 đầu và độ dài.

    Chấp nhận cả đọc rời từng chữ và đọc theo nhóm hai chữ số:
        không tám tám hai một ba bốn năm sáu       -> 088213456
        không tám tám hai mươi mốt ba bốn năm sáu  -> 088213456
    """
    if isinstance(words, str):
        words = words.split()
    out, i, n = [], 0, len(words)
    while i < n:
        w = words[i]
        if w.isdigit():
            out.append(w)
            i += 1
            continue
        if w == "mười":
            if i + 1 < n and words[i + 1] in UNIT_DIGITS and words[i + 1] != "không":
                out.append(str(10 + UNIT_DIGITS[words[i + 1]]))
                i += 2
            else:
                out.append("10")
                i += 1
            continue
        if w in UNIT_DIGITS:
            d = UNIT_DIGITS[w]
            # <d> mươi [đơn vị]
            if i + 1 < n and words[i + 1] == "mươi":
                val = d * 10
                j = i + 2
                if j < n and words[j] in UNIT_DIGITS:
                    val += UNIT_DIGITS[words[j]]
                    j += 1
                out.append(f"{val:02d}")
                i = j
                continue
            # đọc tắt "<d> mốt|lăm|nhăm|tư"
            if d >= 1 and i + 1 < n and words[i + 1] in COLLOQUIAL_UNITS:
                out.append(f"{d * 10 + UNIT_DIGITS[words[i + 1]]:02d}")
                i += 2
                continue
            out.append(str(d))
            i += 1
            continue
        raise ParseError(f"từ {w!r} không thuộc văn phạm dãy chữ số")
    if not out:
        raise ParseError("không đọc được chữ số nào")
    return "".join(out)


# --- chế độ 2: đọc số đếm ------------------------------------------------
def read_cardinal(words) -> int:
    """Đọc số đếm theo văn phạm đầy đủ: trăm / nghìn / triệu / tỷ, lẻ, linh."""
    if isinstance(words, str):
        words = words.split()
    total = section = cur = 0
    pending_unit = False   # vừa gặp mười/mươi/lẻ -> chữ số kế tiếp là hàng đơn vị
    seen = False
    # "trăm"/"nghìn"/"mươi" đứng một mình KHÔNG phải một con số — đó là mảnh vỡ
    # của cụm dài hơn bị cắt sai biên. `seen` cũ bật lên cả với chúng, nên
    # "trăm" đọc ra 0 và "mươi" đọc ra 10: một giá trị sai dựng lặng lẽ từ một
    # biên sai, không cổng nào bắt được.
    saw_digit = False
    for w in words:
        if w.isdigit():
            v = int(w)
            cur = cur + v if pending_unit else v
            pending_unit = False
            seen = saw_digit = True
        elif w in FILLERS:
            pending_unit = True
        elif w == "mười":
            cur = 10
            pending_unit = True
            seen = saw_digit = True
        elif w == "mươi":
            cur = cur * 10 if cur else 10
            pending_unit = True
            seen = True
        elif w in UNIT_DIGITS:
            d = UNIT_DIGITS[w]
            cur = cur + d if pending_unit else d
            pending_unit = False
            seen = saw_digit = True
        elif w == "trăm":
            section += cur * 100        # "không trăm" -> 0, không phải 100
            cur = 0
            pending_unit = False
            seen = True
        elif w in BIG_SCALES:
            total += (section + cur) * SCALES[w]
            section = cur = 0
            pending_unit = False
            seen = True
        else:
            raise ParseError(f"từ {w!r} không thuộc văn phạm số đếm")
    if not seen:
        raise ParseError("không đọc được số đếm")
    if not saw_digit:
        raise ParseError(
            f"{' '.join(words)!r} chỉ có từ chỉ hàng, không có chữ số nào")
    return total + section + cur


def read_number_auto(words) -> int:
    """Dùng cho phần *bên trong* một parser đã biết kiểu, khi cụm chắc chắn là
    một giá trị đếm (ví dụ phần ngày, phần tháng). Không dùng ở tầng dispatch.

    Việc chặn cụm chỉ có từ chỉ hàng nằm trong `read_cardinal` — ở đó mới phân
    biệt được "mười" (là số 10) với "mươi"/"trăm" (chỉ là từ chỉ hàng)."""
    if isinstance(words, str):
        words = words.split()
    if not words:
        raise ParseError("cụm số rỗng")
    if has_structure(words):
        return read_cardinal(words)
    return int(read_digit_string(words))


def read_year(words) -> int:
    """Đọc năm: 'hai không hai sáu'->2026, 'một chín tám mươi'->1980,
    'hai nghìn không trăm hai mươi tư'->2024."""
    if isinstance(words, str):
        words = words.split()
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])

    def tail_value(tail):
        if not tail:
            return None
        if has_structure(tail):
            return read_cardinal(tail)
        return int(read_digit_string(tail))

    if len(words) >= 3 and words[0] == "hai" and words[1] in {"nghìn", "ngàn"}:
        tail = words[2:]
        if tail[:2] == ["không", "trăm"]:
            tail = tail[2:]
        elif tail[:1] == ["không"]:
            tail = tail[1:]
        v = tail_value(tail)
        if v is not None and 0 <= v <= 99:
            return 2000 + v
        if v is None:
            return 2000
    if len(words) >= 3 and words[0] == "hai" and words[1] == "không":
        v = tail_value(words[2:])
        if v is not None and 0 <= v <= 99:
            return 2000 + v
    if len(words) >= 3 and words[0] == "một" and words[1] == "chín":
        v = tail_value(words[2:])
        if v is not None and 0 <= v <= 99:
            return 1900 + v
    if len(words) == 4 and all(w in UNIT_DIGITS for w in words):
        return int(read_digit_string(words))
    return read_cardinal(words)


# --- các bộ chuẩn hoá ----------------------------------------------------
class CardinalParser(Normalizer):
    name = "CardinalParser"

    def parse(self, raw_text, context=None):
        neg, words = strip_negative(raw_text.split())
        value = read_cardinal(words)
        return ("-" if neg else "") + format_thousands(value)


class DigitSequenceParser(Normalizer):
    name = "DigitSequenceParser"

    def parse(self, raw_text, context=None):
        return read_digit_string(raw_text.split())


class OrdinalParser(Normalizer):
    name = "OrdinalParser"
    PREFIXES = ("thứ", "hạng", "số")

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        prefix = ""
        if words and words[0] in self.PREFIXES:
            prefix, words = words[0], words[1:]
        if not words:
            raise ParseError("thiếu phần số sau tiền tố thứ tự")
        if words == ["nhất"]:
            value = 1
        elif words == ["nhì"]:
            value = 2
        else:
            value = read_number_auto(words)
        return f"{prefix} {value}".strip()


class DecimalParser(Normalizer):
    name = "DecimalParser"

    def parse(self, raw_text, context=None):
        neg, words = strip_negative(raw_text.split())
        idx = next((i for i, w in enumerate(words) if w in DECIMAL_SEP_WORDS), None)
        if idx is None:
            raise ParseError("không có từ 'phẩy'")
        left, right = words[:idx], words[idx + 1:]
        if not left or not right:
            raise ParseError("thiếu phần nguyên hoặc phần thập phân")
        # Phần nguyên đọc rời (tần số "một hai một phẩy năm") phải giữ nguyên
        # chuỗi chữ số, không chèn dấu phân nhóm nghìn.
        if has_structure(left):
            int_part = format_thousands(read_cardinal(left))
        else:
            int_part = read_digit_string(left)
        frac_part = read_digit_string(right)
        return f"{'-' if neg else ''}{int_part},{frac_part}"


class FractionParser(Normalizer):
    name = "FractionParser"

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        try:
            idx = words.index("phần")
        except ValueError:
            raise ParseError("không có từ 'phần'")
        num, den = words[:idx], words[idx + 1:]
        if not num or not den:
            raise ParseError("thiếu tử số hoặc mẫu số")
        return f"{read_number_auto(num)}/{read_number_auto(den)}"


class RatioParser(Normalizer):
    name = "RatioParser"
    SEPARATORS = {"trên", "chọi", "so"}

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        idx = next((i for i, w in enumerate(words) if w in self.SEPARATORS), None)
        if idx is None:
            raise ParseError("không có từ nối tỷ lệ")
        left, right = words[:idx], words[idx + 1:]
        if not left or not right:
            raise ParseError("thiếu một vế của tỷ lệ")
        return f"{read_number_auto(left)}:{read_number_auto(right)}"


class PercentParser(Normalizer):
    name = "PercentParser"

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        for tail in (["phần", "trăm"], ["phần_trăm"]):
            if len(words) > len(tail) and words[-len(tail):] == tail:
                words = words[:-len(tail)]
                break
        else:
            raise ParseError("không kết thúc bằng 'phần trăm'")
        neg, words = strip_negative(words)
        if any(w in DECIMAL_SEP_WORDS for w in words):
            body = DecimalParser().parse(" ".join(words))
        else:
            body = format_thousands(read_cardinal(words))
        return f"{'-' if neg else ''}{body}%"


class RangeParser(Normalizer):
    name = "RangeParser"
    STARTERS = {"từ", "khoảng"}
    SEPARATORS = {"đến", "tới", "-"}

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        if words and words[0] in self.STARTERS:
            words = words[1:]
        idx = next((i for i, w in enumerate(words) if w in self.SEPARATORS), None)
        if idx is None:
            raise ParseError("không có từ nối khoảng")
        left, right = words[:idx], words[idx + 1:]
        if not left or not right:
            raise ParseError("thiếu một đầu của khoảng")

        def one(part):
            if any(w in DECIMAL_SEP_WORDS for w in part):
                return DecimalParser().parse(" ".join(part))
            return format_thousands(read_cardinal(part))

        return f"{one(left)}-{one(right)}"


class VersionParser(Normalizer):
    name = "VersionParser"
    PREFIXES = ("phiên", "bản", "phiên_bản", "version")

    def parse(self, raw_text, context=None):
        words = [w for w in raw_text.split() if w not in self.PREFIXES]
        if not words:
            raise ParseError("chỉ có tiền tố, không có phần số")
        parts, cur = [], []
        for w in words:
            if w in DOT_SEP_WORDS or w == ".":
                if not cur:
                    raise ParseError("dấu chấm không có phần số đứng trước")
                parts.append(cur)
                cur = []
            else:
                cur.append(w)
        if cur:
            parts.append(cur)
        if len(parts) < 2:
            raise ParseError("phiên bản cần ít nhất hai thành phần")
        return ".".join(str(read_number_auto(p)) for p in parts)
