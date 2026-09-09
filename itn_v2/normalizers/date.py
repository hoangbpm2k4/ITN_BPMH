"""Ngày, giờ, múi giờ, thời lượng, ETA/ETD, quý.

Khác V1 ở một điểm bắt buộc (spec §4.2): **DateParser KHÔNG được đòi từ 'ngày'
nằm trong span**. V1 đòi, nên "hai mươi tám tháng bốn" mất phần ngày, im lặng,
trên 23,9% số cụm DATE của tập huấn luyện.
"""

from .base import Normalizer, ParseError
from .number import (UNIT_DIGITS, read_cardinal, read_digit_string,
                     read_number_auto, read_year, strip_negative)

DAY_TRIGGERS = {"ngày", "mùng", "mồng"}
HOUR_WORDS = {"giờ", "h"}
MINUTE_WORDS = {"phút"}
SECOND_WORDS = {"giây"}


def _read_small_numbers(words, i):
    """Mọi cách đọc MỘT số 1..99 bắt đầu tại i, dài trước ngắn sau.

    Trả danh sách chứ không trả một đáp án, vì "ba mươi tám" vừa có thể là 38
    vừa có thể là 30 rồi 8 — chỉ khi ghép với phần còn lại mới biết cách nào
    đúng. Cần hàm này vì read_number_auto âm thầm bỏ qua từ thừa: nó đọc
    ["sáu","mười"] ra 10 và ["ba","mươi","mốt","mười"] cũng ra 10.
    """
    if i >= len(words):
        return []
    w = words[i]
    nxt = words[i + 1] if i + 1 < len(words) else None
    after = words[i + 2] if i + 2 < len(words) else None
    real_unit = lambda x: x in UNIT_DIGITS and x not in {"không", "linh"}

    if w == "mười":                                  # 10, hoặc 11..19
        return ([(10 + UNIT_DIGITS[nxt], 2)] if real_unit(nxt) else []) + [(10, 1)]

    if real_unit(w):
        tens = UNIT_DIGITS[w]
        if nxt == "mươi":                            # 20..99
            return ([(tens * 10 + UNIT_DIGITS[after], 3)] if real_unit(after) else []) \
                   + [(tens * 10, 2)]
        return [(tens, 1)]                           # 1..9

    return []


def _split_two_numbers(head):
    """Tách `head` thành ĐÚNG hai số liền nhau, không được thừa từ nào."""
    for value_a, used_a in _read_small_numbers(head, 0):
        for value_b, used_b in _read_small_numbers(head, used_a):
            if used_a + used_b != len(head):
                continue
            if 1 <= value_a <= 31 and 1 <= value_b <= 12:
                return value_a, value_b, used_a
    return None


def _year_tail_start(words):
    """Vị trí bắt đầu phần đọc NĂM nằm ở cuối cụm, hoặc None.

    'nghìn'/'ngàn' là mỏ neo chắc chắn: từ ngay trước nó là hàng nghìn. Nếu
    không có, thử bốn từ cuối đọc rời từng chữ số ('một chín sáu tám').
    """
    for k in range(len(words) - 1, 0, -1):
        if words[k] in {"nghìn", "ngàn"}:
            return k - 1
    if len(words) >= 5 and all(w in UNIT_DIGITS for w in words[-4:]):
        return len(words) - 4
    return None


def _split_day_month_year(words):
    """Tách 'ngày tháng năm' viết liền không có từ dẫn nào.

    'tám chín một nghìn chín trăm sáu mươi chín' = 8/9/1969. Trước khi có hàm
    này, read_year lặng lẽ bỏ qua phần đầu và chỉ trả về 1969, nên 100 cụm
    DATE của corpus bị mất hẳn phần ngày và tháng.
    """
    start = _year_tail_start(words)
    if start is None or start < 2:
        return None
    head, tail = words[:start], words[start:]
    try:
        year = read_year(tail)
    except ParseError:
        return None
    if not 1000 <= year <= 2999:
        return None
    pair = _split_two_numbers(head)
    if pair is None:
        return None
    _day, _month, used_a = pair
    return head[:used_a], head[used_a:], tail


def _is_year_marker(words, k):
    """'năm' ở vị trí k có phải từ dẫn mốc năm không, hay là chữ số 5.

    'một nghìn chín trăm năm mươi ba' = 1953: chữ 'năm' ở đây là số 5 của
    'năm mươi'. Trước khi có hàm này, DateParser cắt câu tại đó và hỏng với
    MỌI năm thuộc thập niên 50 — 1950..1959, 2050..2059.
    """
    if words[k] != "năm":
        return False
    if k + 1 >= len(words):
        return False
    if words[k + 1] in {"mươi", "mười"}:
        return False
    return True


class DateParser(Normalizer):
    name = "DateParser"

    @staticmethod
    def _split_month_year(rest):
        """Tách 'tháng <tháng> <năm>' khi không có từ dẫn 'năm'.

        Thử độ dài phần tháng từ ngắn đến dài (tháng nhiều nhất 2 từ: 'mười
        hai'). Chỉ chấp nhận khi phần đuôi đọc ra năm 4 chữ số hợp lệ, nên
        'tháng sáu mươi tám' hay 'tháng năm ngoái' không bị tách nhầm.
        """
        for month, cut in _read_small_numbers(rest, 0):
            if not 1 <= month <= 12:
                continue
            head, tail = rest[:cut], rest[cut:]
            try:
                year = read_year(tail)
            except ParseError:
                continue
            if 1000 <= year <= 2999:
                return head, tail
        return rest, []

    def parse(self, raw_text, context=None):
        words = [w for w in raw_text.split() if w not in {",", "."}]
        if words and words[0] in DAY_TRIGGERS:
            words = words[1:]
        while words and words[0] in DAY_TRIGGERS:
            words = words[1:]
        if not words:
            raise ParseError("cụm rỗng sau khi bỏ từ dẫn")

        if "tháng" in words:
            i = words.index("tháng")
            day_words = words[:i]
            rest = words[i + 1:]
            # Mốc 'năm' phải nằm sau vị trí đầu của phần tháng, nếu không
            # "tháng năm" (tháng 5) sẽ bị hiểu nhầm là mốc chỉ năm.
            j = next((k for k, w in enumerate(rest)
                      if _is_year_marker(rest, k) and k > 0), None)
            if j is not None:
                month_words = rest[:j]
                year_words = rest[j + 1:]
            else:
                # "tháng sáu một nghìn chín trăm sáu mươi tám": phần năm đứng
                # ngay sau tháng, KHÔNG có từ dẫn "năm". Đây là dạng nói phổ
                # biến nhất; V1 và bản đầu của V2 đều nuốt cả cụm vào tháng rồi
                # ném ParseError, khiến 239 cụm DATE bị loại âm thầm.
                month_words, year_words = self._split_month_year(rest)
        elif any(_is_year_marker(words, k) for k in range(len(words))):
            i = next(k for k in range(len(words)) if _is_year_marker(words, k))
            day_words, month_words, year_words = words[:i], [], words[i + 1:]
        else:
            day_words, month_words, year_words = words, [], []

        # Ngày+tháng+năm viết liền, không từ dẫn: "tám chín một nghìn chín
        # trăm sáu mươi chín" = 8/9/1969. Phải thử TRƯỚC nhánh năm đứng một
        # mình, vì read_year sẽ vui vẻ trả về 1969 và nuốt mất ngày lẫn tháng.
        if day_words and not month_words and not year_words:
            split = _split_day_month_year(day_words)
            if split is not None:
                day_words, month_words, year_words = split

        # Năm đứng một mình: "một nghìn chín trăm linh sáu mươi tám" -> 1968.
        # Không được để CardinalParser xử lý vì nó chèn dấu phân nhóm -> "1.968".
        if day_words and not month_words and not year_words:
            try:
                value = read_year(day_words)
            except ParseError:
                value = None
            if value is not None and 1000 <= value <= 2999:
                return str(value)

        parts = []
        if day_words:
            day = read_number_auto(day_words)
            if not 1 <= day <= 31:
                raise ParseError(f"ngày {day} ngoài khoảng 1-31")
            parts.append(f"{day:02d}")
        if month_words:
            month = read_number_auto(month_words)
            if not 1 <= month <= 12:
                raise ParseError(f"tháng {month} ngoài khoảng 1-12")
            parts.append(f"{month:02d}")
        if year_words:
            parts.append(str(read_year(year_words)))
        if not parts:
            raise ParseError("không tách được thành phần ngày tháng nào")
        return "/".join(parts)


class TimeParser(Normalizer):
    name = "TimeParser"

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        if "giờ" not in words and "h" not in words:
            raise ParseError("không có từ 'giờ'")
        i = next(k for k, w in enumerate(words) if w in HOUR_WORDS)
        hour_words, rest = words[:i], words[i + 1:]
        if not hour_words:
            raise ParseError("thiếu phần giờ")
        hour = read_number_auto(hour_words)
        if not 0 <= hour <= 23:
            raise ParseError(f"giờ {hour} ngoài khoảng 0-23")

        minute, second = 0, None
        if rest == ["rưỡi"]:
            minute = 30
            rest = []
        if rest:
            if "giây" in rest:
                k = rest.index("giây")
                sec_start = k - 1
                while sec_start > 0 and rest[sec_start - 1] in UNIT_DIGITS | {"mười", "mươi", "lẻ", "linh"}:
                    sec_start -= 1
                second = read_number_auto(rest[sec_start:k])
                rest = rest[:sec_start]
            rest = [w for w in rest if w not in MINUTE_WORDS]
            if rest:
                minute = read_number_auto(rest)
        if not 0 <= minute <= 59:
            raise ParseError(f"phút {minute} ngoài khoảng 0-59")
        out = f"{hour:02d}:{minute:02d}"
        if second is not None:
            if not 0 <= second <= 59:
                raise ParseError(f"giây {second} ngoài khoảng 0-59")
            out += f":{second:02d}"
        return out


class TimezoneParser(Normalizer):
    name = "TimezoneParser"
    SIGNS = {"cộng": "+", "dương": "+", "trừ": "-", "âm": "-"}

    def parse(self, raw_text, context=None):
        from ..catalog import get_catalog
        words = raw_text.split()
        idx = next((k for k, w in enumerate(words) if w in self.SIGNS), None)
        if idx is None:
            raise ParseError("không có dấu cộng/trừ của múi giờ")
        label_words, sign, offset_words = words[:idx], self.SIGNS[words[idx]], words[idx + 1:]
        if not offset_words:
            raise ParseError("thiếu độ lệch múi giờ")
        label, _ = get_catalog("acronyms").lookup(" ".join(label_words))
        label = label or "UTC"
        offset = read_number_auto(offset_words)
        if not 0 <= offset <= 14:
            raise ParseError(f"độ lệch {offset} ngoài khoảng 0-14")
        return f"{label}{sign}{offset}"


class _PrefixedTime(Normalizer):
    prefix = ""
    spoken_prefixes = ()

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        for sp in sorted(self.spoken_prefixes, key=lambda s: -len(s.split())):
            sw = sp.split()
            if words[:len(sw)] == sw:
                words = words[len(sw):]
                break
        if not words:
            raise ParseError("thiếu phần giờ sau tiền tố")
        return f"{self.prefix} {TimeParser().parse(' '.join(words))}"


class ETAParser(_PrefixedTime):
    name = "ETAParser"
    prefix = "ETA"
    spoken_prefixes = ("ê tê a", "i ti ây", "eta")


class ETDParser(_PrefixedTime):
    name = "ETDParser"
    prefix = "ETD"
    spoken_prefixes = ("ê tê đê", "i ti đi", "etd")


class DurationParser(Normalizer):
    name = "DurationParser"
    UNITS = [("giờ", "giờ"), ("phút", "phút"), ("giây", "giây"),
             ("ngày", "ngày"), ("tuần", "tuần"), ("tháng", "tháng"), ("năm", "năm")]

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        unit_names = {u for u, _ in self.UNITS}
        out, buf = [], []
        for w in words:
            if w in unit_names:
                if not buf:
                    raise ParseError(f"đơn vị {w!r} không có phần số đứng trước")
                out.append(f"{read_number_auto(buf)} {w}")
                buf = []
            else:
                buf.append(w)
        if not out:
            raise ParseError("không nhận ra đơn vị thời lượng")
        return " ".join(out)


class QuarterParser(Normalizer):
    name = "QuarterParser"
    ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}

    def parse(self, raw_text, context=None):
        words = [w for w in raw_text.split() if w not in {"quý", "quí"}]
        if not words:
            raise ParseError("thiếu số quý")
        value = read_number_auto(words)
        if value not in self.ROMAN:
            raise ParseError(f"quý {value} không hợp lệ")
        return f"Quý {self.ROMAN[value]}"
