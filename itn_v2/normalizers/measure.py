"""Đơn vị đo tổng quát + các lớp đo chuyên ngành hàng hải (spec §11, §4.3)."""

from ..catalog import get_units
from .base import Normalizer, ParseError
from .number import (DECIMAL_SEP_WORDS, DecimalParser, format_thousands,
                     read_cardinal, strip_negative)


def match_unit(words):
    """Khớp đơn vị ở CUỐI cụm. Trả (từ chỉ lượng, ký hiệu).

    Ưu tiên cụm dài: "độ xê" phải thắng "độ"; "<đơn vị> một giờ" phải được thử
    trước khi thử đơn vị trần.
    """
    units, per_unit, _ = get_units()
    for suffix in sorted(per_unit, key=lambda s: -len(s.split())):
        sw = suffix.split()
        if len(words) > len(sw) and words[-len(sw):] == sw:
            base = words[:-len(sw)]
            for uw, sym in units:
                if len(base) > len(uw) and base[-len(uw):] == uw:
                    return base[:-len(uw)], f"{sym}/{per_unit[suffix]}"
    for uw, sym in units:
        if len(words) > len(uw) and words[-len(uw):] == uw:
            return words[:-len(uw)], sym
    raise ParseError("không nhận ra đơn vị đo ở cuối cụm")


def format_quantity(words):
    if not words:
        raise ParseError("thiếu phần lượng trước đơn vị")
    if any(w in DECIMAL_SEP_WORDS for w in words):
        return DecimalParser().parse(" ".join(words))
    neg, plain = strip_negative(words)
    return ("-" if neg else "") + format_thousands(read_cardinal(plain))


def render(quantity: str, symbol: str) -> str:
    _, _, glue = get_units()
    if any(symbol.startswith(g) for g in glue):
        return f"{quantity}{symbol}"
    return f"{quantity} {symbol}"


class MeasureParser(Normalizer):
    name = "MeasureParser"

    def parse(self, raw_text, context=None):
        qty_words, symbol = match_unit(raw_text.split())
        return render(format_quantity(qty_words), symbol)


class _UnitConstrained(Normalizer):
    """Lớp đo chuyên ngành: chỉ chấp nhận một tập ký hiệu, có thể ánh xạ lại."""
    allowed = ()
    remap = {}

    def parse(self, raw_text, context=None):
        qty_words, symbol = match_unit(raw_text.split())
        symbol = self.remap.get(symbol, symbol)
        if self.allowed and symbol not in self.allowed:
            raise ParseError(f"đơn vị {symbol!r} không hợp với kiểu {self.name}")
        return render(format_quantity(qty_words), symbol)


class SpeedParser(_UnitConstrained):
    name = "SpeedParser"
    # "mười hai hải lý một giờ" -> NM/h -> ký hiệu chuẩn hàng hải là kn
    remap = {"NM/h": "kn", "NM": "kn"}
    allowed = ("kn", "km/h", "m/s", "m/h")


class DistanceParser(_UnitConstrained):
    name = "DistanceParser"
    allowed = ("NM", "km", "m", "cm", "mm", "dm", "fm")


class DepthParser(_UnitConstrained):
    name = "DepthParser"
    allowed = ("m", "cm", "fm")


class DraftParser(_UnitConstrained):
    name = "DraftParser"
    allowed = ("m", "cm")


class FrequencyParser(_UnitConstrained):
    name = "FrequencyParser"
    allowed = ("Hz", "kHz", "MHz", "GHz")


class ChannelParser(Normalizer):
    name = "ChannelParser"
    # Từ dẫn được GIỮ NGUYÊN theo người nói, không dịch. Bản trước quy mọi
    # từ dẫn về "kênh" nên "channel mười sáu" ra "kênh 16" trong khi bản gốc
    # viết "Channel 16" — sai 11/12 span CHANNEL chỉ vì một phép dịch.
    PREFIXES = {"kênh": "Kênh", "channel": "Channel", "ch": "Ch."}

    def parse(self, raw_text, context=None):
        from .number import read_number_auto
        words = raw_text.split()
        prefix = ""
        if words and words[0] in self.PREFIXES:
            prefix, words = self.PREFIXES[words[0]], words[1:]
        if not words:
            raise ParseError("thiếu số hiệu kênh")
        # "channel không sáu" là số hiệu hai chữ số có số 0 dẫn đầu, không
        # phải số đếm 6: đọc thành chuỗi chữ số thì mới ra "Channel 06".
        if len(words) > 1 and words[0] == "không":
            from .number import UNIT_DIGITS as DIGIT_WORDS
            if all(w in DIGIT_WORDS for w in words):
                value = "".join(str(DIGIT_WORDS[w]) for w in words)
                return f"{prefix} {value}".strip()
        value = read_number_auto(words)
        return f"{prefix} {value}".strip()


class MoneyParser(Normalizer):
    name = "MoneyParser"
    CURRENCIES = {
        "đồng": "đồng", "vnd": "VND", "việt nam đồng": "VND",
        "đô": "USD", "đô la": "USD", "đô la mỹ": "USD", "usd": "USD",
        "euro": "EUR", "ơ rô": "EUR", "yên": "JPY", "bảng": "GBP",
        "nhân dân tệ": "CNY", "tệ": "CNY",
    }

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        currency = None
        for spoken in sorted(self.CURRENCIES, key=lambda s: -len(s.split())):
            sw = spoken.split()
            if len(words) > len(sw) and words[-len(sw):] == sw:
                currency = self.CURRENCIES[spoken]
                words = words[:-len(sw)]
                break
        if currency is None:
            raise ParseError("không nhận ra đơn vị tiền tệ ở cuối cụm")
        neg, plain = strip_negative(words)
        if any(w in DECIMAL_SEP_WORDS for w in plain):
            amount = DecimalParser().parse(" ".join(plain))
        else:
            amount = format_thousands(read_cardinal(plain))
        return f"{'-' if neg else ''}{amount} {currency}"
