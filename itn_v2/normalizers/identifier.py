"""Định danh có cấu trúc: MMSI, IMO, callsign, mã cảng, số hiệu văn bản."""

import re

from ..catalog import get_catalog, get_letters
from .base import Normalizer, ParseError
from .number import UNIT_DIGITS, read_digit_string, read_number_auto

DIGIT_LIKE = set(UNIT_DIGITS) | {"mười", "mươi"}
SLASH_WORDS = {"xẹt", "xẹc", "trên", "phần", "gạch chéo", "/"}
DASH_WORDS = {"gạch", "gạch ngang", "-"}


def read_alphanumeric(words) -> str:
    """Đọc chuỗi trộn chữ cái và chữ số: 'vê en hát pê hát' -> VNHPH."""
    letters = get_letters()
    out, i, n = [], 0, len(words)
    while i < n:
        for spoken, symbol in letters:
            sw = spoken.split()
            if words[i:i + len(sw)] == sw:
                out.append(symbol)
                i += len(sw)
                break
        else:
            if words[i] in DIGIT_LIKE:
                j = i
                while j < n and words[j] in DIGIT_LIKE:
                    j += 1
                out.append(read_digit_string(words[i:j]))
                i = j
            elif re.fullmatch(r"[A-Za-z0-9]+", words[i]):
                out.append(words[i].upper())
                i += 1
            else:
                raise ParseError(f"từ {words[i]!r} không đọc được thành chữ/số")
    if not out:
        raise ParseError("không đọc được ký tự nào")
    return "".join(out)


class _DigitIdentifier(Normalizer):
    """Định danh thuần chữ số: giữ nguyên độ dài và số 0 đầu."""
    prefix = ""
    spoken_prefixes = ()

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        had_prefix = False
        for sp in sorted(self.spoken_prefixes, key=lambda s: -len(s.split())):
            sw = sp.split()
            if words[:len(sw)] == sw:
                words = words[len(sw):]
                had_prefix = True
                break
        if not words:
            raise ParseError("thiếu phần chữ số")
        digits = read_digit_string(words)
        return f"{self.prefix} {digits}" if (had_prefix and self.prefix) else digits


class MMSIParser(_DigitIdentifier):
    name = "MMSIParser"
    spoken_prefixes = ("em em ét i", "mờ mờ ét i", "mmsi")


class IMOParser(_DigitIdentifier):
    name = "IMOParser"
    prefix = "IMO"
    spoken_prefixes = ("i em ô", "ai em ô", "imo")


class TelephoneParser(_DigitIdentifier):
    name = "TelephoneParser"
    spoken_prefixes = ("số điện thoại", "điện thoại", "số")


class CallsignParser(Normalizer):
    name = "CallsignParser"

    def parse(self, raw_text, context=None):
        words = [w for w in raw_text.split() if w not in {"hô", "hiệu", "hô_hiệu", "callsign"}]
        return read_alphanumeric(words)


class VesselIDParser(Normalizer):
    name = "VesselIDParser"

    def parse(self, raw_text, context=None):
        return read_alphanumeric(raw_text.split())


class PortCodeParser(Normalizer):
    name = "PortCodeParser"

    def parse(self, raw_text, context=None):
        canonical, _ = get_catalog("port_codes").lookup(raw_text)
        if canonical:
            return canonical
        return read_alphanumeric(raw_text.split())


class DocumentIDParser(Normalizer):
    """'số mười hai xẹt hai không hai tư' -> 'Số 12/2024'."""

    name = "DocumentIDParser"

    def parse(self, raw_text, context=None):
        words = raw_text.replace("gạch chéo", "xẹt").split()
        prefix = ""
        if words and words[0] == "số":
            prefix, words = "Số ", words[1:]
        if not words:
            raise ParseError("thiếu phần số hiệu")

        groups, cur, seps = [], [], []
        for w in words:
            if w in SLASH_WORDS:
                groups.append(cur); seps.append("/"); cur = []
            elif w in DASH_WORDS:
                groups.append(cur); seps.append("-"); cur = []
            else:
                cur.append(w)
        groups.append(cur)
        if any(not g for g in groups):
            raise ParseError("có dấu phân cách không kèm phần số")

        rendered = []
        for g in groups:
            if len(g) >= 3 and g[0] == "hai" and g[1] == "không":
                tail = read_number_auto(g[2:])
                rendered.append(f"20{tail:02d}" if 0 <= tail <= 99 else str(read_number_auto(g)))
            elif all(w in DIGIT_LIKE for w in g):
                rendered.append(str(read_number_auto(g)))
            else:
                rendered.append(read_alphanumeric(g))
        out = rendered[0]
        for sep, part in zip(seps, rendered[1:]):
            out += sep + part
        return prefix + out


class LegalDocumentParser(Normalizer):
    """'nờ đê cê pê' -> 'NĐ-CP'; kèm số hiệu -> 'Số 12/2024/NĐ-CP'."""

    name = "LegalDocumentParser"

    def parse(self, raw_text, context=None):
        catalog = get_catalog("legal_docs")
        words = raw_text.split()
        for k in range(len(words), 0, -1):
            canonical, score = catalog.lookup(" ".join(words[-k:]))
            if canonical and score >= 1.0:
                head = words[:-k]
                if not head:
                    return canonical
                return DocumentIDParser().parse(" ".join(head)) + "/" + canonical
        canonical, score = catalog.lookup(raw_text)
        if canonical:
            return canonical
        raise ParseError(f"không tra được ký hiệu văn bản (điểm khớp {score:.2f})")


class VehiclePlateParser(Normalizer):
    name = "VehiclePlateParser"

    def parse(self, raw_text, context=None):
        code = read_alphanumeric(raw_text.split())
        m = re.fullmatch(r"(\d{2})([A-Z]{1,2})(\d{4,5})", code)
        if not m:
            raise ParseError(f"chuỗi {code!r} không đúng khuôn biển số")
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"


class AddressParser(Normalizer):
    """Chuẩn hoá phần số trong địa chỉ và khôi phục hoa cho tên riêng."""

    name = "AddressParser"
    KEEP = {"đường", "phố", "ngõ", "hẻm", "quận", "huyện", "phường", "xã",
            "thành", "tỉnh", "số", "khu", "tổ", "thôn", "ấp", "đại", "lộ"}
    # Từ dẫn báo hiệu ngay sau nó là TÊN RIÊNG cần viết hoa
    # ("đường trường chinh" -> "đường Trường Chinh").
    NAME_TRIGGERS = {"đường", "phố", "ngõ", "hẻm", "quận", "huyện", "phường",
                     "xã", "tỉnh", "thôn", "ấp", "lộ"}
    # DIGIT_LIKE thiếu hàng trăm/nghìn nên "một trăm năm mươi tám" từng bị cắt
    # thành "1 trăm 58".
    NUMBER_WORDS = DIGIT_LIKE | {"trăm", "nghìn", "ngàn", "lẻ", "linh"}

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        out, buf = [], []
        naming = False           # đang ở trong một tên riêng?

        def flush():
            if buf:
                out.append(str(read_number_auto(buf)))
                buf.clear()

        for w in words:
            if w in self.NUMBER_WORDS:
                buf.append(w)
                continue
            flush()
            if w in self.KEEP:
                naming = w in self.NAME_TRIGGERS
                out.append(w)
            elif naming:
                out.append(w[:1].upper() + w[1:])
            else:
                out.append(w)
        flush()
        if not any(ch.isdigit() for ch in " ".join(out)):
            raise ParseError("địa chỉ không chứa thành phần số nào")
        return " ".join(out)
