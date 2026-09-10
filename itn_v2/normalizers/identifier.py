"""Định danh có cấu trúc: MMSI, IMO, callsign, mã cảng, số hiệu văn bản."""

import re

from ..catalog import get_catalog, get_letters
from .base import Normalizer, ParseError
from .number import UNIT_DIGITS, read_digit_string, read_number_auto

DIGIT_LIKE = set(UNIT_DIGITS) | {"mười", "mươi"}
# Số hiệu văn bản đọc theo SỐ NGUYÊN chứ không phải từng chữ số: "số một trăm
# sáu mươi lăm xẹt hai nghìn mười chín". Bản trước chỉ nhận chữ số rời nên gặp
# "trăm"/"nghìn" là rơi xuống read_alphanumeric rồi ném ParseError.
NUMBER_LIKE = DIGIT_LIKE | {"trăm", "nghìn", "ngàn", "triệu", "lẻ", "linh"}
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
    """Điện thoại nội địa, quốc tế và đầu số dịch vụ.

    Ba khuôn, phân biệt bằng chính dạng nói chứ không bằng ngoại lệ:
      - "cộng" mở đầu   -> số quốc tế  "+84 908 123 456"
      - đầu số 1900/1800 -> dịch vụ     "1900 1234"
      - còn lại          -> nội địa     "0908123456"

    Từ "cộng" được người đọc phát ra thành tiếng, nên đây là quy tắc đọc thật.
    Bản trước bỏ qua nó và trả về "84908123456", vừa sai khuôn vừa bị validator
    bác vì không mở đầu bằng số 0.
    """

    name = "TelephoneParser"
    spoken_prefixes = ("số điện thoại", "điện thoại", "số")
    PLUS_WORDS = {"cộng", "+"}
    # Mã quốc gia dài 2 chữ số cho vùng đang phục vụ; phần còn lại nhóm ba một.
    COUNTRY_CODE_LEN = 2
    SERVICE_PREFIXES = ("1900", "1800")

    @staticmethod
    def _group(digits, size=3):
        return " ".join(digits[i:i + size] for i in range(0, len(digits), size))

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        international = False
        if words and words[0] in self.PLUS_WORDS:
            international, words = True, words[1:]
        digits = super().parse(" ".join(words), context)
        if international:
            code = digits[:self.COUNTRY_CODE_LEN]
            rest = digits[self.COUNTRY_CODE_LEN:]
            if not rest:
                raise ParseError(f"số quốc tế {digits!r} thiếu phần thuê bao")
            return f"+{code} {self._group(rest)}"
        for prefix in self.SERVICE_PREFIXES:
            if digits.startswith(prefix) and len(digits) > len(prefix):
                return f"{prefix} {digits[len(prefix):]}"
        return digits


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

    # Năm ban hành hợp lệ. Ngoài khoảng này thì "năm" là chữ số 5.
    YEAR_RANGE = (1900, 2100)

    @classmethod
    def _is_year_separator(cls, word, current, rest):
        """"số hai mươi hai NĂM hai nghìn không trăm hai mươi tư" -> 22/2024.

        Người đọc số hiệu văn bản hay thay dấu "/" bằng chữ "năm" vì phần sau
        đúng là năm ban hành. Nhưng "năm" cũng là chữ số 5, và số hiệu hay được
        đọc từng chữ số: "một NĂM chín" = 159. Phân biệt bằng cách nhìn ra sau —
        chỉ là dấu phân cách khi phần theo sau đọc ra một NĂM thật.
        Không có điều kiện này thì "một năm chín" ra "1/9".
        """
        if word != "năm" or not current:
            return False
        # "năm mươi"/"năm mười" là số 50-59. Chỉ dựa vào phép thử năm ở dưới là
        # chưa đủ: "năm mươi năm hai nghìn không trăm mười sáu" thì phần đuôi
        # tính từ "mươi" vẫn đọc ra 2016, nên "năm" đầu bị cắt oan.
        if rest[:1] and rest[0] in {"mươi", "mười"}:
            return False
        run = []
        for w in rest:
            if w not in NUMBER_LIKE:
                break
            run.append(w)
        if not run:
            return False
        try:
            value = read_number_auto(run)
        except ParseError:
            return False
        return cls.YEAR_RANGE[0] <= value <= cls.YEAR_RANGE[1]

    def parse(self, raw_text, context=None):
        words = raw_text.replace("gạch chéo", "xẹt").split()
        prefix = ""
        if words and words[0] == "số":
            prefix, words = "Số ", words[1:]
        if not words:
            raise ParseError("thiếu phần số hiệu")

        groups, cur, seps = [], [], []
        for i, w in enumerate(words):
            if w in SLASH_WORDS or self._is_year_separator(w, cur, words[i + 1:]):
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
            elif all(w in NUMBER_LIKE for w in g):
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
    # Dấu gạch trong ký hiệu ("quy đê GẠCH NGANG tê tê giê") được đọc thành
    # tiếng. Dấu này ta tự sinh ở đầu ra nên bỏ đi trước khi tra danh mục.
    # Dấu bên trong ký hiệu được đọc thành tiếng theo nhiều lối: "en quy GẠCH
    # NGANG cê pê", "nờ đê XẸT cê pê", "quy đê TRÊN tê tê gờ".
    SYMBOL_DASH = ("dấu gạch ngang", "gạch ngang", "gạch nối", "dấu gạch",
                   "gạch chéo", "xẹt", "xẹc", "trên", "gạch")

    @classmethod
    def _drop_dash(cls, text):
        for dash in cls.SYMBOL_DASH:
            text = text.replace(" " + dash + " ", " ")
        return " ".join(text.split())

    def parse(self, raw_text, context=None):
        catalog = get_catalog("legal_docs")
        words = raw_text.split()
        for k in range(len(words), 0, -1):
            canonical, score = catalog.lookup(self._drop_dash(" ".join(words[-k:])))
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
    """"ba không a gạch ngang một hai ba chấm bốn năm" -> 30A-123.45

    Người đọc biển số Việt Nam đọc CẢ dấu phân cách. Bản trước ném thẳng cả cụm
    vào `read_alphanumeric`, vốn không biết "gạch ngang"/"chấm" là gì, nên mọi
    biển số thật đều rơi vào ParseError — 17/17 câu của chủ đề hỏng, âm thầm.
    """

    name = "VehiclePlateParser"
    # Người đọc hay chèn từ "dấu": "gạch chín chín bảy dấu chấm ba sáu".
    SEPARATORS = {"dấu gạch ngang": "-", "dấu gạch nối": "-", "dấu gạch": "-",
                  "gạch ngang": "-", "gạch nối": "-", "gạch": "-",
                  "dấu chấm": ".", "chấm": "."}
    # Có dấu: 30A-123.45. Không dấu: 29A-23532.
    SHAPES = (r"\d{2}[A-Z]{1,2}-\d{3}\.\d{2}", r"\d{2}[A-Z]{1,2}-\d{4,5}")

    @staticmethod
    def _read_group(words):
        """Nhóm giữa hai dấu: đọc từng chữ số, hoặc đọc theo số nguyên.

        "chín trăm bảy mươi sáu" = 976 — read_alphanumeric không biết "trăm"
        nên trước đây cả biển số hỏng.
        """
        if words and all(w in AddressParser.NUMBER_WORDS for w in words) \
                and any(w in {"trăm", "nghìn", "ngàn", "lẻ", "linh"} for w in words):
            return str(read_number_auto(words))
        return read_alphanumeric(words)

    def _read_with_separators(self, words):
        parts, buf, i, n = [], [], 0, len(words)
        while i < n:
            for spoken, symbol in self.SEPARATORS.items():
                sw = spoken.split()
                if words[i:i + len(sw)] == sw:
                    parts.append(self._read_group(buf) if buf else "")
                    parts.append(symbol)
                    buf = []
                    i += len(sw)
                    break
            else:
                buf.append(words[i])
                i += 1
        parts.append(self._read_group(buf) if buf else "")
        return "".join(parts)

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        has_sep = any(w in {"gạch", "chấm"} for w in words)
        code = (self._read_with_separators(words) if has_sep
                else read_alphanumeric(words))
        if not has_sep:
            m = re.fullmatch(r"(\d{2})([A-Z]{1,2})(\d{4,5})", code)
            if not m:
                raise ParseError(f"chuỗi {code!r} không đúng khuôn biển số")
            return f"{m.group(1)}{m.group(2)}-{m.group(3)}"
        if not any(re.fullmatch(shape, code) for shape in self.SHAPES):
            raise ParseError(f"chuỗi {code!r} không đúng khuôn biển số")
        return code


class AddressParser(Normalizer):
    """Chuẩn hoá phần số trong địa chỉ và khôi phục hoa cho tên riêng."""

    name = "AddressParser"
    KEEP = {"đường", "phố", "ngõ", "hẻm", "quận", "huyện", "phường", "xã",
            "tỉnh", "số", "khu", "tổ", "thôn", "ấp", "đại", "lộ"}
    # "thành" là từ dẫn CHỈ khi đi cùng "phố"; đứng lẻ nó là một tiếng trong
    # tên riêng ("Nguyễn Tất Thành") và phải được viết hoa.
    KEEP_IF_NEXT = {"thành": "phố"}
    # Từ dẫn báo hiệu ngay sau nó là TÊN RIÊNG cần viết hoa
    # ("đường trường chinh" -> "đường Trường Chinh").
    NAME_TRIGGERS = {"đường", "phố", "ngõ", "hẻm", "quận", "huyện", "phường",
                     "xã", "tỉnh", "thôn", "ấp", "lộ"}
    # Sau SỐ NHÀ là tên đường, kể cả khi không có từ dẫn "đường":
    # "số mười hai trần phú" -> "Số 12 Trần Phú". Bản gốc thật viết như vậy ở
    # cả 8/8 địa chỉ; bản trước để "số 12 trần phú" nên hỏng hết.
    HOUSE_NUMBER_WORDS = {"số"}
    # DIGIT_LIKE thiếu hàng trăm/nghìn nên "một trăm năm mươi tám" từng bị cắt
    # thành "1 trăm 58".
    NUMBER_WORDS = DIGIT_LIKE | {"trăm", "nghìn", "ngàn", "lẻ", "linh"}

    # Mã lô/khu là chuỗi CHỮ + SỐ có dấu nối, không phải số nhà: "lô bi tám
    # gạch tám bốn hai" -> "Lô B8-842". Văn phạm số nhà đọc "bi" thành chữ
    # thường rồi "tám" thành số, ra "lô bi 8 trừ 842".
    BLOCK_LEADS = {"lô": "Lô", "block": "Block", "khu": "Khu"}
    BLOCK_SEPARATORS = ("dấu gạch ngang", "gạch ngang", "gạch nối", "dấu gạch",
                        "gạch", "trừ")

    def _parse_block(self, lead, words):
        parts, buf = [], []
        i, n = 0, len(words)
        while i < n:
            for sep in self.BLOCK_SEPARATORS:
                sw = sep.split()
                if words[i:i + len(sw)] == sw:
                    parts.append(read_alphanumeric(buf) if buf else "")
                    parts.append("-")
                    buf = []
                    i += len(sw)
                    break
            else:
                buf.append(words[i])
                i += 1
        parts.append(read_alphanumeric(buf) if buf else "")
        code = "".join(parts)
        if not any(ch.isdigit() for ch in code):
            raise ParseError(f"mã lô {code!r} không có phần số")
        return f"{self.BLOCK_LEADS[lead]} {code}"

    def parse(self, raw_text, context=None):
        words = raw_text.split()
        if words and words[0] in self.BLOCK_LEADS and len(words) > 1:
            return self._parse_block(words[0], words[1:])
        out, buf = [], []
        naming = False           # đang ở trong một tên riêng?
        after_house_number = False

        def flush():
            nonlocal naming
            if buf:
                out.append(str(read_number_auto(buf)))
                buf.clear()
                if after_house_number:
                    naming = True     # từ kế tiếp là tên đường

        for i, w in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if w in self.KEEP_IF_NEXT and nxt == self.KEEP_IF_NEXT[w]:
                flush()
                naming = False
                after_house_number = False
                out.append(w)
                continue
            if w in self.NUMBER_WORDS:
                buf.append(w)
                continue
            flush()
            if w in self.KEEP:
                naming = w in self.NAME_TRIGGERS
                after_house_number = w in self.HOUSE_NUMBER_WORDS
                # "Số" mở đầu một địa chỉ thì viết hoa (8/8 địa chỉ trong bản gốc).
                out.append("Số" if (after_house_number and not out) else w)
            elif naming:
                out.append(w[:1].upper() + w[1:])
            else:
                after_house_number = False
                out.append(w)
        flush()
        if not any(ch.isdigit() for ch in " ".join(out)):
            raise ParseError("địa chỉ không chứa thành phần số nào")
        return " ".join(out)
