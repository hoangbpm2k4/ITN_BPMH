"""Địa chỉ điện tử: EMAIL / URL / IPV4 / IPV6 / MAC (spec §12).

Đây là các *phân loại con tất định*, không phải lớp ngữ nghĩa riêng của mô hình.
Không dùng tra danh mục cho địa chỉ tự do.
"""

import re
import unicodedata

from ..catalog import get_letters
from .base import Normalizer, ParseError
from .identifier import DIGIT_LIKE
from .number import read_cardinal, read_digit_string

SYMBOL_WORDS = [
    ("vê kép vê kép vê kép", "www"), ("kép kép kép", "www"), ("ba vê kép", "www"),
    ("hát tê tê pê ét", "https"), ("hát tê tê pê", "http"),
    ("gạch dưới", "_"), ("gạch chân", "_"),
    ("gạch ngang", "-"), ("gạch nối", "-"),
    ("hai chấm", ":"),
    ("gạch chéo", "/"), ("sờ lát", "/"), ("xẹt", "/"),
    ("a còng", "@"), ("a móc", "@"), ("át", "@"), ("còng", "@"),
    ("chấm", "."), ("dot", "."), ("gạch", "-"),
]

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
URL_RE = re.compile(r"^(https?://)?(www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/\S*)?$")
IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
IPV6_RE = re.compile(r"^[0-9A-Fa-f:]+$")
MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


# Từ báo hiệu cách đọc GỘP thay vì đọc rời từng chữ số.
# Chỉ TỪ CẤU TRÚC mới báo hiệu đọc gộp. "lăm"/"mốt"/"nhăm" là biến thể đọc
# của chữ số rời ("hai lăm" = 2 rồi 5 = 25), đưa vào đây sẽ hỏng.
GROUPED_WORDS = {"mười", "mươi", "trăm", "nghìn", "ngàn", "triệu"}

# DIGIT_LIKE của identifier chỉ có chữ số rời, thiếu "trăm"/"nghìn" nên dãy
# "một trăm chín mươi sáu" bị đứt giữa chừng.
# "lẻ" đồng nghĩa hoàn toàn với "linh" nhưng không có trong UNIT_DIGITS,
# nên "hai trăm lẻ chín" từng đứt dãy và ra "200le9".
NUMBER_WORDS = DIGIT_LIKE | {"trăm", "nghìn", "ngàn", "triệu", "lẻ"}


def deaccent(word):
    """Bỏ dấu tiếng Việt: tên miền đọc thành tiếng Việt ("huấn luyện") phải
    ghép lại thành nhãn ASCII ("huanluyen")."""
    text = unicodedata.normalize("NFD", word)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("đ", "d").replace("Đ", "D")


def read_number_run(words):
    """Đọc một dãy từ chỉ số nằm giữa hai dấu chấm của địa chỉ.

    Hai cách đọc cùng tồn tại trong thực tế: "một chín hai" là đọc RỜI từng
    chữ số (192), còn "mười tám" là đọc GỘP (18). Trước đây parser luôn đọc
    rời nên "mười tám chấm một trăm chín mươi sáu" ra sai hoàn toàn.
    """
    if any(w in GROUPED_WORDS for w in words):
        try:
            return str(read_cardinal(words))
        except ParseError:
            pass
    return read_digit_string(words)


def detect_subtype(text):
    if EMAIL_RE.match(text):
        return "EMAIL"
    if MAC_RE.match(text):
        return "MAC"
    if IPV4_RE.match(text) and all(0 <= int(p) <= 255 for p in text.split(".")):
        return "IPV4"
    if "::" in text or (text.count(":") >= 2 and IPV6_RE.match(text)):
        return "IPV6"
    if URL_RE.match(text):
        return "URL"
    return None


class ElectronicParser(Normalizer):
    name = "ElectronicParser"

    # Tên giao thức đứng MỘT MÌNH, không kèm địa chỉ: "máy chủ dùng IPv6".
    # Đây không phải địa chỉ nên văn phạm địa chỉ bên dưới không đọc được; bản
    # trước ghép thành "ipphienban6" rồi ném ParseError.
    PROTOCOL_LEAD = ("giao thức", "chuẩn", "địa chỉ")
    PROTOCOL_VERSION = {"bốn": "4", "sáu": "6"}
    PROTOCOL_STEMS = ("i p v", "i p vê", "i p phiên bản", "ai pi vi", "ip vờ")

    def _protocol_label(self, raw_text):
        """Trả (nhãn giao thức, phần còn lại) — phần còn lại rỗng nếu nhãn đứng một mình."""
        text = " ".join(raw_text.split())
        for lead in self.PROTOCOL_LEAD:
            if text.startswith(lead + " "):
                text = text[len(lead) + 1:]
        for stem in self.PROTOCOL_STEMS:
            if not text.startswith(stem + " "):
                continue
            rest = text[len(stem) + 1:].split()
            if not rest:
                continue
            version = self.PROTOCOL_VERSION.get(rest[0])
            if version:
                return f"IPv{version}", " ".join(rest[1:])
        return None, ""

    def parse(self, raw_text, context=None):
        # Nhãn giao thức có thể đứng một mình ("máy chủ dùng IPv6") hoặc đứng
        # NGAY TRƯỚC địa chỉ ("IPv6 2001:db8:42e::162"). Bản trước chỉ nhận
        # trường hợp đầu nên mọi span kèm địa chỉ đều ghép dính thành
        # "ipv620012.db8..." rồi trượt.
        label, rest = self._protocol_label(raw_text)
        if label and not rest:
            return label
        if label:
            return f"{label} {self._parse_address(rest)}"
        return self._parse_address(raw_text)

    def _parse_address(self, raw_text):
        # ASR có thể đã trả về dạng viết sẵn -> chỉ cần xác nhận
        compact = raw_text.replace(" ", "")
        subtype = detect_subtype(compact)
        if subtype:
            return self._render(compact, subtype)

        # "hai chấm" vừa là dấu hai chấm, vừa là số 2 rồi dấu chấm. Không có
        # cách nào biết trước, nên thử cả hai và lấy cách cho ra địa chỉ hợp lệ.
        errors = []
        for colon_first in (True, False):
            try:
                return self._parse_once(raw_text, colon_first)
            except ParseError as exc:
                errors.append(str(exc))
        raise ParseError(errors[0])

    def _parse_once(self, raw_text, colon_first=True):
        symbols = SYMBOL_WORDS if colon_first else [
            (sp, sym) for sp, sym in SYMBOL_WORDS if sp != "hai chấm"]
        words = raw_text.split()
        letters = get_letters()
        out, i, n = [], 0, len(words)
        while i < n:
            for spoken, symbol in symbols:
                sw = spoken.split()
                if words[i:i + len(sw)] == sw:
                    out.append(symbol)
                    i += len(sw)
                    break
            else:
                if words[i] in NUMBER_WORDS:
                    j = i
                    while j < n and words[j] in NUMBER_WORDS:
                        # "hai" mở đầu cụm "hai chấm" (dấu :) thuộc về DẤU chứ
                        # không thuộc về số. Không dừng ở đây thì 2001 nuốt luôn
                        # chữ "hai" của dấu hai chấm và cả địa chỉ IPv6 hỏng.
                        # Chỉ áp dụng ở lượt thử CÓ dấu hai chấm; lượt kia đọc
                        # "chín mươi hai chấm" là 92 rồi dấu chấm, phải để yên.
                        if colon_first and j > i and words[j:j + 2] == ["hai", "chấm"]:
                            break
                        j += 1
                    out.append(read_number_run(words[i:j]))
                    i = j
                    continue
                matched = False
                for spoken, symbol in letters:
                    sw = spoken.split()
                    if words[i:i + len(sw)] == sw:
                        out.append(symbol.lower())
                        i += len(sw)
                        matched = True
                        break
                if matched:
                    continue
                if re.fullmatch(r"[A-Za-z0-9._%+:/@-]+", words[i]):
                    out.append(words[i].lower())
                    i += 1
                    continue
                # Âm tiết tiếng Việt: một phần của tên miền đọc thành lời
                # ("chấm huấn luyện chấm" -> ".huanluyen."). Ghép dính, bỏ dấu.
                plain = deaccent(words[i]).lower()
                if re.fullmatch(r"[a-z0-9]+", plain):
                    out.append(plain)
                    i += 1
                    continue
                raise ParseError(f"từ {words[i]!r} không thuộc văn phạm địa chỉ điện tử")
        text = "".join(out)
        subtype = detect_subtype(text)
        if subtype is None:
            raise ParseError(f"chuỗi {text!r} không khớp EMAIL/URL/IPV4/IPV6/MAC")
        return self._render(text, subtype)

    @staticmethod
    def _render(text, subtype):
        if subtype == "MAC":
            return text.upper()
        if subtype in {"EMAIL", "URL"}:
            return text.lower()
        return text
