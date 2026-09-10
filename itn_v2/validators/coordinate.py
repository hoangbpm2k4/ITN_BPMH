import re

# Chấp nhận cả nháy ASCII lẫn dấu nguyên U+2032/U+2033: dữ liệu cũ dùng ASCII,
# bản gốc thật và dữ liệu mới dùng dấu nguyên.
# Phút có thể kèm phần thập phân: khuôn độ-phút-thập-phân 10°25.500′N mà hải đồ
# điện tử dùng cho waypoint, bên cạnh khuôn độ-phút-giây 10°25′30″N.
ONE_RE = re.compile(r"(\d+)°(?:(\d{2})(?:\.(\d{1,3}))?[′'])?(?:(\d{2})[″\"])?([NSEW])?")
COORD_RE = re.compile(rf"^{ONE_RE.pattern}(?:,\s*{ONE_RE.pattern})*$")


def _check_one(degree, minute, minute_fraction, second, direction):
    degree = int(degree)
    minute = int(minute) if minute else 0
    second = int(second) if second else 0
    if direction in {"N", "S"} and not 0 <= degree <= 90:
        return False, f"vĩ độ {degree} ngoài khoảng 0-90"
    if direction in {"E", "W"} and not 0 <= degree <= 180:
        return False, f"kinh độ {degree} ngoài khoảng 0-180"
    if direction is None and not 0 <= degree <= 180:
        return False, f"độ {degree} ngoài khoảng 0-180"
    if not 0 <= minute < 60:
        return False, f"phút {minute} ngoài khoảng 0-59"
    if not 0 <= second < 60:
        return False, f"giây {second} ngoài khoảng 0-59"
    return True, ""


def validate_coordinate(normalized):
    """Vĩ độ 0-90, kinh độ 0-180, phút/giây 0-59 (spec §15.2).

    Một span toạ độ thật thường là CẶP "vĩ độ, kinh độ" nên phải kiểm từng vế.
    """
    text = normalized or ""
    if not COORD_RE.match(text):
        return False, f"{normalized!r} không đúng khuôn toạ độ"
    parts = [p.strip() for p in text.split(",")]
    for part in parts:
        m = ONE_RE.fullmatch(part)
        if not m:
            return False, f"{part!r} không đúng khuôn toạ độ"
        ok, why = _check_one(*m.groups())
        if not ok:
            return False, why
    return True, ""
