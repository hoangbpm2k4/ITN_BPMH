import re

COORD_RE = re.compile(r"^(\d+)°(?:(\d{2})')?(?:(\d{2})\")?([NSEW])?$")


def validate_coordinate(normalized):
    """Vĩ độ 0-90, kinh độ 0-180, phút/giây 0-59 (spec §15.2)."""
    m = COORD_RE.match(normalized or "")
    if not m:
        return False, f"{normalized!r} không đúng khuôn toạ độ"
    degree = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    second = int(m.group(3)) if m.group(3) else 0
    direction = m.group(4)
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
