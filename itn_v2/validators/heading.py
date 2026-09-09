import re

HEADING_RE = re.compile(r"^(\d{3})°$")


def validate_heading(normalized):
    """0 <= hướng <= 359, luôn ba chữ số (spec §15.1)."""
    m = HEADING_RE.match(normalized or "")
    if not m:
        return False, f"{normalized!r} không đúng khuôn ba chữ số kèm ký hiệu độ"
    value = int(m.group(1))
    if not 0 <= value <= 359:
        return False, f"hướng {value} ngoài khoảng 0-359"
    return True, ""
