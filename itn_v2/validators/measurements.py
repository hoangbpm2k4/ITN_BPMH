import re

MEASURE_RE = re.compile(r"^-?[\d.]+(?:,\d+)?\s?°?[A-Za-zµ°²³/%]+$")


def validate_measure_like(normalized):
    """Đại lượng đo phải có phần số và phần đơn vị."""
    text = (normalized or "").strip()
    if not text:
        return False, "rỗng"
    if not any(ch.isdigit() for ch in text):
        return False, f"{text!r} không có phần số"
    if not MEASURE_RE.match(text):
        return False, f"{text!r} không đúng khuôn 'lượng + đơn vị'"
    return True, ""
