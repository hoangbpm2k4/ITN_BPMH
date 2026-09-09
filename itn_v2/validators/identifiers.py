import re


def validate_mmsi(normalized):
    """MMSI đúng 9 chữ số (spec §15.3)."""
    if not re.fullmatch(r"\d{9}", normalized or ""):
        return False, f"MMSI phải đúng 9 chữ số, nhận được {normalized!r}"
    return True, ""


def validate_imo(normalized):
    digits = re.sub(r"\D", "", normalized or "")
    if len(digits) != 7:
        return False, f"số IMO phải có 7 chữ số, nhận được {len(digits)}"
    return True, ""


def validate_telephone(normalized):
    digits = normalized or ""
    if not re.fullmatch(r"\d{9,11}", digits):
        return False, f"số điện thoại phải có 9-11 chữ số, nhận được {digits!r}"
    if not digits.startswith("0"):
        return False, "số điện thoại nội địa phải bắt đầu bằng 0"
    return True, ""


def validate_port_code(normalized):
    if not re.fullmatch(r"[A-Z]{5}", normalized or ""):
        return False, f"mã cảng UN/LOCODE phải là 5 chữ cái, nhận được {normalized!r}"
    return True, ""


def validate_vehicle_plate(normalized):
    if not re.fullmatch(r"\d{2}[A-Z]{1,2}-\d{4,5}", normalized or ""):
        return False, f"{normalized!r} không đúng khuôn biển số"
    return True, ""
