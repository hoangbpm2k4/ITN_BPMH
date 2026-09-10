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


# Ba khuôn hợp lệ, tương ứng ba nhánh của TelephoneParser.
PHONE_SHAPES = (
    (r"0\d{8,10}", "nội địa"),
    (r"\+\d{1,3}(?: \d{2,4})+", "quốc tế"),
    (r"1(?:900|800) \d{4,6}", "đầu số dịch vụ"),
)


def validate_telephone(normalized):
    text = normalized or ""
    if not any(re.fullmatch(shape, text) for shape, _ in PHONE_SHAPES):
        names = ", ".join(name for _, name in PHONE_SHAPES)
        return False, f"{normalized!r} không khớp khuôn số điện thoại nào ({names})"
    return True, ""


def validate_port_code(normalized):
    if not re.fullmatch(r"[A-Z]{5}", normalized or ""):
        return False, f"mã cảng UN/LOCODE phải là 5 chữ cái, nhận được {normalized!r}"
    return True, ""


# Hai khuôn hợp lệ: có dấu chấm phân nhóm (30A-123.45) và không có (29A-23532).
# Bản trước chỉ nhận khuôn không dấu chấm, nên khi parser đã đọc được biển số
# thật thì validator lại bác — pipeline lặng lẽ trả về dạng nói. Cùng họ lỗi với
# validator toạ độ: cổng kiểm chứng chặn đúng thứ nó phải cho qua.
PLATE_SHAPES = (r"\d{2}[A-Z]{1,2}-\d{3}\.\d{2}", r"\d{2}[A-Z]{1,2}-\d{4,5}")


def validate_vehicle_plate(normalized):
    text = normalized or ""
    if not any(re.fullmatch(shape, text) for shape in PLATE_SHAPES):
        return False, f"{normalized!r} không đúng khuôn biển số"
    return True, ""
