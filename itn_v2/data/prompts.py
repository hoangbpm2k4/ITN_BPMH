"""Mẫu prompt sinh dữ liệu, chia theo nhóm kiểu để phủ đủ 46 lớp.

Ví dụ trong prompt được lấy từ bộ test đã kiểm chứng, nên mô hình bắt chước
đúng những khuôn mà bộ chuẩn hoá tất định thực sự đọc được.
"""

TYPE_GROUPS = {
    "so_co_ban": {
        "types": ["CARDINAL", "ORDINAL", "DIGIT_SEQ", "DECIMAL",
                  "FRACTION", "PERCENT", "RANGE", "RATIO"],
        "examples": [
            "đơn vị tiếp nhận [[hai trăm năm mươi|CARDINAL]] bộ quân trang .",
            "xếp [[thứ hai|ORDINAL]] toàn quân khu .",
            "mã số hiệu [[không ba một hai|DIGIT_SEQ]] đã được cấp .",
            "sai số [[ba phẩy năm|DECIMAL]] so với thiết kế .",
            "chỉ còn [[một phần hai|FRACTION]] cơ số đạn .",
            "quân số đạt [[hai mươi lăm phần trăm|PERCENT]] .",
            "huấn luyện [[từ mười đến hai mươi|RANGE]] ngày .",
            "tỷ lệ tổn thất [[một trên mười|RATIO]] .",
        ],
    },
    "tien_do_phienban": {
        "types": ["MONEY", "MEASURE", "VERSION"],
        "examples": [
            "kinh phí [[hai triệu đồng|MONEY]] cho đợt diễn tập .",
            "áp suất buồng máy [[bốn phẩy năm bar|MEASURE]] .",
            "nhiệt độ ngoài trời [[âm mười độ xê|MEASURE]] .",
            "sản lượng [[tám mươi lăm tấn một giờ|MEASURE]] .",
            "phần mềm điều khiển [[phiên bản một chấm hai chấm ba|VERSION]] .",
        ],
    },
    "ngay_gio": {
        "types": ["DATE", "TIME", "TIMEZONE", "DURATION", "ETA", "ETD", "QUARTER"],
        "examples": [
            "xuất phát [[mười lăm tháng tám năm hai không hai sáu|DATE]] .",
            "tập trung lúc [[sáu giờ bốn mươi lăm|TIME]] .",
            "giờ địa phương [[u tê xê cộng bảy|TIMEZONE]] .",
            "hành quân liên tục [[hai giờ ba mươi phút|DURATION]] .",
            "[[ê tê a tám giờ mười lăm|ETA]] tại khu vực tập kết .",
            "[[ê tê đê mười sáu giờ bốn mươi|ETD]] rời cảng .",
            "kế hoạch [[quý ba|QUARTER]] đã được phê duyệt .",
        ],
    },
    "hang_hai": {
        "types": ["COORD", "HEADING", "BEARING", "SPEED", "DISTANCE",
                  "DEPTH", "DRAFT", "FREQUENCY", "CHANNEL"],
        "examples": [
            "vị trí [[mười độ hai mươi lăm phút bắc|COORD]] .",
            "tàu giữ hướng [[không chín không|HEADING]] .",
            "phương vị mục tiêu [[một hai không|BEARING]] .",
            "tốc độ [[mười hai hải lý một giờ|SPEED]] .",
            "cách bờ [[mười hải lý|DISTANCE]] .",
            "độ sâu luồng [[ba mươi mét|DEPTH]] .",
            "mớn nước [[tám phẩy hai mét|DRAFT]] .",
            "liên lạc trên tần số [[một hai một phẩy năm mê ga héc|FREQUENCY]] .",
            "trực canh [[kênh mười sáu|CHANNEL]] .",
        ],
    },
    "dinh_danh": {
        "types": ["MMSI_ID", "IMO_ID", "CALLSIGN", "VESSEL_ID", "PORT_CODE",
                  "DOCUMENT_ID", "LEGAL_DOC_ID", "TELEPHONE", "VEHICLE_PLATE", "ADDRESS"],
        "examples": [
            "tàu mang số [[hai ba bốn năm sáu bảy tám chín không|MMSI_ID]] .",
            "số [[i em ô chín không bảy bốn bảy hai chín|IMO_ID]] đã đăng kiểm .",
            "hô hiệu [[ba vê hát a|CALLSIGN]] .",
            "cảng đích [[vê en hát pê hát|PORT_CODE]] .",
            "theo [[số mười hai xẹt hai không hai tư|DOCUMENT_ID]] .",
            "căn cứ [[số mười hai xẹt hai không hai tư nờ đê cê pê|LEGAL_DOC_ID]] .",
            "liên hệ [[không chín không tám một hai ba bốn năm sáu|TELEPHONE]] .",
            "xe biển [[ba mươi a một hai ba bốn năm|VEHICLE_PLATE]] .",
            "trụ sở [[số mười hai đường lê lợi|ADDRESS]] .",
        ],
    },
    "dien_tu": {
        "types": ["ELECTRONIC"],
        "examples": [
            "gửi báo cáo về [[operations@coastguard.vn|ELECTRONIC]] .",
            "tra cứu tại [[www.vinamarine.gov.vn|ELECTRONIC]] .",
            "máy chủ nội bộ [[192.168.10.25|ELECTRONIC]] .",
        ],
    },
    "thuc_the": {
        "types": ["EQUIPMENT_ID", "EQUIPMENT_NAME", "FOREIGN_NAME",
                  "PERSON_NAME", "LOCATION_NAME", "ACRONYM", "MARITIME_TERM", "RANK"],
        "examples": [
            "biên đội [[xu ba lăm|EQUIPMENT_ID]] xuất kích .",
            "trực thăng [[mi mười bảy|EQUIPMENT_ID]] hạ cánh .",
            "hệ thống [[pa tri ốt|EQUIPMENT_NAME]] chuyển trạng thái .",
            "hội đàm tại [[oa sinh tơn|FOREIGN_NAME]] .",
            "thuyền trưởng [[nguyễn văn hùng|PERSON_NAME]] báo cáo .",
            "tàu cập cảng [[hải phòng|LOCATION_NAME]] .",
            "liên lạc qua [[vê hát ép|ACRONYM]] .",
            "thuyền trưởng ra lệnh [[đét xờ lâu a hét|MARITIME_TERM]] .",
            "[[đại tá|RANK]] chủ trì hội nghị .",
        ],
    },
}

HARD_NEGATIVE_GROUP = {
    "types": ["CARDINAL", "CHANNEL", "SPEED", "DISTANCE", "HEADING", "ORDINAL"],
    "examples": [
        "khu vực có [[mười hai|CARDINAL]] tàu cá đang hoạt động .",
        "chuyển sang [[kênh mười hai|CHANNEL]] để liên lạc .",
        "tốc độ [[mười hai hải lý một giờ|SPEED]] .",
        "cách bờ [[mười hai hải lý|DISTANCE]] .",
        "giữ hướng [[một hai không|HEADING]] .",
    ],
}

BASE_PROMPT = """Bạn tạo dữ liệu huấn luyện cho hệ chuẩn hoá văn bản tiếng Việt (ITN).

VIẾT {n} CÂU tiếng Việt ở DẠNG NÓI — đúng như máy nhận dạng tiếng nói trả ra:
- toàn bộ chữ thường
- MỌI con số viết bằng chữ, tuyệt đối không dùng ký tự số
- chủ đề: hàng hải, biên phòng, quân sự Việt Nam
- câu dài 8 đến 25 từ, văn phong báo cáo nghiệp vụ

Đánh dấu cụm cần chuẩn hoá bằng [[nội dung|KIỂU]].
CHỈ dùng các kiểu sau: {types}
Dấu câu là token riêng cách bằng dấu trắng: , . ?
Mỗi câu kết thúc bằng dấu . hoặc ?

Bám sát các khuôn dưới đây, chỉ đổi nội dung xung quanh và giá trị bên trong:
{examples}

{extra}
In ra {n} dòng câu, mỗi dòng một câu."""

STYLE_HINT = """Tham khảo văn phong của các câu nghiệp vụ thật sau (đây là dạng VIẾT,
bạn phải viết lại thành dạng NÓI):
{seeds}
"""

HARD_NEGATIVE_HINT = """QUAN TRỌNG: tạo các cặp ĐỐI CHỨNG — cùng một cụm số nhưng
ngữ cảnh khác nhau thì kiểu khác nhau. Mô hình phải học nghĩa từ ngữ cảnh, không
phải chỉ nhận mặt chữ số.
"""


def build_prompt(group, n=12, seeds=None, hard_negative=False):
    extra = ""
    if seeds:
        extra += STYLE_HINT.format(seeds="\n".join(f"- {s}" for s in seeds))
    if hard_negative:
        extra += HARD_NEGATIVE_HINT
    return BASE_PROMPT.format(
        n=n,
        types=" ".join(group["types"]),
        examples="\n".join(group["examples"]),
        extra=extra,
    )


# --- prompt có ràng buộc cho lớp đóng ------------------------------------
# Ba kiểu EQUIPMENT_ID / EQUIPMENT_NAME / IMO_ID trượt hết ở vòng thử vì mô hình
# tự bịa số hiệu không có trong danh mục và không đúng số chữ số. Với lớp đóng,
# phải NẠP SẴN cách đọc hợp lệ vào prompt thay vì để mô hình sáng tác.

def catalog_spoken_forms(name, limit=None):
    """Lấy các cách đọc hợp lệ từ danh mục ngoài."""
    from ..catalog import load_raw
    raw = load_raw(name)
    forms = []
    for entry in raw.get("entries", []) + raw.get("models", []):
        for spoken in entry.get("spoken", []):
            forms.append((spoken, entry["canonical"]))
    return forms[:limit] if limit else forms


CONSTRAINED_PROMPT = """Bạn tạo dữ liệu huấn luyện cho hệ chuẩn hoá văn bản tiếng Việt (ITN).

VIẾT {n} CÂU tiếng Việt ở DẠNG NÓI (chữ thường, chủ đề hàng hải/quân sự Việt Nam).

QUY TẮC TUYỆT ĐỐI:
- KHÔNG được dùng bất kỳ ký tự số nào (0 1 2 3 4 5 6 7 8 9). Mọi con số viết bằng chữ.
- Cụm bên trong [[...]] phải LẤY NGUYÊN VĂN từ danh sách dưới đây, không được tự chế.
- Chỉ thay đổi phần câu XUNG QUANH cụm được đánh dấu.

CÁC CỤM ĐƯỢC PHÉP DÙNG (chọn ngẫu nhiên, mỗi câu một hoặc hai cụm):
{allowed}

Dấu câu là token riêng cách bằng dấu trắng: , . ?
Mỗi câu kết thúc bằng . hoặc ?

VÍ DỤ:
{examples}

In ra {n} dòng câu, mỗi dòng một câu."""


def build_constrained_prompt(allowed_pairs, examples, n=12):
    """allowed_pairs: [(cách đọc, dạng chuẩn, KIỂU)]"""
    lines = [f"- [[{spoken}|{type_name}]]   (sẽ thành {canonical})"
             for spoken, canonical, type_name in allowed_pairs]
    return CONSTRAINED_PROMPT.format(
        n=n, allowed="\n".join(lines), examples="\n".join(examples))
