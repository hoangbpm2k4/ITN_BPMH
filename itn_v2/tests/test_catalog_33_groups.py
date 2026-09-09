"""Hồi quy cho toàn bộ 33 nhóm test hiện có, ánh xạ sang taxonomy V2 (spec §5).

Mỗi nhóm chạy qua ĐÚNG đường chạy thật: tách từ -> chiếu span -> normalizer
nhận văn bản ASR gốc -> validator -> cổng tin cậy -> kết xuất.
"""

import unittest

from .helpers import run

GROUPS = [
    # (số nhóm, tên nhóm, câu ASR, [(cụm, kiểu)], kết quả mong đợi)
    (1, "Đơn vị đo", "áp suất bốn phẩy năm bar",
     [("bốn phẩy năm bar", "MEASURE")], "Áp suất 4,5 bar."),
    (2, "Phần trăm", "đạt hai mươi lăm phần trăm",
     [("hai mươi lăm phần trăm", "PERCENT")], "Đạt 25%."),
    (3, "Tiền", "thiệt hại hai triệu đồng",
     [("hai triệu đồng", "MONEY")], "Thiệt hại 2.000.000 đồng."),
    (4, "Ngày", "xuất phát mười lăm tháng tám năm hai không hai sáu",
     [("mười lăm tháng tám năm hai không hai sáu", "DATE")],
     "Xuất phát 15/08/2026."),
    (5, "Giờ", "cập cảng lúc sáu giờ bốn mươi lăm",
     [("sáu giờ bốn mươi lăm", "TIME")], "Cập cảng lúc 06:45."),
    (6, "Múi giờ", "giờ địa phương u tê xê cộng bảy",
     [("u tê xê cộng bảy", "TIMEZONE")], "Giờ địa phương UTC+7."),
    (7, "ETA / ETD", "ê tê a tám giờ mười lăm",
     [("ê tê a tám giờ mười lăm", "ETA")], "ETA 08:15."),
    (8, "Toạ độ", "vị trí mười độ hai mươi lăm phút bắc",
     [("mười độ hai mươi lăm phút bắc", "COORD")], "Vị trí 10°25'N."),
    (9, "Nhận dạng tàu", "tàu số hai ba bốn năm sáu bảy tám chín không",
     [("hai ba bốn năm sáu bảy tám chín không", "MMSI_ID")],
     "Tàu số 234567890."),
    (10, "Viết tắt hàng hải", "thiết bị giê pê ét hoạt động tốt",
     [("giê pê ét", "ACRONYM")], "Thiết bị GPS hoạt động tốt."),
    (11, "VHF / kênh", "gọi trên vê hát ép kênh mười sáu",
     [("vê hát ép", "ACRONYM"), ("kênh mười sáu", "CHANNEL")],
     "Gọi trên VHF kênh 16."),
    (12, "Lệnh điều động", "thuyền trưởng ra lệnh đét xờ lâu a hét",
     [("đét xờ lâu a hét", "MARITIME_TERM")],
     "Thuyền trưởng ra lệnh Dead slow ahead."),
    (13, "Tín hiệu khẩn cấp", "phát tín hiệu mê đây",
     [("mê đây", "MARITIME_TERM")], "Phát tín hiệu MAYDAY."),
    (14, "Email", "gửi về operations@coastguard.vn",
     [("operations@coastguard.vn", "ELECTRONIC")],
     "Gửi về operations@coastguard.vn."),
    (15, "Website", "xem tại www.vinamarine.gov.vn",
     [("www.vinamarine.gov.vn", "ELECTRONIC")],
     "Xem tại www.vinamarine.gov.vn."),
    (16, "Địa chỉ mạng", "máy chủ 192.168.10.25",
     [("192.168.10.25", "ELECTRONIC")], "Máy chủ 192.168.10.25."),
    (17, "Điện thoại", "liên hệ không chín không tám một hai ba bốn năm sáu",
     [("không chín không tám một hai ba bốn năm sáu", "TELEPHONE")],
     "Liên hệ 0908123456."),
    (18, "Văn bản pháp lý", "theo số mười hai xẹt hai không hai tư nờ đê cê pê",
     [("số mười hai xẹt hai không hai tư nờ đê cê pê", "LEGAL_DOC_ID")],
     "Theo Số 12/2024/NĐ-CP."),
    (19, "Tên người", "thuyền trưởng nguyễn văn hùng báo cáo",
     [("nguyễn văn hùng", "PERSON_NAME")],
     "Thuyền trưởng Nguyễn Văn Hùng báo cáo."),
    (20, "Tỉnh / thành phố", "cập cảng hải phòng",
     [("hải phòng", "LOCATION_NAME")], "Cập cảng Hải Phòng."),
    (21, "Tên nước ngoài", "hệ thống pa tri ốt sẵn sàng",
     [("pa tri ốt", "FOREIGN_NAME")], "Hệ thống Patriot sẵn sàng."),
    (22, "Quý / Roman", "kế hoạch quý ba",
     [("quý ba", "QUARTER")], "Kế hoạch Quý III."),
    (23, "Số thứ tự", "xếp thứ hai toàn quân",
     [("thứ hai", "ORDINAL")], "Xếp thứ 2 toàn quân."),
    (24, "Phân số / tỷ lệ", "tỷ lệ một trên mười",
     [("một trên mười", "RATIO")], "Tỷ lệ 1:10."),
    (25, "Mã cảng", "mã cảng vê en hát pê hát",
     [("vê en hát pê hát", "PORT_CODE")], "Mã cảng VNHPH."),
    (26, "Vũ khí / trang bị", "biên đội xu ba lăm xuất kích",
     [("xu ba lăm", "EQUIPMENT_ID")], "Biên đội Su-35 xuất kích."),
    (27, "Vũ khí hải quân", "ngư lôi ét e tê sáu lăm",
     [("ét e tê sáu lăm", "EQUIPMENT_ID")], "Ngư lôi SET-65."),
    (28, "Acronym cơ quan", "văn bản của u bê en đê tỉnh",
     [("u bê en đê", "ACRONYM")], "Văn bản của UBND tỉnh."),
    (29, "Acronym quân sự", "lực lượng xê ét bê tuần tra",
     [("xê ét bê", "ACRONYM")], "Lực lượng CSB tuần tra."),
    (30, "Số âm", "nhiệt độ âm mười độ xê",
     [("âm mười độ xê", "MEASURE")], "Nhiệt độ -10°C."),
    (31, "Version", "phần mềm phiên bản một chấm hai chấm ba",
     [("phiên bản một chấm hai chấm ba", "VERSION")],
     "Phần mềm 1.2.3."),
    (32, "Địa chỉ", "trụ sở số mười hai đường Lê Lợi",
     [("số mười hai đường Lê Lợi", "ADDRESS")],
     "Trụ sở số 12 đường Lê Lợi."),
    (33, "Biển số xe", "xe biển ba mươi a một hai ba bốn năm",
     [("ba mươi a một hai ba bốn năm", "VEHICLE_PLATE")],
     "Xe biển 30A-12345."),
]


class TestAll33Groups(unittest.TestCase):
    pass


def _make(number, name, text, pairs, expected):
    def test(self):
        out, spans, _, conflicts = run(text, *pairs)
        self.assertEqual(conflicts, [], f"nhóm {number} bị từ ghép cắt ngang span")
        for s in spans:
            self.assertTrue(s.emitted,
                            f"nhóm {number} span {s.raw_span!r} không phát ra: {s.reason}")
        self.assertEqual(out, expected, f"nhóm {number} — {name}")
    test.__name__ = f"test_{number:02d}_{name.replace(' ', '_').replace('/', '_')}"
    return test


for _n, _name, _text, _pairs, _exp in GROUPS:
    _t = _make(_n, _name, _text, _pairs, _exp)
    setattr(TestAll33Groups, _t.__name__, _t)


class TestCoverage(unittest.TestCase):
    def test_du_33_nhom(self):
        self.assertEqual(len(GROUPS), 33)
        self.assertEqual(sorted(g[0] for g in GROUPS), list(range(1, 34)))


if __name__ == "__main__":
    unittest.main()
