"""Nhãn V2: 46 kiểu ngữ nghĩa + O, và 5 nhãn ranh giới BIOES.

Hai trục được dự đoán ĐỘC LẬP (spec §6):
  - ranh giới : B / I / E / S / O   -> CRF
  - kiểu      : 46 kiểu + O          -> softmax trên span

Không có nhãn ghép kiểu ``B-COORD``.
"""

# --- trục ranh giới ------------------------------------------------------
BOUNDARY_LABELS = ["O", "B", "I", "E", "S"]
BOUNDARY2ID = {l: i for i, l in enumerate(BOUNDARY_LABELS)}
ID2BOUNDARY = {i: l for l, i in BOUNDARY2ID.items()}

# Chuyển trạng thái hợp lệ: O | S | B -> I* -> E
_VALID_NEXT = {
    "O": {"O", "B", "S"},
    "S": {"O", "B", "S"},
    "E": {"O", "B", "S"},
    "B": {"I", "E"},
    "I": {"I", "E"},
}
_VALID_START = {"O", "B", "S"}
_VALID_END = {"O", "E", "S"}


def boundary_transition_mask():
    """Ma trận [5,5] với True = chuyển trạng thái hợp lệ. Dùng để mask CRF."""
    return [[ID2BOUNDARY[j] in _VALID_NEXT[ID2BOUNDARY[i]] for j in range(5)] for i in range(5)]


def boundary_start_mask():
    return [ID2BOUNDARY[i] in _VALID_START for i in range(5)]


def boundary_end_mask():
    return [ID2BOUNDARY[i] in _VALID_END for i in range(5)]


# --- trục kiểu ngữ nghĩa -------------------------------------------------
# Thứ tự đóng băng cho baseline V2 (spec §4). O luôn ở chỉ số 0.
NUMERIC_TYPES = [
    "CARDINAL", "ORDINAL", "DIGIT_SEQ", "DECIMAL", "FRACTION",
    "PERCENT", "RANGE", "RATIO", "MONEY", "MEASURE", "VERSION",
]
DATETIME_TYPES = [
    "DATE", "TIME", "TIMEZONE", "DURATION", "ETA", "ETD", "QUARTER",
]
MARITIME_TYPES = [
    "COORD", "HEADING", "BEARING", "SPEED", "DISTANCE",
    "DEPTH", "DRAFT", "FREQUENCY", "CHANNEL",
]
IDENTIFIER_TYPES = [
    "MMSI_ID", "IMO_ID", "CALLSIGN", "VESSEL_ID", "PORT_CODE",
    "DOCUMENT_ID", "LEGAL_DOC_ID", "TELEPHONE", "ELECTRONIC",
    "VEHICLE_PLATE", "ADDRESS",
]
ENTITY_TYPES = [
    "EQUIPMENT_ID", "EQUIPMENT_NAME", "FOREIGN_NAME", "PERSON_NAME",
    "LOCATION_NAME", "ACRONYM", "MARITIME_TERM", "RANK",
]

SEMANTIC_TYPES = NUMERIC_TYPES + DATETIME_TYPES + MARITIME_TYPES + IDENTIFIER_TYPES + ENTITY_TYPES
assert len(SEMANTIC_TYPES) == 46, f"taxonomy phải có đúng 46 kiểu, đang có {len(SEMANTIC_TYPES)}"

TYPE_LABELS = ["O"] + SEMANTIC_TYPES
TYPE2ID = {l: i for i, l in enumerate(TYPE_LABELS)}
ID2TYPE = {i: l for l, i in TYPE2ID.items()}
NUM_TYPES = len(TYPE_LABELS)  # 47

# Kiểu "quan trọng": sai một chữ số là hỏng cả bản tin (spec §26.2).
CRITICAL_TYPES = [
    "COORD", "HEADING", "BEARING", "MMSI_ID", "IMO_ID",
    "FREQUENCY", "EQUIPMENT_ID", "DATE", "TIME",
]

# Kiểu tất định: oracle phải đạt xấp xỉ 100% (spec §29).
DETERMINISTIC_TYPES = [
    t for t in SEMANTIC_TYPES
    if t not in {"EQUIPMENT_NAME", "FOREIGN_NAME", "PERSON_NAME",
                 "LOCATION_NAME", "ACRONYM", "MARITIME_TERM", "ADDRESS"}
]

# Kiểu dựa vào danh mục ngoài -> có thể trả fallback khi không tra được.
CATALOG_TYPES = ["EQUIPMENT_NAME", "FOREIGN_NAME", "ACRONYM", "MARITIME_TERM"]


def decode_spans(boundaries):
    """Chuỗi nhãn ranh giới -> danh sách (start, end) bao gồm cả hai đầu.

    Bỏ qua cấu trúc không hợp lệ thay vì ném lỗi: lúc suy luận CRF đã bị mask
    nhưng dữ liệu vàng lỗi hoặc chuỗi ghép cửa sổ vẫn có thể sinh chuỗi lạ.
    """
    spans, start = [], None
    for i, tag in enumerate(boundaries):
        if tag == "S":
            if start is not None:
                spans.append((start, i - 1))
                start = None
            spans.append((i, i))
        elif tag == "B":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "I":
            if start is None:
                start = i
        elif tag == "E":
            spans.append((start if start is not None else i, i))
            start = None
        else:  # O
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(boundaries) - 1))
    return spans


def encode_spans(spans, length):
    """Danh sách (start, end) -> chuỗi nhãn BIOES độ dài ``length``."""
    tags = ["O"] * length
    for s, e in spans:
        if s == e:
            tags[s] = "S"
        else:
            tags[s] = "B"
            for i in range(s + 1, e):
                tags[i] = "I"
            tags[e] = "E"
    return tags
