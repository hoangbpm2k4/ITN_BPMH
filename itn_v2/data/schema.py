"""Hợp đồng dữ liệu V2: một bản ghi = một phát ngôn đã gán nhãn.

Nhãn được gắn trên **token ASR gốc**, không phải trên model word. Việc chiếu
sang model word do ``dataset`` làm lúc nạp, dùng đúng bộ tách từ mà lúc suy
luận sẽ dùng — nên train và inference không bao giờ lệch nhau.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

from ..labels import SEMANTIC_TYPES

PUNCT_LABELS = ("O", "COMMA", "PERIOD", "QUESTION")
DIGIT_RE = re.compile(r"\d")


@dataclass
class Sample:
    id: str
    spoken: str                              # văn bản dạng nói (đầu vào)
    written: str                             # dạng viết (đầu ra vàng)
    words: List[str]                         # token ASR gốc
    spans: List[Tuple[int, int, str]]        # (start, end, TYPE) trên token gốc
    punct: List[str]                         # nhãn dấu câu, một nhãn mỗi token gốc
    types: List[str] = field(default_factory=list)
    # Dạng viết THẬT lấy từ nguồn (bài báo gốc). Khác `written` — cái đó là đầu
    # ra của pipeline tất định. Giữ cả hai để đo được khoảng cách thật: `written`
    # dùng làm đích huấn luyện nhất quán, `written_source` dùng để biết taxonomy
    # còn cách văn bản thật bao xa.
    written_source: str = ""
    leak_key: str = ""                       # khoá chống rò rỉ giữa các tập
    source: str = ""

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


def validate_sample(sample: Sample):
    """Trả danh sách lỗi. Rỗng nghĩa là bản ghi hợp lệ."""
    errors = []
    if len(sample.words) != len(sample.punct):
        errors.append(f"số từ {len(sample.words)} khác số nhãn dấu câu {len(sample.punct)}")
    if any(DIGIT_RE.search(w) for w in sample.words):
        errors.append("dạng nói không được chứa chữ số")
    for p in sample.punct:
        if p not in PUNCT_LABELS:
            errors.append(f"nhãn dấu câu lạ: {p!r}")
    last_end = -1
    for start, end, type_name in sample.spans:
        if type_name not in SEMANTIC_TYPES:
            errors.append(f"kiểu không có trong taxonomy: {type_name!r}")
        if not 0 <= start <= end < len(sample.words):
            errors.append(f"span ({start},{end}) ngoài phạm vi")
        elif start <= last_end:
            errors.append(f"span ({start},{end}) chồng lấn span trước")
        last_end = max(last_end, end)
    # Câu KHÔNG có span là dữ liệu hợp lệ và cần thiết: nó dạy mô hình đừng
    # chuẩn hoá bậy (spec §23 hard negatives, §26.3 False Normalization Rate).
    return errors


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                d["spans"] = [tuple(s) for s in d["spans"]]
                yield Sample(**d)


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.to_json() + "\n")
