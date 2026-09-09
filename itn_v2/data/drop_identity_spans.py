"""Loại câu có span mà chuẩn hoá KHÔNG đổi gì so với đầu vào.

Những câu này đến từ dữ liệu seed sinh bằng Gemma: URL và email được để nguyên
dạng viết trong `spoken`, nên đầu vào đã chứa sẵn đáp án. Giữ lại thì vừa dạy
mô hình chép lại, vừa thổi phồng điểm — F1 của ELECTRONIC từng đạt 0,93 mà
14/15 span dev là loại này.

Chỉ tính span mà normalizer trả về CHUỖI Y HỆT đầu vào. Span chỉ khác chữ hoa
(`việt nam` -> `Việt Nam`) là việc thật, không bị loại.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ..registry import get_normalizer

ROOT = Path(__file__).resolve().parents[2]


def identity_spans(record):
    out = []
    for start, end, type_name in record["spans"]:
        chunk = " ".join(record["words"][start:end + 1]).replace("_", " ")
        fn = get_normalizer(type_name)
        if fn is None:
            continue
        try:
            res = fn(chunk, type_name)
        except Exception:
            continue
        if res.valid and res.normalized == chunk:
            out.append((chunk, type_name))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for name in args.files:
        path = ROOT / name
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        keep, dropped = [], Counter()
        for r in rows:
            bad = identity_spans(r)
            if bad:
                for _, t in bad:
                    dropped[t] += 1
            else:
                keep.append(r)
        print(f"{name}: {len(rows)} -> {len(keep)} câu "
              f"(bỏ {len(rows)-len(keep)}; span: {dict(dropped)})")
        if not args.dry_run and len(keep) != len(rows):
            with path.open("w", encoding="utf-8") as fh:
                for r in keep:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
