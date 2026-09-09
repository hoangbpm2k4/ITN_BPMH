"""Dựng bộ đánh giá đầu-cuối từ data_v1/data (hai zip ghép cặp).

`data_ITN.zip`  = dạng VIẾT (bản gốc có chữ số, viết hoa, viết tắt).
`data.zip`      = dạng NÓI  (chữ thường, số đọc thành chữ, viết tắt mở rộng).

Hai cây thư mục cùng cấu trúc 33 chủ đề nên ghép được theo tên. Bộ này KHÔNG
có nhãn span; nó dùng để đo đầu-cuối: chạy pipeline trên dạng nói rồi so với
dạng viết. Đây là thước đo sát thực tế nhất vì không phụ thuộc nhãn của ta.
"""

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def strip_key(name):
    """Khoá ghép cặp: bỏ dấu, hạ chữ thường, gộp khoảng trắng."""
    text = unicodedata.normalize("NFD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def sentences(text):
    out = []
    for part in SENT_SPLIT.split(text.replace("\n", " ")):
        part = part.strip()
        if len(part.split()) >= 3:
            out.append(part)
    return out


def collect(root):
    """{(chủ_đề, số_hiệu): nội dung}"""
    found = {}
    for path in root.rglob("*.txt"):
        topic = strip_key(path.parent.name)
        m = re.search(r"\((\d+)\)", path.stem)
        idx = m.group(1) if m else "1"
        found[(topic, idx)] = path.read_text(encoding="utf-8", errors="replace")
    return found


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--written-root", required=True)
    parser.add_argument("--spoken-root", required=True)
    parser.add_argument("--out", default="datasets_v2/dv1_eval.jsonl")
    args = parser.parse_args(argv)

    written = collect(Path(args.written_root))
    spoken = collect(Path(args.spoken_root))
    keys = sorted(set(written) & set(spoken))
    print(f"{len(written)} file viết · {len(spoken)} file nói · {len(keys)} ghép được")
    missing = sorted(set(written) ^ set(spoken))
    if missing:
        print(f"  không ghép được: {missing[:5]}")

    rows, skipped = [], 0
    for topic, idx in keys:
        ws, ss = sentences(written[(topic, idx)]), sentences(spoken[(topic, idx)])
        # Số câu hai bên có thể lệch (dấu chấm trong "www.x.vn" chẳng hạn),
        # nên ghép bằng độ tương đồng thay vì ghép theo thứ tự cứng.
        matcher = SequenceMatcher(None, [strip_key(x) for x in ws],
                                  [strip_key(x) for x in ss])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if i2 - i1 != j2 - j1:
                skipped += (i2 - i1)
                continue
            for w, s in zip(ws[i1:i2], ss[j1:j2]):
                if not s.strip() or not w.strip():
                    continue
                rows.append({"topic": topic, "file": idx, "spoken": s, "written": w})

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nghi {len(rows)} câu -> {args.out} (bỏ {skipped} câu không ghép được)")
    import collections
    for t, n in collections.Counter(r["topic"] for r in rows).most_common():
        print(f"  {n:4}  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
