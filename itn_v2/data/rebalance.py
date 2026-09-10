"""Hạ tỉ trọng một lớp đang lấn át trong tập huấn luyện.

Bản v7.1 đổ 2.672 span EQUIPMENT_ID (71% toàn bộ phần thêm) mà không đổ gì
vào các lớp đang đói. Kết quả đo được trên data_v1: EQUIPMENT_ID +13 span
đúng, mười lớp khác mất tổng cộng 67 — RANK, CALLSIGN, VESSEL_ID tụt hẳn về
0 span phát ra. Vấn đề là TỈ LỆ chứ không phải chất lượng, nên cách sửa là
cắt bớt phần dư của lớp lấn át, giữ nguyên mọi câu đóng góp cho lớp khác.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="tập gốc, giữ nguyên toàn bộ")
    ap.add_argument("--full", required=True, help="tập đã đổ thêm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", action="append", default=[],
                    help="KIỂU=số span tối đa, lặp lại được")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    caps = {}
    for item in args.cap:
        name, _, value = item.partition("=")
        caps[name] = int(value)

    base, full = load(args.base), load(args.full)
    seen = {r["spoken"] for r in base}
    delta = [r for r in full if r["spoken"] not in seen]
    print(f"gốc {len(base)} câu · đầy đủ {len(full)} câu · phần thêm {len(delta)} câu")

    count = collections.Counter()
    for r in base:
        count.update(r.get("types", []))

    random.Random(args.seed).shuffle(delta)
    kept, dropped = [], 0
    for r in delta:
        types = r.get("types", [])
        # Bỏ câu CHỈ khi mọi lớp nó đóng góp đều đã chạm trần. Câu vừa có lớp
        # lấn át vừa có lớp đang đói thì vẫn giữ — cắt nó là cắt cả phần cần.
        if types and all(t in caps and count[t] >= caps[t] for t in types):
            dropped += 1
            continue
        kept.append(r)
        count.update(types)

    rows = base + kept
    random.Random(args.seed).shuffle(rows)
    out = ROOT / args.out
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"giữ {len(kept)}/{len(delta)} câu thêm (bỏ {dropped}) -> {len(rows)} câu")

    print(f"\n{'kiểu':16}{'gốc':>8}{'đầy đủ':>9}{'cân lại':>9}")
    ref = collections.Counter()
    for r in full:
        ref.update(r.get("types", []))
    was = collections.Counter()
    for r in base:
        was.update(r.get("types", []))
    for t in sorted(set(ref), key=lambda x: -ref[x])[:12]:
        print(f"{t:16}{was[t]:>8}{ref[t]:>9}{count[t]:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
