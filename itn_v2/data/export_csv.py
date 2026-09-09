"""Xuất corpus ra CSV để đối chiếu 1:1 bằng mắt.

Mở bằng Excel / LibreOffice: dùng UTF-8 BOM để không bị vỡ tiếng Việt.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from .schema import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def export_pairs(src, dst):
    """Corpus 1:1 thô: dạng nói <-> dạng viết."""
    rows = [json.loads(l) for l in (ROOT / src).open(encoding="utf-8") if l.strip()]
    with (ROOT / dst).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "chu_de", "nguon", "dang_noi", "dang_viet",
                         "so_tu", "so_phay", "so_cham"])
        for r in rows:
            writer.writerow([r["id"], r["topic"], r["kind"], r["spoken"], r["written"],
                             len(r["written"].split()),
                             r["written"].count(","), r["written"].count(".")])
    return len(rows)


def export_labeled(src, dst):
    """Bản đã gán nhãn: nói -> pipeline dựng lại -> văn bản gốc, kèm span."""
    samples = list(read_jsonl(ROOT / src))
    with (ROOT / dst).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "nguon", "dang_noi", "pipeline_dung_lai",
                         "van_ban_goc", "khop_goc", "so_span", "cac_kieu",
                         "chi_tiet_span"])
        for s in samples:
            detail = "; ".join(
                f"{' '.join(s.words[a:b + 1])} -> {t}" for a, b, t in s.spans)
            src_text = s.written_source or ""
            match = ""
            if src_text:
                match = "ĐÚNG" if s.written.rstrip(".") == src_text.rstrip(".") else "khác"
            writer.writerow([s.id, s.source, s.spoken, s.written, src_text, match,
                             len(s.spans), " ".join(s.types), detail])
    return len(samples)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="datasets_v2/v1_pairs.jsonl")
    # ``v1_labeled.jsonl`` là pipeline diff cũ. Tập train hiện dùng bản
    # căn hàng end-to-end, nên CSV mặc định phải soi đúng artifact đó.
    parser.add_argument("--labeled", default="datasets_v2/v1_aligned.jsonl")
    parser.add_argument("--out-dir", default="datasets_v2/csv")
    args = parser.parse_args(argv)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n1 = export_pairs(args.pairs, f"{args.out_dir}/v1_pairs.csv")
    print(f"{n1:5d} dòng -> {args.out_dir}/v1_pairs.csv   (nói ⇄ viết, 1:1)")
    n2 = export_labeled(args.labeled, f"{args.out_dir}/v1_labeled.csv")
    print(f"{n2:5d} dòng -> {args.out_dir}/v1_labeled.csv (kèm span và nhãn)")

    oot = ROOT / "datasets_v2/out_of_taxonomy.jsonl"
    if oot.exists():
        rows = [json.loads(l) for l in oot.open(encoding="utf-8") if l.strip()]
        path = out_dir / "out_of_taxonomy.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["dang_viet", "dang_noi", "ngu_canh"])
            for r in rows:
                writer.writerow([r.get("written", ""), r.get("spoken", ""),
                                 r.get("context", "")])
        print(f"{len(rows):5d} dòng -> {args.out_dir}/out_of_taxonomy.csv "
              f"(tên tổ chức, ngoài 46 kiểu)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
