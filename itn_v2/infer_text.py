"""Chạy mô hình trên văn bản nói thô và xuất văn bản viết cuối cùng.

Dùng để soi bằng mắt: đầu vào là câu như ASR trả về (chữ thường, không dấu
câu, số đọc thành chữ), đầu ra là câu đã chuẩn hoá đầy đủ.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ITNv2Dataset, collate
from .model import ITNv2Model
from .pipeline import normalize_utterance
from .data.schema import Sample, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
PUNCT_LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
MARKUP = re.compile(r"\[\[(.+?)::[^\]]+\]\]")
STRIP = " \t.,;:!?\"'()[]"


def clean_line(line):
    """Bỏ markup [[...::tag]] và số thứ tự đầu dòng, hạ chữ thường."""
    text = MARKUP.sub(r"\1", line)
    text = re.sub(r"^\s*\d+\s*[.)]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def read_inputs(paths, min_words):
    seen, out = set(), []
    for path in paths:
        p = Path(path)
        raw = p.read_text(encoding="utf-8-sig").splitlines()
        if p.suffix.lower() == ".csv":
            rows = list(csv.reader(raw))
            raw = [r[0] for r in rows if r] [1:]        # bỏ dòng tiêu đề
        for line in raw:
            text = clean_line(line)
            words = [w.strip(STRIP) for w in text.split()]
            words = [w for w in words if w]
            if len(words) < min_words or any(c.isdigit() for c in text):
                continue
            key = " ".join(words)
            if key in seen:
                continue
            seen.add(key)
            out.append((p.name, words))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="file .csv hoặc .txt chứa câu dạng nói")
    parser.add_argument("--checkpoint", default="checkpoints_v2/standardized_e15.pt")
    parser.add_argument("--out", default="datasets_v2/csv/infer_output.csv")
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="ép mọi ngưỡng về một giá trị — chỉ để chẩn đoán")
    args = parser.parse_args(argv)

    rows = read_inputs(args.inputs, args.min_words)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} câu đầu vào")

    tmp = ROOT / "datasets_v2" / "_infer_tmp.jsonl"
    write_jsonl(tmp, [
        Sample(id=f"x{i:05d}", spoken=" ".join(w), written="", words=w,
               spans=[], punct=["O"] * len(w), types=[], leak_key="",
               source=src, written_source=None)
        for i, (src, w) in enumerate(rows)])

    config = Config()
    if args.threshold is not None:
        config.thresholds = {k: args.threshold for k in config.thresholds}
        print(f"[CHẨN ĐOÁN] ép mọi ngưỡng về {args.threshold}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ITNv2Model(config).to(device)
    state = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    ds = ITNv2Dataset(tmp, config)
    print(f"{len(ds)} câu vào được mô hình (bỏ {len(ds.skipped)})")

    results = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=8, collate_fn=collate):
            preds, out = model.predict(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["pooling_matrix"].to(device), batch["word_mask"].to(device))
            for i, item in enumerate(batch["batch"]):
                n = len(item["boundary"])
                punct = [PUNCT_LABELS[k]
                         for k in out["punct_logits"][i][:n].argmax(-1).tolist()]
                text, outputs = normalize_utterance(
                    item["alignment"].model_words, preds[i]["boundaries"][:n],
                    preds[i]["types"], punct_labels=punct,
                    boundary_confidences=preds[i]["boundary_confidences"],
                    type_confidences=preds[i]["type_confidences"],
                    config=config, alignment=item["alignment"])
                spans = "; ".join(
                    f"{o.raw_span} -> {o.normalized} [{o.predicted_type}]"
                    for o in outputs if getattr(o, "emitted", False))
                results.append({
                    "nguon": item["sample"].source,
                    "so_tu": len(item["sample"].words),
                    "dau_vao": item["sample"].spoken,
                    "dau_ra": text,
                    "so_span": sum(1 for o in outputs if getattr(o, "emitted", False)),
                    "chi_tiet_span": spans,
                    "bi_nguong_chan": "; ".join(
                        f"{o.raw_span} -> {o.normalized} [{o.predicted_type}] "
                        f"tin_cay={o.model_confidence:.2f}<{o.threshold}"
                        for o in outputs if not o.emitted and o.normalized),
                })

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    tmp.unlink(missing_ok=True)

    n_changed = sum(1 for r in results if r["dau_ra"].lower().rstrip(".") != r["dau_vao"])
    print(f"\nghi {len(results)} câu -> {args.out}")
    print(f"  {n_changed} câu có thay đổi ({n_changed/max(len(results),1):.0%})")
    print(f"  {sum(r['so_span'] for r in results)} span được phát ra")
    return 0


if __name__ == "__main__":
    sys.exit(main())
