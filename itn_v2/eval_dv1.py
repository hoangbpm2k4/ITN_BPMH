"""Đánh giá ĐẦU-CUỐI trên data_v1/data: nói -> pipeline -> viết, so với bản gốc.

Không dùng nhãn span nào của ta, nên đây là thước đo độc lập nhất hiện có.
Xuất CSV để soi bằng mắt.
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ITNv2Dataset, collate
from .data.schema import Sample, write_jsonl
from .model import ITNv2Model
from .pipeline import normalize_utterance

ROOT = Path(__file__).resolve().parents[1]
PUNCT_LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
STRIP = " \t.,;:!?\"'()[]"


def canon(text):
    """Chuẩn hoá nhẹ để so: gộp khoảng trắng, bỏ dấu câu ở hai đầu."""
    return re.sub(r"\s+", " ", text).strip().strip(STRIP)


def token_f1(gold, pred):
    g, p = canon(gold).split(), canon(pred).split()
    if not g or not p:
        return 0.0
    common = Counter(g) & Counter(p)
    hit = sum(common.values())
    prec, rec = hit / len(p), hit / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_v2/best.pt")
    parser.add_argument("--eval-set", default="datasets_v2/dv1_eval.jsonl")
    parser.add_argument("--out", default="datasets_v2/csv/dv1_eval_output.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    rows = [json.loads(l) for l in (ROOT / args.eval_set).open(encoding="utf-8")]
    print(f"{len(rows)} câu đánh giá")

    tmp = ROOT / "datasets_v2" / "_dv1_tmp.jsonl"
    samples = []
    for i, r in enumerate(rows):
        words = [w.strip(STRIP).lower() for w in r["spoken"].split()]
        words = [w for w in words if w]
        if not words or any(c.isdigit() for c in " ".join(words)):
            continue
        samples.append(Sample(
            id=f"e{i:05d}", spoken=" ".join(words), written=r["written"],
            words=words, spans=[], punct=["O"] * len(words), types=[],
            leak_key="", source=r["topic"], written_source=r["written"]))
    write_jsonl(tmp, samples)
    print(f"{len(samples)} câu dạng nói hợp lệ (bỏ {len(rows)-len(samples)} câu còn chữ số)")

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
    out_rows, per_topic = [], defaultdict(lambda: [0, 0, 0.0])
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
                gold = item["sample"].written_source
                exact = canon(text) == canon(gold)
                f1 = token_f1(gold, text)
                topic = item["sample"].source
                per_topic[topic][0] += 1
                per_topic[topic][1] += int(exact)
                per_topic[topic][2] += f1
                emitted = [o for o in outputs if getattr(o, "emitted", False)]
                out_rows.append({
                    "chu_de": topic,
                    "dau_vao": item["sample"].spoken,
                    "dau_ra": text,
                    "ban_goc": gold,
                    "khop": "ĐÚNG" if exact else "khác",
                    "f1_tu": round(f1, 3),
                    "so_span": len(emitted),
                    "chi_tiet_span": "; ".join(
                        f"{o.raw_span} -> {o.normalized} [{o.predicted_type}]"
                        for o in emitted),
                    "bi_nguong_chan": "; ".join(
                        f"{o.raw_span} -> {o.normalized} [{o.predicted_type}] "
                        f"{o.model_confidence:.2f}<{o.threshold}"
                        for o in outputs if not o.emitted and o.normalized),
                })

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    tmp.unlink(missing_ok=True)

    n = len(out_rows)
    exact = sum(1 for r in out_rows if r["khop"] == "ĐÚNG")
    f1 = sum(r["f1_tu"] for r in out_rows) / n
    spans = sum(r["so_span"] for r in out_rows)
    blocked = sum(1 for r in out_rows if r["bi_nguong_chan"])
    print(f"\n=== KẾT QUẢ ĐẦU-CUỐI ===")
    print(f"  câu khớp hoàn toàn : {exact}/{n} = {exact/n:.1%}")
    print(f"  F1 theo từ (TB)    : {f1:.3f}")
    print(f"  span phát ra       : {spans}")
    print(f"  câu có span bị ngưỡng chặn: {blocked}")
    print(f"\n{'CHỦ ĐỀ':40}{'n':>4}{'khớp':>6}{'F1':>7}")
    for t, (cnt, ok, fs) in sorted(per_topic.items(), key=lambda x: -x[1][2] / max(x[1][0], 1)):
        print(f"  {t:38}{cnt:4}{ok:6}{fs/cnt:7.3f}")
    print(f"\nCSV -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
