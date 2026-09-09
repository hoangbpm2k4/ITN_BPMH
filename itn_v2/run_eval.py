"""Đánh giá một checkpoint trên nhiều tập dev, có bóc tách theo lớp."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ITNv2Dataset, collate
from .evaluate import boundary_strict_f1, punctuation_f1, type_strict_f1
from .labels import ID2BOUNDARY, ID2TYPE, decode_spans
from .model import ITNv2Model
from .pipeline import normalize_utterance

ROOT = Path(__file__).resolve().parents[1]

# Dấu câu ĐÃ DỰ ĐOÁN phải được đưa vào bộ kết xuất, nếu không văn bản ra
# luôn trắng dấu câu và utterance_exact vĩnh viễn bằng 0.
PUNCT_LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]


def run(model, ds, device, config, show=0):
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
    gold_b, pred_b, gold_t, pred_t = [], [], [], []
    gold_p, pred_p = [], []
    exact = wrong = 0
    src_exact = src_total = 0
    per_type_hit = Counter()
    per_type_total = Counter()
    examples = []

    with torch.no_grad():
        for batch in loader:
            results, out = model.predict(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["pooling_matrix"].to(device), batch["word_mask"].to(device))
            for i, item in enumerate(batch["batch"]):
                n = len(item["boundary"])
                gb = [ID2BOUNDARY[t] for t in item["boundary"]]
                pb = results[i]["boundaries"][:n]
                gold_b.append(gb); pred_b.append(pb)
                gt = [(s, e, ID2TYPE[item["types"][s]]) for s, e in decode_spans(gb)]
                pt = list(zip([s for s, _ in results[i]["spans"]],
                              [e for _, e in results[i]["spans"]], results[i]["types"]))
                gold_t.append(gt); pred_t.append(pt)
                gold_p.append(item["punct"])
                pred_p.append(out["punct_logits"][i][:n].argmax(-1).tolist())

                gold_set = set(gt)
                for span in gt:
                    per_type_total[span[2]] += 1
                    if span in set(pt):
                        per_type_hit[span[2]] += 1

                punct_pred = [PUNCT_LABELS[k]
                              for k in out["punct_logits"][i][:n].argmax(-1).tolist()]
                text, _ = normalize_utterance(
                    item["alignment"].model_words, pb, results[i]["types"],
                    punct_labels=punct_pred,
                    boundary_confidences=results[i]["boundary_confidences"],
                    type_confidences=results[i]["type_confidences"],
                    config=config, alignment=item["alignment"])
                src = item["sample"].written_source
                if src:
                    src_total += 1
                    src_exact += int(text.rstrip(".") == src.rstrip("."))
                if text == item["sample"].written:
                    exact += 1
                else:
                    wrong += 1
                    if len(examples) < show:
                        examples.append((item["sample"].spoken,
                                         item["sample"].written, text))

    b = boundary_strict_f1(gold_b, pred_b)
    t = type_strict_f1(gold_t, pred_t)
    p = punctuation_f1(gold_p, pred_p)
    punct_acc = sum(int(a == c) for g, q in zip(gold_p, pred_p) for a, c in zip(g, q))
    punct_n = sum(len(g) for g in gold_p)
    return {
        "n": len(ds), "boundary_f1": round(b["f1"], 4), "type_f1": round(t["f1"], 4),
        "punct_acc_thô": round(punct_acc / max(1, punct_n), 4),
        "punct_macroF1_bỏ_O": round(p["macro_f1"], 4),
        "punct_theo_lớp": {["O", "COMMA", "PERIOD", "QUESTION"][k]:
                           f"P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}"
                           for k, v in p["per_class"].items()},
        "utterance_exact": round(exact / max(1, exact + wrong), 4),
        # So với VĂN BẢN GỐC của bài báo, không phải đầu ra pipeline. Đây mới là
        # khoảng cách thật; nó thấp chủ yếu vì taxonomy chưa phủ tên riêng, địa
        # danh và viết hoa của văn bản thật.
        "khớp_văn_bản_gốc": (round(src_exact / src_total, 4) if src_total else None),
        "n_có_văn_bản_gốc": src_total,
        "per_type": {k: f"{per_type_hit[k]}/{per_type_total[k]}"
                     for k in sorted(per_type_total)},
        "examples": examples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_v2/best.pt")
    parser.add_argument("--splits", nargs="+",
                        default=["random_dev", "entity_holdout_dev", "template_holdout_dev"])
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="ép mọi lớp về một ngưỡng — CHỈ để chẩn đoán, "
                             "không dùng khi đo thật (spec §9.4 cấm ngưỡng chung)")
    args = parser.parse_args(argv)

    config = Config()
    if args.threshold is not None:
        config.thresholds = {k: args.threshold for k in config.thresholds}
        print(f"[CHẨN ĐOÁN] ép mọi ngưỡng về {args.threshold} — con số dưới đây "
              f"KHÔNG phải kết quả đo thật")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ITNv2Model(config).to(device)
    state = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    for name in args.splits:
        path = ROOT / "datasets_v2" / f"{name}.jsonl"
        if not path.exists() or not path.stat().st_size:
            print(f"{name}: TRỐNG"); continue
        ds = ITNv2Dataset(path, config)
        res = run(model, ds, device, config, show=args.show)
        examples = res.pop("examples")
        print(f"\n=== {name} ===")
        per_class = res.pop("punct_theo_lớp", {})
        print(json.dumps({k: v for k, v in res.items() if k != "per_type"},
                         ensure_ascii=False))
        print("  dấu câu theo lớp:", json.dumps(per_class, ensure_ascii=False))
        print("  đúng span theo lớp:", json.dumps(res["per_type"], ensure_ascii=False))
        for spoken, gold, pred in examples:
            print(f"\n  nói : {spoken}")
            print(f"  cần : {gold}")
            print(f"  ra  : {pred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
