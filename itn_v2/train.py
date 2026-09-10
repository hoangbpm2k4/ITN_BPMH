"""Vòng huấn luyện V2 (spec §24, §25, §27).

Learning rate phân tầng (encoder chậm, các đầu ra nhanh), warmup tuyến tính,
chọn checkpoint bằng điểm tổng hợp của spec §27 trên tập dev — KHÔNG chọn bằng
loss huấn luyện.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ITNv2Dataset, collate
from .evaluate import (boundary_strict_f1, checkpoint_score,
                       critical_category_accuracy, exact_normalization_accuracy,
                       invalid_transition_rate, type_strict_f1)
from .labels import ID2BOUNDARY, ID2TYPE, decode_spans
from .model import ITNv2Model
from .pipeline import normalize_utterance

ROOT = Path(__file__).resolve().parents[1]

# Dấu câu ĐÃ DỰ ĐOÁN phải được đưa vào bộ kết xuất, nếu không văn bản ra
# luôn trắng dấu câu và utterance_exact vĩnh viễn bằng 0.
PUNCT_LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]


def linear_warmup_decay(step, total, warmup_ratio):
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return step / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    gold_b, pred_b, gold_t, pred_t, records = [], [], [], [], []
    punct_correct = punct_total = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pool = batch["pooling_matrix"].to(device)
        wmask = batch["word_mask"].to(device)
        results, out = model.predict(ids, attn, pool, wmask)

        for i, item in enumerate(batch["batch"]):
            n = len(item["boundary"])
            gold_b.append([ID2BOUNDARY[t] for t in item["boundary"]])
            pred_b.append(results[i]["boundaries"][:n])
            gold_t.append([(s, e, ID2TYPE[item["types"][s]])
                           for s, e in decode_spans(gold_b[-1])])
            pred_t.append([(s, e, t) for (s, e), t in
                           zip(results[i]["spans"], results[i]["types"])])

            punct_pred = [PUNCT_LABELS[k]
                          for k in out["punct_logits"][i][:n].argmax(-1).tolist()]
            text, outputs = normalize_utterance(
                item["alignment"].model_words, results[i]["boundaries"][:n],
                results[i]["types"], punct_labels=punct_pred,
                boundary_confidences=results[i]["boundary_confidences"],
                type_confidences=results[i]["type_confidences"],
                config=config, alignment=item["alignment"])
            records.append({"type": "UTTERANCE", "gold": item["sample"].written,
                            "pred": text, "raw": item["sample"].spoken})

            punct_pred = out["punct_logits"][i][:n].argmax(-1).tolist()
            for p, g in zip(punct_pred, item["punct"]):
                punct_total += 1
                punct_correct += int(p == g)

    b = boundary_strict_f1(gold_b, pred_b)
    t = type_strict_f1(gold_t, pred_t)
    exact = exact_normalization_accuracy(records)
    punct_acc = punct_correct / punct_total if punct_total else 0.0
    score = checkpoint_score(exact["overall"], critical_category_accuracy(records),
                             b["f1"], t["f1"], punct_acc)
    model.train()
    return {
        "boundary_f1": b["f1"], "type_f1": t["f1"],
        "utterance_exact": exact["overall"], "punct_acc": punct_acc,
        "invalid_transition_rate": invalid_transition_rate(pred_b),
        "checkpoint_score": score,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets_v2/train.jsonl")
    parser.add_argument("--dev", default="datasets_v2/random_dev.jsonl")
    parser.add_argument("--out-dir", default="checkpoints_v2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8, help="batch hiệu dụng = batch*accum")
    parser.add_argument("--smoke", action="store_true", help="chạy thử vài bước")
    parser.add_argument("--device", default=None)
    parser.add_argument("--punct-weight", type=float, default=None,
                        help="ghi đè trọng số nhánh dấu câu (spec §8 đặt 0.30)")
    parser.add_argument("--out-name", default="best.pt")
    parser.add_argument("--encoder", default=None,
                        help="đường dẫn backbone khác PhoBERT (vd xlm-roberta-base)")
    parser.add_argument("--no-segment", action="store_true",
                        help="tắt tách từ ghép — bắt buộc với backbone không phải PhoBERT")
    args = parser.parse_args(argv)

    config = Config()
    if args.punct_weight is not None:
        config.w_punct = args.punct_weight
    if args.encoder:
        config.phobert_path = args.encoder
        print(f"[BACKBONE] {args.encoder}")
    if args.no_segment:
        config.segment_words = False
        print("[BACKBONE] tắt tách từ ghép")
    epochs = args.epochs or config.epochs
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = ITNv2Dataset(ROOT / args.train, config)
    dev_ds = ITNv2Dataset(ROOT / args.dev, config)
    print(f"train {len(train_ds)} câu (bỏ {len(train_ds.skipped)}) · "
          f"dev {len(dev_ds)} câu (bỏ {len(dev_ds.skipped)})")
    if not len(train_ds):
        print("không có dữ liệu huấn luyện"); return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate)

    model = ITNv2Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(),
                                  weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: linear_warmup_decay(s, total_steps, config.warmup_ratio))

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    step = 0

    for epoch in range(1, epochs + 1):
        running = 0.0
        optimizer.zero_grad()
        for k, batch in enumerate(train_loader):
            out = model(batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["pooling_matrix"].to(device))
            losses = model.compute_loss(out, batch["boundary_tags"].to(device),
                                        batch["type_tags"].to(device),
                                        batch["punct_tags"].to(device),
                                        batch["word_mask"].to(device))
            (losses["loss"] / args.accum).backward()
            running += float(losses["loss"])
            if (k + 1) % args.accum == 0 or k + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            if args.smoke and k >= 3:
                break

        metrics = evaluate(model, dev_loader, device, config)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(1, k + 1)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        if metrics["checkpoint_score"] > best:
            best = metrics["checkpoint_score"]
            torch.save({"model": model.state_dict(), "metrics": metrics},
                       out_dir / args.out_name)
        if args.smoke:
            break

    print(f"điểm checkpoint tốt nhất: {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
