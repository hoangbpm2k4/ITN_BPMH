"""Chọn ngưỡng tin cậy CHO TỪNG LỚP bằng số liệu, không phải bằng cảm giác.

Spec §9.4 cấm dùng một ngưỡng chung, nhưng bộ ngưỡng hiện tại (0,60/0,75/0,90)
cũng chỉ là phỏng đoán ban đầu. Đo trên bản gốc thật cho thấy hạ đều mọi ngưỡng
là hoà: mở thêm 48 span đúng nhưng cũng thả ra 46 span sai. Tách theo lớp thì
khác hẳn — TIME mở ra 12 đúng 0 sai, còn LOCATION_NAME mở ra 2 đúng 10 sai.

Cách làm: chạy với mọi ngưỡng = 0 để mô hình phát ra TẤT CẢ, chấm từng span
bằng cách hỏi chuỗi đã chuẩn hoá có nằm trong bản gốc không, rồi với mỗi lớp
quét ngưỡng để tối đa hoá (số đúng - số sai).

KHÔNG chọn ngưỡng trên `data_v1/data`: đó là bộ TEST CUỐI. Chọn ngưỡng trên nó
là dùng đáp án để chấm bài — điểm sẽ đẹp lên đúng phần bị tối ưu và không
mang sang dữ liệu mới. Mặc định ở đây là dev tổng hợp.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ITNv2Dataset, collate
from .data.schema import Sample, write_jsonl
from .eval_dv1 import STRIP, canon
from .model import ITNv2Model
from .pipeline import normalize_utterance

ROOT = Path(__file__).resolve().parents[1]
PUNCT_LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]


def collect(checkpoint, eval_set, device):
    """[(kiểu, tin cậy, đúng?)] cho mọi span mô hình phát ra khi bỏ hết ngưỡng."""
    rows = [json.loads(l) for l in (ROOT / eval_set).open(encoding="utf-8")]
    tmp = ROOT / "datasets_v2" / "_tune_tmp.jsonl"
    samples = []
    for i, r in enumerate(rows):
        words = [w.strip(STRIP).lower() for w in r["spoken"].split()]
        words = [w for w in words if w]
        if not words or any(c.isdigit() for c in " ".join(words)):
            continue
        samples.append(Sample(
            id=f"t{i:05d}", spoken=" ".join(words), written=r["written"],
            words=words, spans=[], punct=["O"] * len(words), types=[],
            leak_key="", source=r.get("topic") or r.get("source", "?"),
            written_source=r["written"]))
    write_jsonl(tmp, samples)

    config = Config()
    config.thresholds = {k: 0.0 for k in config.thresholds}
    model = ITNv2Model(config).to(device)
    state = torch.load(ROOT / checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    observations = []
    ds = ITNv2Dataset(tmp, config)
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=8, collate_fn=collate):
            preds, out = model.predict(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["pooling_matrix"].to(device), batch["word_mask"].to(device))
            for i, item in enumerate(batch["batch"]):
                n = len(item["boundary"])
                punct = [PUNCT_LABELS[k]
                         for k in out["punct_logits"][i][:n].argmax(-1).tolist()]
                _, outputs = normalize_utterance(
                    item["alignment"].model_words, preds[i]["boundaries"][:n],
                    preds[i]["types"], punct_labels=punct,
                    boundary_confidences=preds[i]["boundary_confidences"],
                    type_confidences=preds[i]["type_confidences"],
                    config=config, alignment=item["alignment"])
                gold = item["sample"].written_source
                for o in outputs:
                    if not o.normalized:
                        continue
                    observations.append(
                        (o.predicted_type, float(o.model_confidence),
                         o.normalized in gold))
    tmp.unlink(missing_ok=True)
    return observations


def best_threshold(points, floor, ceiling):
    """Ngưỡng tối đa hoá (đúng - sai). Hoà thì chọn ngưỡng CAO hơn.

    Ưu tiên ngưỡng cao khi hoà vì chuẩn hoá sai nguy hiểm hơn giữ nguyên dạng
    nói: người đọc thấy chữ thì biết là chưa xử lý, thấy số sai thì không.
    """
    candidates = sorted({round(c, 2) for _, c, _ in points} | {floor, ceiling})
    best, best_gain = ceiling, None
    for thr in candidates:
        if not floor <= thr <= ceiling:
            continue
        gain = sum(1 if ok else -1 for _, c, ok in points if c >= thr)
        if best_gain is None or gain > best_gain or (gain == best_gain and thr > best):
            best, best_gain = thr, gain
    return best, best_gain


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-set", default="datasets_v2/mil_v51_dev.jsonl",
                        help="KHÔNG trỏ vào datasets_v2/dv1_eval.jsonl — đó là test cuối")
    parser.add_argument("--out", default="itn_v2/thresholds_tuned.json")
    parser.add_argument("--min", type=float, default=0.20)
    parser.add_argument("--max", type=float, default=0.95)
    parser.add_argument("--min-support", type=int, default=3,
                        help="dưới mức này thì giữ ngưỡng mặc định")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if "dv1_eval" in args.eval_set:
        raise SystemExit(
            "từ chối: dv1_eval.jsonl là TEST CUỐI, chọn ngưỡng trên nó là "
            "tối ưu vào đáp án. Dùng dev tổng hợp (mil_v51_dev.jsonl).")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    obs = collect(args.checkpoint, args.eval_set, device)
    by_type = defaultdict(list)
    for t, c, ok in obs:
        by_type[t].append((t, c, ok))

    base = Config().thresholds
    tuned, report = dict(base), []
    for t, points in sorted(by_type.items(), key=lambda x: -len(x[1])):
        n_ok = sum(1 for _, _, ok in points if ok)
        if len(points) < args.min_support:
            report.append((t, len(points), n_ok, base[t], base[t], "ít mẫu, giữ nguyên"))
            continue
        thr, gain = best_threshold(points, args.min, args.max)
        tuned[t] = thr
        note = "" if thr != base[t] else "không đổi"
        report.append((t, len(points), n_ok, base[t], thr, note))

    print(f"{len(obs)} span quan sát trên {len(by_type)} kiểu\n")
    print(f"{'KIỂU':16}{'span':>6}{'đúng':>6}{'cũ':>7}{'mới':>7}  ghi chú")
    for t, n, k, old, new, note in report:
        mark = "" if old == new else ("  ↓" if new < old else "  ↑")
        print(f"  {t:14}{n:6}{k:6}{old:7.2f}{new:7.2f}{mark} {note}")

    out = ROOT / args.out
    out.write_text(json.dumps(tuned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
