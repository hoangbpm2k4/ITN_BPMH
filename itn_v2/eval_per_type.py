"""Đánh giá theo TỪNG kiểu span: đạt / không đạt / bị lẫn sang kiểu nào.

Tách ba loại hỏng khác nhau, vì cách sửa khác hẳn nhau:
  BIÊN SAI   mô hình không khoanh trúng cụm  -> thiếu dữ liệu hoặc CRF yếu.
  KIỂU LẪN   khoanh trúng nhưng gán nhầm kiểu -> hai lớp giống nhau, cần đặc trưng.
  DỰNG SAI   trúng cả hai nhưng normalizer ra sai chuỗi -> lỗi code chuẩn hoá.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import csv as csvmod

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ITNv2Dataset, collate
from .labels import ID2BOUNDARY, ID2TYPE, SEMANTIC_TYPES, decode_spans
from .model import ITNv2Model

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_v2/standardized_e15.pt")
    parser.add_argument("--splits", nargs="+",
                        default=["random_dev", "entity_holdout_dev", "template_holdout_dev"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--csv", default="datasets_v2/csv/bao_cao_theo_kieu.csv")
    parser.add_argument("--baseline", default=None,
                        help="CSV bản trước, để thêm cột so sánh f1_truoc/thay_doi")
    parser.add_argument("--extra-split", default=None,
                        help="tập dev thứ hai (vd mil_dev), đo riêng để so miền")
    args = parser.parse_args(argv)

    config = Config()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ITNv2Model(config).to(device)
    state = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    gold_n = Counter()          # số span vàng theo kiểu
    hit = Counter()             # trúng cả biên lẫn kiểu
    boundary_only = Counter()   # trúng biên, sai kiểu
    missed = Counter()          # không trúng biên
    pred_n = Counter()          # số span mô hình dự đoán theo kiểu
    confusion = defaultdict(Counter)
    spurious = Counter()        # dự đoán ra span mà vàng không có
    examples = defaultdict(list)
    ok_ex = defaultdict(list)       # ví dụ mô hình làm ĐÚNG
    any_ex = defaultdict(list)      # ví dụ bất kỳ của kiểu, kể cả khi mô hình sai
    wrong_ex = defaultdict(list)    # ví dụ bị lẫn, kèm kiểu đoán nhầm

    for name in args.splits:
        path = ROOT / "datasets_v2" / f"{name}.jsonl"
        if not path.exists() or not path.stat().st_size:
            continue
        ds = ITNv2Dataset(path, config)
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=8, collate_fn=collate):
                preds, _ = model.predict(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["pooling_matrix"].to(device), batch["word_mask"].to(device))
                for i, item in enumerate(batch["batch"]):
                    n = len(item["boundary"])
                    gb = [ID2BOUNDARY[t] for t in item["boundary"]]
                    gold = {(s, e): ID2TYPE[item["types"][s]] for s, e in decode_spans(gb)}
                    pred = {(s, e): t for (s, e), t
                            in zip(preds[i]["spans"], preds[i]["types"])}
                    words = item["alignment"].model_words
                    for (s, e), gt in gold.items():
                        gold_n[gt] += 1
                        chunk = " ".join(words[s:e + 1]).replace("_", " ")
                        if len(any_ex[gt]) < 15:
                            any_ex[gt].append(chunk)
                        if (s, e) not in pred:
                            missed[gt] += 1
                            if len(wrong_ex[gt]) < 2:
                                wrong_ex[gt].append(f"{chunk} (không khoanh ra)")
                            continue
                        pt = pred[(s, e)]
                        if pt == gt:
                            hit[gt] += 1
                            if len(ok_ex[gt]) < 2:
                                ok_ex[gt].append(chunk)
                        else:
                            if len(wrong_ex[gt]) < 2:
                                wrong_ex[gt].append(f"{chunk} -> đoán {pt}")
                            boundary_only[gt] += 1
                            confusion[gt][pt] += 1
                            if len(examples[(gt, pt)]) < 2:
                                examples[(gt, pt)].append(" ".join(words[s:e + 1]))
                    for (s, e), pt in pred.items():
                        pred_n[pt] += 1
                        if (s, e) not in gold:
                            spurious[pt] += 1

    # Đo lại trên tập phụ (nếu có) để tách miền tổng hợp khỏi miền thật.
    extra_gold, extra_hit = Counter(), Counter()
    if args.extra_split:
        path = ROOT / "datasets_v2" / f"{args.extra_split}.jsonl"
        if path.exists() and path.stat().st_size:
            ds = ITNv2Dataset(path, config)
            with torch.no_grad():
                for batch in DataLoader(ds, batch_size=8, collate_fn=collate):
                    preds, _ = model.predict(
                        batch["input_ids"].to(device), batch["attention_mask"].to(device),
                        batch["pooling_matrix"].to(device), batch["word_mask"].to(device))
                    for i, item in enumerate(batch["batch"]):
                        gb = [ID2BOUNDARY[t] for t in item["boundary"]]
                        gold = {(a, b): ID2TYPE[item["types"][a]]
                                for a, b in decode_spans(gb)}
                        pred = {(a, b): t for (a, b), t
                                in zip(preds[i]["spans"], preds[i]["types"])}
                        for k, gt in gold.items():
                            extra_gold[gt] += 1
                            if pred.get(k) == gt:
                                extra_hit[gt] += 1

    print(f"checkpoint: {args.checkpoint}")
    print(f"{sum(gold_n.values())} span vàng · {sum(pred_n.values())} span dự đoán\n")

    header = f"{'KIỂU':16} {'vàng':>5} {'đúng':>5} {'lẫn':>5} {'sót':>5} {'thừa':>5} {'P':>6} {'R':>6} {'F1':>6}"
    print(header); print("-" * len(header))
    rows = []
    for t in sorted(gold_n, key=lambda x: -gold_n[x]):
        if gold_n[t] < args.min_support:
            continue
        h, g, p = hit[t], gold_n[t], pred_n[t]
        prec = h / p if p else 0.0
        rec = h / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((t, g, h, boundary_only[t], missed[t], spurious[t], prec, rec, f1))
        print(f"{t:16} {g:5} {h:5} {boundary_only[t]:5} {missed[t]:5} "
              f"{spurious[t]:5} {prec:6.2f} {rec:6.2f} {f1:6.2f}")

    print("\n=== ĐẠT (F1 >= 0.50) ===")
    ok = [r for r in rows if r[8] >= 0.50]
    print("  " + ", ".join(f"{r[0]}({r[8]:.2f}, n={r[1]})" for r in ok) if ok else "  không có")

    print("\n=== KHÔNG ĐẠT (F1 < 0.50) ===")
    for r in sorted([r for r in rows if r[8] < 0.50], key=lambda x: -x[1]):
        ly = "sót biên" if r[4] >= r[3] else "lẫn kiểu"
        print(f"  {r[0]:16} F1={r[8]:.2f} n={r[1]:3}  hỏng chủ yếu do: {ly}")

    print("\n=== BỊ LẪN (vàng -> mô hình đoán thành) ===")
    pairs = [(gt, pt, c) for gt, cs in confusion.items() for pt, c in cs.items()]
    for gt, pt, c in sorted(pairs, key=lambda x: -x[2])[:20]:
        ex = " | ".join(examples[(gt, pt)])[:56]
        print(f"  {c:4}  {gt:16} -> {pt:16} vd: {ex}")

    # ---- xuất CSV đầy đủ 46 kiểu ----
    train_n = Counter()
    train_ex = defaultdict(list)
    for line in (ROOT / "datasets_v2" / "train.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        for _s, _e, t in rec["spans"]:
            train_n[t] += 1
            if len(train_ex[t]) < 15:
                train_ex[t].append(" ".join(rec["words"][_s:_e + 1]))

    from .registry import get_normalizer

    def viet_hoa(chunk, type_name):
        """Dạng viết mà normalizer sinh ra cho cụm này — để người đọc thấy đích."""
        fn = get_normalizer(type_name)
        if fn is None or not chunk:
            return ""
        try:
            res = fn(chunk, type_name)
        except Exception:
            return ""
        return res.normalized if res.valid and res.normalized else ""

    def la_vi_du_that(chunk, type_name):
        """Ví dụ chỉ dùng được khi chuẩn hoá ĐỔI nội dung.

        CSV này dùng làm mẫu để sinh thêm dữ liệu, nên một dòng như
        `www.vinamarine.gov.vn -> www.vinamarine.gov.vn` là mẫu độc: nó dạy
        người (hoặc LLM) đọc bên NÓI y hệt bên VIẾT, đúng cái lỗi đã làm
        ELECTRONIC chết. Chỉ khác chữ hoa thì vẫn hợp lệ với lớp tên riêng.
        """
        out = viet_hoa(chunk, type_name)
        return bool(out) and out != chunk

    def chon_vi_du(type_name):
        """Ví dụ tốt nhất: ưu tiên dev, rồi train; bỏ mọi ví dụ không đổi gì."""
        for pool in (any_ex[type_name], train_ex[type_name]):
            for chunk in pool:
                if la_vi_du_that(chunk, type_name):
                    return chunk
        # không có ví dụ nào đổi nội dung -> nói thẳng thay vì đưa mẫu sai
        return ""

    out_rows = []
    for t in SEMANTIC_TYPES:
        if t == "O":
            continue
        g, h, p = gold_n[t], hit[t], pred_n[t]
        prec = h / p if p else 0.0
        rec = h / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lan = "; ".join(f"{pt}:{c}" for pt, c in confusion[t].most_common(3))
        if g == 0:
            trang_thai = "KHÔNG CÓ SPAN TRONG DEV"
        elif f1 >= 0.50:
            trang_thai = "ĐẠT"
        elif h > 0:
            trang_thai = "YẾU"
        else:
            trang_thai = "HỎNG (0 span đúng)"
        nguyen_nhan = ("thiếu dữ liệu train" if train_n[t] < 20
                       else ("lẫn kiểu" if boundary_only[t] >= missed[t] else "sót biên"))
        f1_extra = (extra_hit[t] / extra_gold[t]) if extra_gold[t] else None
        out_rows.append({
            "kieu": t,
            "span_train": train_n[t],
            "span_dev": g,
            "dung": h,
            "sai_lan_kieu": boundary_only[t],
            "thieu_sot_bien": missed[t],
            "thua_bia_ra": spurious[t],
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "trang_thai": trang_thai,
            "nguyen_nhan": nguyen_nhan if g else "",
            "bi_lan_sang": lan,
            "vi_du_noi": chon_vi_du(t) or "(chưa có mẫu hợp lệ)",
            "vi_du_viet": viet_hoa(chon_vi_du(t), t) or "(chưa có mẫu hợp lệ)",
            "vi_du_mo_hinh_lam_dung": "; ".join(ok_ex[t]) or "(không có)",
            "vi_du_mo_hinh_lam_sai": "; ".join(wrong_ex[t]) or "(không có)",
            "vi_du_khac": "; ".join(
                f"{c} -> {viet_hoa(c, t)}"
                for c in (any_ex[t] + train_ex[t])
                if la_vi_du_that(c, t) and c != chon_vi_du(t))[:160],
            "f1_dev_tong_hop": "" if f1_extra is None else round(f1_extra, 3),
            "span_dev_tong_hop": extra_gold[t],
        })
    if args.baseline:
        base_path = ROOT / args.baseline
        if base_path.exists():
            with base_path.open(encoding="utf-8-sig") as fh:
                base = {r["kieu"]: r for r in csvmod.DictReader(fh)}
            for r in out_rows:
                b = base.get(r["kieu"])
                if b:
                    old = float(b["f1"])
                    r["f1_truoc"] = old
                    r["thay_doi"] = round(r["f1"] - old, 3)
                    r["span_train_truoc"] = int(b["span_train"])
                else:
                    r["f1_truoc"] = ""
                    r["thay_doi"] = ""
                    r["span_train_truoc"] = ""

    # Ưu tiên sinh thêm dữ liệu: lớp nào còn yếu mà đã có nhiều mẫu train thì
    # vấn đề là RANH GIỚI LỚP, thêm dữ liệu cùng khuôn sẽ không cứu được.
    for r in out_rows:
        n_train, f1 = r["span_train"], r["f1"]
        if r["span_dev"] == 0 and not r.get("span_dev_tong_hop"):
            r["viec_can_lam"] = "chưa có dữ liệu đánh giá — cần thêm câu dev"
        elif f1 >= 0.80:
            r["viec_can_lam"] = "đạt — không cần thêm"
        elif n_train < 200:
            r["viec_can_lam"] = f"THÊM DỮ LIỆU (mới {n_train} mẫu train)"
        elif r["sai_lan_kieu"] > r["thieu_sot_bien"]:
            r["viec_can_lam"] = "lẫn kiểu — cần mẫu PHÂN BIỆT với " + \
                (r["bi_lan_sang"].split(":")[0] if r["bi_lan_sang"] else "lớp gần")
        else:
            r["viec_can_lam"] = "sót biên — cần câu dài/nhiều span hơn"

    out_rows.sort(key=lambda r: (-r["span_dev"], -r["span_train"]))
    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csvmod.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print(f"\n=== đã ghi {len(out_rows)} kiểu -> {args.csv} ===")
    print(f"  tổng span train {sum(train_n.values())} · dev {sum(gold_n.values())} · "
          f"đúng {sum(hit.values())} · lẫn kiểu {sum(boundary_only.values())} · "
          f"sót biên {sum(missed.values())} · thừa {sum(spurious.values())}")

    print("\n=== SPAN THỪA (mô hình bịa ra, vàng không có) ===")
    for t, c in spurious.most_common(12):
        print(f"  {c:4}  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
