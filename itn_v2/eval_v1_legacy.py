"""Chạy ITN V1 (PhoBERT joint, checkpoint v11 epoch14) trên đúng bộ test của V2.

Dùng LẠI nguyên thước đo của itn_v2/eval_dv1.py — canon() + token_f1() + khớp
tuyệt đối — để con số đặt cạnh nhau được. Đầu vào cũng tiền xử lý y hệt: hạ chữ
thường, bỏ dấu câu hai đầu mỗi từ, loại câu còn chữ số.
"""
import csv, json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path("/home/hoangbpm/inverse_text_norm")
sys.path.insert(0, str(ROOT))
from config import Config
from train import ITNLightningModule
from post_process import ITNPostProcessor

STRIP = " \t.,;:!?\"'()[]"

def canon(text):
    return re.sub(r"\s+", " ", text).strip().strip(STRIP)

def token_f1(gold, pred):
    g, p = canon(gold).split(), canon(pred).split()
    if not g or not p:
        return 0.0
    hit = sum((Counter(g) & Counter(p)).values())
    prec, rec = hit / len(p), hit / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

CKPT = ROOT / "checkpoints/v11_joint_fullattn-epoch=14-val_punc_f1=0.8440-val_itn_f1=0.9794.ckpt"

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    pl_model = ITNLightningModule.load_from_checkpoint(
        CKPT, config=config, tokenizer=tokenizer, map_location=device, weights_only=False)
    model = pl_model.model.eval().to(device)
    processor = ITNPostProcessor()

    rows = [json.loads(l) for l in (ROOT / "datasets_v2/dv1_eval.jsonl").open(encoding="utf-8")]
    print(f"{len(rows)} câu đánh giá")

    items = []
    for r in rows:
        words = [w.strip(STRIP).lower() for w in r["spoken"].split()]
        words = [w for w in words if w]
        if not words or any(c.isdigit() for c in " ".join(words)):
            continue
        items.append((r["topic"], words, r["written"]))
    print(f"{len(items)} câu dạng nói hợp lệ (bỏ {len(rows)-len(items)} câu còn chữ số)")

    out_rows, per_topic = [], defaultdict(lambda: [0, 0, 0.0])
    for topic, words, gold in items:
        input_ids = [tokenizer.cls_token_id]
        position_ids = [0]
        for i, word in enumerate(words):
            for s_id in tokenizer.encode(word, add_special_tokens=False):
                input_ids.append(s_id)
                position_ids.append(i + 1)
        input_ids.append(tokenizer.sep_token_id)
        position_ids.append(len(words) + 1)
        if len(input_ids) > config.max_len:
            input_ids = input_ids[:config.max_len - 1] + [tokenizer.sep_token_id]
            position_ids = position_ids[:config.max_len]
        ids = torch.tensor([input_ids], device=device)
        pos = torch.tensor([position_ids], device=device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            _, nc_logits, punc_logits = model(ids, mask, position_ids=pos)
        nc = nc_logits.argmax(-1)[0].tolist()
        pc = punc_logits.argmax(-1)[0].tolist()
        # Gộp theo position_id chứ không theo hậu tố subword: tokenizer PhoBERT ở
        # đây đánh dấu mảnh giữa bằng '@@', trong khi post_process.decode chờ
        # '</w>'. Ta biết chắc ranh giới từ nên gộp thẳng: nhãn nội dung lấy ở
        # mảnh ĐẦU của từ (đúng như get_word_level_labels), nhãn dấu câu lấy ở
        # mảnh CUỐI. Rồi đưa vào decode dưới dạng một mục = một từ.
        pos_list = position_ids
        rt, rl, rp = [], [], []
        last = None
        for i, po in enumerate(pos_list):
            if po == 0 or po > len(words):
                continue
            if po != last:
                rt.append(words[po - 1] + "</w>")
                rl.append(config.nc_id2label[nc[i]])
                rp.append(config.punc_id2label[pc[i]])
                last = po
            else:
                rp[-1] = config.punc_id2label[pc[i]]
        text = processor.decode(rt, rl, rp)
        exact = canon(text) == canon(gold)
        f1 = token_f1(gold, text)
        per_topic[topic][0] += 1
        per_topic[topic][1] += int(exact)
        per_topic[topic][2] += f1
        # cụm nhãn khác O = span V1 phát ra
        spans, i = [], 0
        words_only = " ".join(words).split()
        while i < len(rl):
            if rl[i] == "O":
                i += 1; continue
            j = i
            while j < len(rl) and rl[j] == rl[i]:
                j += 1
            spans.append(rl[i]); i = j
        out_rows.append({
            "chu_de": topic, "dau_vao": " ".join(words), "dau_ra": text,
            "ban_goc": gold, "khop": "ĐÚNG" if exact else "khác",
            "f1_tu": round(f1, 3), "so_span": len(spans),
            "kieu_span": "; ".join(spans),
        })

    out_path = ROOT / "datasets_v2/csv/test_v1_cu.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)

    n = len(out_rows)
    exact = sum(1 for r in out_rows if r["khop"] == "ĐÚNG")
    f1 = sum(r["f1_tu"] for r in out_rows) / n
    print("\n=== KẾT QUẢ V1 (ĐẦU-CUỐI) ===")
    print(f"  câu khớp hoàn toàn : {exact}/{n} = {exact/n:.1%}")
    print(f"  F1 theo từ (TB)    : {f1:.3f}")
    print(f"  span phát ra       : {sum(r['so_span'] for r in out_rows)}")
    kinds = Counter(k for r in out_rows for k in r["kieu_span"].split("; ") if k)
    print("  theo kiểu: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    print(f"\n{'CHỦ ĐỀ':40}{'n':>4}{'khớp':>6}{'F1':>7}")
    for t, (cnt, ok, fs) in sorted(per_topic.items(), key=lambda x: -x[1][2] / max(x[1][0], 1)):
        print(f"  {t:38}{cnt:4}{ok:6}{fs/cnt:7.3f}")
    print(f"\nCSV -> {out_path}")

main()
