"""Chia tập có chống rò rỉ (spec §21, §22).

Sáu tập: train, random_dev, entity_holdout_dev, template_holdout_dev,
real_dev, real_test. Hai tập cuối phải là dữ liệu THẬT do người cung cấp —
script này không tự sinh chúng, chỉ tạo chỗ trống và cảnh báo.

Quy tắc chống rò rỉ: mọi biến thể sinh ra từ cùng một giá trị chuẩn phải nằm
CÙNG một tập. Ví dụ "su ba lăm" và "xu ba mươi lăm" đều cho Su-35 nên không
được chia một câu vào train, một câu vào test.
"""

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .schema import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def template_signature(sample):
    """Khung câu: bỏ nội dung span, chỉ giữ từ ngoài span + vị trí kiểu."""
    marked = list(sample.words)
    for start, end, type_name in sorted(sample.spans, reverse=True):
        marked[start:end + 1] = [f"<{type_name}>"]
    return re.sub(r"\s+", " ", " ".join(marked)).strip()


def build_splits(samples, entity_holdout_ratio=0.12, template_holdout_ratio=0.12,
                 dev_ratio=0.12, seed=13):
    """Chia bốn tập sinh tự động.

    Hai trục được xử lý KHÁC nhau, có chủ đích:

    - Trục thực thể (§21): biến thể của cùng một giá trị chuẩn không bao giờ
      bị chia giữa train / random_dev / entity_holdout_dev. Đây là trục có nguy
      cơ học vẹt.
    - Trục khung câu (§22): template_holdout_dev CỐ Ý dùng lại thực thể đã có
      trong train — chỉ khung câu là mới. Có vậy mới cô lập được đúng một biến.
    """
    rng = random.Random(seed)

    by_entity = defaultdict(list)
    for s in samples:
        by_entity[s.leak_key].append(s)
    entity_keys = sorted(by_entity)
    rng.shuffle(entity_keys)

    budget = max(0, len(entity_keys) - max(1, len(entity_keys) // 2))
    n_entity = min(max(1, int(len(entity_keys) * entity_holdout_ratio)), budget)
    entity_holdout = [s for k in entity_keys[:n_entity] for s in by_entity[k]]
    remaining = [s for k in entity_keys[n_entity:] for s in by_entity[k]]

    by_template = defaultdict(list)
    for s in remaining:
        by_template[template_signature(s)].append(s)
    template_keys = sorted(by_template)
    rng.shuffle(template_keys)
    t_budget = max(0, len(template_keys) - max(1, len(template_keys) // 2))
    n_template = min(max(1, int(len(template_keys) * template_holdout_ratio)), t_budget)
    template_holdout = [s for k in template_keys[:n_template] for s in by_template[k]]
    rest = [s for k in template_keys[n_template:] for s in by_template[k]]

    # train / random_dev chia theo NHÓM thực thể, không chia theo từng câu
    rest_by_entity = defaultdict(list)
    for s in rest:
        rest_by_entity[s.leak_key].append(s)
    rest_keys = sorted(rest_by_entity)
    rng.shuffle(rest_keys)
    n_dev = min(max(1, int(len(rest_keys) * dev_ratio)),
                max(0, len(rest_keys) - 1))
    return {
        "train": [s for k in rest_keys[n_dev:] for s in rest_by_entity[k]],
        "random_dev": [s for k in rest_keys[:n_dev] for s in rest_by_entity[k]],
        "entity_holdout_dev": entity_holdout,
        "template_holdout_dev": template_holdout,
    }


def check_leakage(splits):
    """Hai kiểm tra khác nhau cho hai trục.

    1. Giá trị chuẩn không được xuất hiện ở nhiều tập trong nhóm
       {train, random_dev, entity_holdout_dev}.
    2. Khung câu của template_holdout_dev không được xuất hiện trong train.
    """
    violations = []
    entity_axis = ("train", "random_dev", "entity_holdout_dev")
    seen = defaultdict(set)
    for name in entity_axis:
        for s in splits.get(name, []):
            seen[s.leak_key].add(name)
    violations += [(f"entity:{k}", sorted(v)) for k, v in seen.items() if len(v) > 1]

    train_templates = {template_signature(s) for s in splits.get("train", [])}
    for s in splits.get("template_holdout_dev", []):
        sig = template_signature(s)
        if sig in train_templates:
            violations.append((f"template:{sig}", ["train", "template_holdout_dev"]))
    return violations


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets_v2/seed_generated.jsonl")
    parser.add_argument("--out-dir", default="datasets_v2")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    samples = list(read_jsonl(ROOT / args.input))
    print(f"nạp {len(samples)} câu")
    splits = build_splits(samples, seed=args.seed)

    violations = check_leakage(splits)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", items)
        types = Counter(t for s in items for t in s.types)
        print(f"  {name:22s} {len(items):5d} câu, {len(types):2d} kiểu")

    print(f"\nrò rỉ biến thể giữa các tập: {len(violations)} "
          f"{'(ĐẠT)' if not violations else '(PHẢI SỬA)'}")
    for key, names in violations[:5]:
        print(f"    {key} xuất hiện ở {names}")

    for name in ("real_dev", "real_test"):
        path = out_dir / f"{name}.jsonl"
        if not path.exists():
            path.write_text("", encoding="utf-8")
            print(f"  {name:22s} TRỐNG — phải nạp dữ liệu THẬT, không được sinh")
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
