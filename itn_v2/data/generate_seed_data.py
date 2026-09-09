"""Sinh dữ liệu hạt giống cho ITN V2 bằng Gemma trên Google AI Studio.

Nguyên tắc: mô hình sinh LỎNG, pipeline tất định lọc CHẶT. Một câu chỉ được
giữ khi mọi span trong đó chạy qua được normalizer + validator + cổng ngưỡng.
Nhờ vậy nhãn sai do mô hình bịa sẽ tự bị loại, không lọt vào tập huấn luyện.

Chạy:
    export GOOGLE_API_KEY=...
    python -m itn_v2.data.generate_seed_data --rounds 6 --per-prompt 12
"""

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..inference import normalize_with_gold_spans
from ..labels import SEMANTIC_TYPES
from .gemma_client import GemmaClient
from .markup import harvest_lines, parse_markup
from .prompts import (HARD_NEGATIVE_GROUP, TYPE_GROUPS, build_constrained_prompt,
                      build_prompt, catalog_spoken_forms)
from .schema import Sample, validate_sample, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
SEED_FILES = [
    ROOT / "data_raw" / "military_data_processed.csv",
    ROOT / "data_raw" / "qdnd_raw_text_itn.csv",
    ROOT / "data_raw" / "itn_foreign_5000_final.csv",
]


def load_style_seeds(limit=400):
    """Câu nghiệp vụ THẬT trong kho, dùng làm mẫu văn phong cho prompt."""
    seeds = []
    for path in SEED_FILES:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    text = (row.get("text") or "").strip()
                    if 40 <= len(text) <= 220 and "\n" not in text:
                        seeds.append(re.sub(r"\s+", " ", text))
                    if len(seeds) >= limit:
                        break
        except Exception:
            continue
    return seeds


def make_sample(words, spans, punct, source):
    """Chạy qua pipeline tất định; trả Sample nếu MỌI span đều phát ra được."""
    spoken = " ".join(words)
    annotations = [(s, e, t) for s, e, t in spans]
    try:
        written, outputs, _alignment, conflicts = normalize_with_gold_spans(
            spoken, annotations, punct)
    except Exception as exc:
        return None, f"pipeline lỗi: {type(exc).__name__}"
    if conflicts:
        return None, "từ ghép cắt ngang span"
    for out in outputs:
        if not out.emitted:
            return None, f"{out.predicted_type}: {out.reason[:60]}"

    types = sorted({t for _, _, t in spans})
    leak_key = "|".join(sorted(o.normalized for o in outputs))
    sample = Sample(
        id=hashlib.sha1(spoken.encode()).hexdigest()[:16],
        spoken=spoken, written=written, words=words,
        spans=annotations, punct=punct, types=types,
        leak_key=leak_key, source=source,
    )
    errors = validate_sample(sample)
    if errors:
        return None, "schema: " + errors[0]
    return sample, None


def build_closed_class_pairs():
    """Cách đọc hợp lệ cho các lớp đóng, lấy thẳng từ danh mục ngoài."""
    pairs = []
    for spoken, canonical in catalog_spoken_forms("equipment"):
        pairs.append((spoken, canonical, "EQUIPMENT_ID"))
    for spoken, canonical in catalog_spoken_forms("foreign_names"):
        pairs.append((spoken, canonical, "FOREIGN_NAME"))
        pairs.append((spoken, canonical, "EQUIPMENT_NAME"))
    for spoken, canonical in catalog_spoken_forms("acronyms"):
        pairs.append((spoken, canonical, "ACRONYM"))
    for spoken, canonical in catalog_spoken_forms("maritime_terms"):
        pairs.append((spoken, canonical, "MARITIME_TERM"))
    return pairs


CLOSED_EXAMPLES = [
    "biên đội [[xu ba lăm|EQUIPMENT_ID]] xuất kích lúc rạng sáng .",
    "hệ thống [[pa tri ốt|EQUIPMENT_NAME]] đã chuyển trạng thái sẵn sàng .",
    "tổ công tác liên lạc qua [[vê hát ép|ACRONYM]] với đài bờ .",
    "thuyền trưởng ra lệnh [[đét xờ lâu a hét|MARITIME_TERM]] khi vào luồng .",
]

IMO_EXAMPLES = [
    "tàu hàng số [[i em ô chín không bảy bốn bảy hai chín|IMO_ID]] đã rời cảng .",
    "đăng kiểm [[i em ô chín một hai ba bốn năm sáu|IMO_ID]] còn hiệu lực .",
]

IMO_PROMPT = """Viết {n} câu tiếng Việt DẠNG NÓI về hàng hải, chữ thường, TUYỆT ĐỐI
không dùng ký tự số.

Mỗi câu chứa đúng một cụm [[...|IMO_ID]]. Cụm đó phải bắt đầu bằng "i em ô" rồi
ĐÚNG BẢY chữ số đọc rời (không, một, hai, ba, bốn, năm, sáu, bảy, tám, chín).
Đếm cho đủ bảy chữ số, không thừa không thiếu.

Dấu câu là token riêng: , . ?

VÍ DỤ:
{examples}

In ra {n} dòng câu."""


def run_round(client, group_name, group, seeds, per_prompt, hard_negative=False):
    if group_name == "lop_dong":
        pairs = random.sample(CLOSED_CLASS_PAIRS, min(18, len(CLOSED_CLASS_PAIRS)))
        prompt = build_constrained_prompt(pairs, CLOSED_EXAMPLES, n=per_prompt)
    elif group_name == "imo":
        prompt = IMO_PROMPT.format(n=per_prompt, examples="\n".join(IMO_EXAMPLES))
    else:
        prompt = build_prompt(group, n=per_prompt,
                              seeds=random.sample(seeds, min(3, len(seeds))) if seeds else None,
                              hard_negative=hard_negative)
    text = client.generate(prompt)
    results, reasons = [], Counter()
    for line in harvest_lines(text):
        parsed = parse_markup(line)
        if parsed is None:
            reasons["không phân tích được đánh dấu"] += 1
            continue
        words, spans, punct = parsed
        if any(t not in SEMANTIC_TYPES for _, _, t in spans):
            reasons["kiểu ngoài taxonomy"] += 1
            continue
        sample, why = make_sample(words, spans, punct, f"gemma:{group_name}")
        if sample is None:
            reasons[why or "không rõ"] += 1
            continue
        results.append(sample)
    return results, reasons


CLOSED_CLASS_PAIRS = []


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4, help="số vòng cho mỗi nhóm kiểu")
    parser.add_argument("--per-prompt", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--out", default="datasets_v2/seed_generated.jsonl")
    args = parser.parse_args(argv)

    global CLOSED_CLASS_PAIRS
    CLOSED_CLASS_PAIRS = build_closed_class_pairs()
    client = GemmaClient(model=args.model)
    seeds = load_style_seeds()
    print(f"nạp {len(seeds)} câu mẫu văn phong từ data_raw")

    jobs = []
    for name, group in TYPE_GROUPS.items():
        for _ in range(args.rounds):
            jobs.append((name, group, False))
    for _ in range(max(2, args.rounds // 2)):
        jobs.append(("doi_chung", HARD_NEGATIVE_GROUP, True))
    # Lớp đóng: nạp sẵn cách đọc hợp lệ thay vì để mô hình sáng tác
    for _ in range(max(3, args.rounds)):
        jobs.append(("lop_dong", None, False))
    for _ in range(max(2, args.rounds // 2)):
        jobs.append(("imo", None, False))
    random.shuffle(jobs)
    print(f"{len(jobs)} lượt gọi, {args.workers} luồng, model {args.model}")

    kept, reasons = {}, Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_round, client, n, g, seeds, args.per_prompt, hn): n
                   for n, g, hn in jobs}
        for future in as_completed(futures):
            done += 1
            try:
                samples, why = future.result()
            except Exception as exc:
                reasons[f"gọi API lỗi: {type(exc).__name__}"] += 1
                continue
            reasons.update(why)
            for s in samples:
                kept.setdefault(s.id, s)
            print(f"  [{done}/{len(jobs)}] {futures[future]:16s} "
                  f"+{len(samples):3d} câu, tổng giữ {len(kept)}", flush=True)

    samples = list(kept.values())
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, samples)

    type_counts = Counter(t for s in samples for t in s.types)
    print(f"\nđã ghi {len(samples)} câu vào {out_path}")
    print(f"gọi API: {client.calls} thành công, {client.retries} lần phải thử lại")
    print("\n--- phủ kiểu ---")
    for t in SEMANTIC_TYPES:
        print(f"  {t:16s} {type_counts.get(t, 0)}")
    missing = [t for t in SEMANTIC_TYPES if not type_counts.get(t)]
    print(f"\nkiểu chưa có câu nào: {missing or 'không'}")
    print("\n--- lý do loại ---")
    for why, n in reasons.most_common(15):
        print(f"  {n:4d}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
