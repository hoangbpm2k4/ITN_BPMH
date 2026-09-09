"""Gán nhãn ITN V2 từ VĂN BẢN GỐC bằng Gemini.

Khác hẳn `from_data_v1.py`: ở đó ta đi ngược từ bản augment của pipeline v14 —
mà bản đó đã hỏng sẵn ("qua 80" -> "Q U A tám mươi"). Ở đây ta quay về nguồn:
đưa CÂU GỐC dạng viết cho Gemini, yêu cầu nó chỉ ra từng cụm cần chuẩn hoá kèm
kiểu và cách đọc. Văn bản gốc là chân lý, không phải bản augment.

Vẫn giữ nguyên cổng chất lượng: chỉ nhận span nào mà normalizer chạy trên cách
đọc tái tạo ĐÚNG dạng viết trong câu gốc. Gemini gán sai thì bị loại.

Gộp nhiều câu mỗi lượt gọi để không đốt hạn mức.

    export GOOGLE_API_KEY=...
    python -m itn_v2.data.annotate_with_gemini --limit 400
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..inference import normalize_with_gold_spans
from ..labels import SEMANTIC_TYPES
from ..registry import get_normalizer
from ..validators import validate
from .from_data_v1 import PUNCT_MAP, STRIP_CHARS
from .gemma_client import GemmaClient, RateLimiter
from .schema import Sample, validate_sample, write_jsonl

ROOT = Path(__file__).resolve().parents[2]

# Kiểu ngoài taxonomy: KHÔNG đưa vào huấn luyện, chỉ đếm để có bằng chứng cho
# quyết định "có nên mở thêm lớp không". Taxonomy 46 kiểu đang đóng băng (§5).
OUT_OF_TAXONOMY = {"ORGANIZATION"}
OUT_OF_TAXONOMY_EXAMPLES = []

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "written": {"type": "string"},
                                "type": {"type": "string"},
                                "spoken": {"type": "string"},
                            },
                            "required": ["written", "type", "spoken"],
                        },
                    },
                },
                "required": ["index", "spans"],
            },
        }
    },
    "required": ["items"],
}

PROMPT = """Bạn gán nhãn dữ liệu cho hệ chuẩn hoá văn bản tiếng Việt sau nhận dạng tiếng nói.

Với MỖI câu dưới đây, tìm mọi cụm mà khi đọc thành tiếng sẽ khác dạng viết,
hoặc là tên riêng cần viết hoa. Trả về cho từng cụm:
  written : trích NGUYÊN VĂN từ câu, không sửa một ký tự nào
  type    : một kiểu trong danh sách dưới
  spoken  : cách người Việt đọc cụm đó, TOÀN CHỮ THƯỜNG, mọi con số viết bằng chữ

Danh sách kiểu:
{types}
Tên tổ chức, đơn vị, cơ quan không thuộc kiểu nào ở trên thì dùng ORGANIZATION.

QUY TẮC:
- `written` phải xuất hiện y nguyên trong câu, phân biệt hoa thường.
- `written` là phần NGẮN NHẤT thay đổi khi đọc. TUYỆT ĐỐI không kèm từ ngữ cảnh.
    ĐÚNG : written="4"                type=CARDINAL   spoken="bốn"
    SAI  : written="Quân khu 4"       type=CARDINAL   spoken="quân khu bốn"
  Riêng tên người, địa danh, viết tắt thì lấy TRỌN tên:
    ĐÚNG : written="Nguyễn Quốc Trị"  type=PERSON_NAME spoken="nguyễn quốc trị"
- `spoken` phải đọc ĐÚNG phần trong `written`, không thừa không thiếu chữ nào.
- `spoken` không được chứa ký tự số.
- Các cụm không được chồng lấn nhau.
- Không bịa cụm không có trong câu. Câu không có cụm nào thì trả mảng rỗng.

CÁC CÂU:
{sentences}"""


def build_prompt(batch):
    lines = [f"[{i}] {text}" for i, text in batch]
    return PROMPT.format(types=" ".join(SEMANTIC_TYPES), sentences="\n".join(lines))


def apply_spans(text, spans):
    """Ghép dạng nói + chỉ số span theo từ. Trả (từ, span, nhãn dấu câu) hoặc None."""
    placed = []
    used = []
    for span in spans:
        written = span.get("written", "").strip()
        spoken = span.get("spoken", "").strip()
        type_name = span.get("type", "").strip().upper()
        if not written or not spoken or not type_name:
            continue
        start = -1
        for match in re.finditer(re.escape(written), text):
            if all(match.end() <= a or match.start() >= b for a, b in used):
                start = match.start()
                break
        if start < 0:
            continue
        used.append((start, start + len(written)))
        placed.append((start, start + len(written), written, spoken, type_name))
    placed.sort()

    words, spans_out, puncts = [], [], []

    def add_plain(chunk):
        for raw in chunk.split():
            tail = "O"
            token = raw
            while token and token[-1] in STRIP_CHARS:
                if token[-1] in PUNCT_MAP:
                    tail = PUNCT_MAP[token[-1]]
                token = token[:-1]
            token = token.strip(STRIP_CHARS)
            if not token:
                if puncts and tail != "O":
                    puncts[-1] = tail
                continue
            words.append(token.lower())
            puncts.append(tail)

    cursor = 0
    for start, end, written, spoken, type_name in placed:
        add_plain(text[cursor:start])
        span_words = [w.lower() for w in spoken.split()]
        if not span_words:
            cursor = end
            continue
        begin = len(words)
        words.extend(span_words)
        puncts.extend(["O"] * len(span_words))
        # dấu câu dính ngay sau cụm
        tail = text[end:end + 1]
        if tail in PUNCT_MAP:
            puncts[-1] = PUNCT_MAP[tail]
        spans_out.append((begin, len(words) - 1, type_name))
        cursor = end
    add_plain(text[cursor:])
    if not words:
        return None
    return words, spans_out, puncts


def verify_span(spoken_words, written, type_name):
    """Chỉ nhận nhãn nào normalizer tái tạo đúng dạng viết trong câu gốc."""
    if type_name not in SEMANTIC_TYPES:
        return False
    normalizer = get_normalizer(type_name)
    if normalizer is None:
        return False
    target = written.strip().strip(STRIP_CHARS)
    result = normalizer(" ".join(spoken_words), type_name)
    if not result.valid or result.normalized != target:
        return False
    ok, _ = validate(type_name, result.normalized)
    return ok


def make_sample(text, raw_spans, source, stats):
    kept_spans = []
    for span in raw_spans:
        type_name = span.get("type", "").strip().upper()
        if type_name in OUT_OF_TAXONOMY:
            stats[f"ngoài taxonomy: {type_name}"] += 1
            OUT_OF_TAXONOMY_EXAMPLES.append(
                {"type": type_name, "written": span.get("written", ""),
                 "spoken": span.get("spoken", ""), "context": text[:120]})
            continue
        if type_name not in SEMANTIC_TYPES:
            stats["kiểu lạ"] += 1
            continue
        kept_spans.append(span)
    if not kept_spans:
        return None

    applied = apply_spans(text, kept_spans)
    if applied is None:
        stats["không đặt được span vào câu"] += 1
        return None
    words, spans, puncts = applied
    if not spans:
        stats["không span nào khớp câu"] += 1
        return None
    if any(ch.isdigit() for w in words for ch in w):
        stats["dạng nói còn chữ số"] += 1
        return None

    verified = []
    for (begin, end, type_name), span in zip(spans, kept_spans):
        if verify_span(words[begin:end + 1], span["written"], type_name):
            verified.append((begin, end, type_name))
        else:
            stats[f"không xác thực: {type_name}"] += 1
    if not verified:
        return None

    spoken_text = " ".join(words)
    try:
        written, outputs, _al, conflicts = normalize_with_gold_spans(
            spoken_text, verified, puncts)
    except Exception as exc:
        stats[f"pipeline lỗi: {type(exc).__name__}"] += 1
        return None
    if conflicts:
        stats["từ ghép cắt ngang span"] += 1
        return None
    if not all(o.emitted for o in outputs):
        stats["span không phát ra được"] += 1
        return None

    sample = Sample(
        id=hashlib.sha1(spoken_text.encode()).hexdigest()[:16],
        spoken=spoken_text, written=written, words=words,
        spans=verified, punct=puncts,
        types=sorted({t for _, _, t in verified}),
        leak_key="|".join(sorted(o.normalized for o in outputs)),
        source=source, written_source=" ".join(text.split()),
    )
    if validate_sample(sample):
        stats["schema không hợp lệ"] += 1
        return None
    return sample


def load_chunks(input_dir, limit=None, min_words=8, max_words=60):
    seen, out = set(), []
    for path in sorted((ROOT / input_dir).rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        topic = doc.get("article", {}).get("topic", "?")
        for chunk in doc.get("chunks", []):
            text = " ".join(unicodedata.normalize("NFC", chunk.get("text", "")).split())
            if not (min_words <= len(text.split()) <= max_words):
                continue
            key = hashlib.sha1(text.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append((text, topic))
            if limit and len(out) >= limit:
                return out
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_v1/json")
    parser.add_argument("--out", default="datasets_v2/gemini_labeled.jsonl")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8, help="số câu mỗi lượt gọi")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--tpm", type=int, default=200_000)
    args = parser.parse_args(argv)

    chunks = load_chunks(args.input, args.limit)
    print(f"{len(chunks)} câu gốc đủ điều kiện")
    batches = [chunks[i:i + args.batch] for i in range(0, len(chunks), args.batch)]
    print(f"{len(batches)} lượt gọi, {args.workers} luồng, model {args.model}, "
          f"hạn mức {args.rpm} RPM / {args.tpm} TPM")

    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm, rpd=10 ** 9)
    client = GemmaClient(model=args.model, limiter=limiter)
    stats, kept = Counter(), {}
    done = 0

    def work(batch):
        indexed = list(enumerate(t for t, _ in batch))
        prompt = build_prompt(indexed)
        text = client.generate(prompt, temperature=0.2, max_tokens=4096,
                               response_schema=RESPONSE_SCHEMA)
        try:
            payload = json.loads(text)
        except Exception:
            return [], Counter({"JSON trả về hỏng": 1})
        local, local_stats = [], Counter()
        for item in payload.get("items", []):
            idx = item.get("index")
            if not isinstance(idx, int) or not 0 <= idx < len(batch):
                local_stats["index sai"] += 1
                continue
            source_text, topic = batch[idx]
            sample = make_sample(source_text, item.get("spans", []),
                                 f"gemini:{topic}", local_stats)
            if sample:
                local.append(sample)
        return local, local_stats

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, b): b for b in batches}
        for future in as_completed(futures):
            done += 1
            try:
                samples, local_stats = future.result()
            except Exception as exc:
                stats[f"gọi API lỗi: {type(exc).__name__}"] += 1
                continue
            stats.update(local_stats)
            for s in samples:
                kept.setdefault(s.id, s)
            if done % 5 == 0 or done == len(batches):
                print(f"  [{done}/{len(batches)}] giữ {len(kept)} câu", flush=True)

    samples = list(kept.values())
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, samples)

    types = Counter(t for s in samples for t in s.types)
    print(f"\nghi {len(samples)} câu vào {out_path}")
    print(f"gọi API: {client.calls} lần, {client.retries} lần thử lại, "
          f"{client.tokens} token, chờ hạn mức {limiter.waited:.0f}s")
    print("\n--- phủ kiểu ---")
    for t in SEMANTIC_TYPES:
        if types.get(t):
            print(f"  {t:16s} {types[t]}")
    print(f"kiểu chưa có: {[t for t in SEMANTIC_TYPES if not types.get(t)]}")
    if OUT_OF_TAXONOMY_EXAMPLES:
        side = out_path.with_name("out_of_taxonomy.jsonl")
        with side.open("w", encoding="utf-8") as fh:
            for item in OUT_OF_TAXONOMY_EXAMPLES:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n{len(OUT_OF_TAXONOMY_EXAMPLES)} cụm NGOÀI taxonomy -> {side}")
        print("  (bằng chứng cho quyết định có mở thêm lớp hay không)")

    print("\n--- thống kê loại (15 đầu) ---")
    for why, n in stats.most_common(15):
        print(f"  {n:5d}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
