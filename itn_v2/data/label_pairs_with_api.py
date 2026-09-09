"""Gán nhãn kiểu cho corpus cặp 1:1 bằng API.

Chia việc theo đúng thế mạnh của từng bên:

  TẤT ĐỊNH  căn hàng dạng viết <-> dạng nói để tìm VÙNG cần chuẩn hoá.
            Đây là phép so chuỗi, không cần mô hình ngôn ngữ.
  API       chỉ trả lời MỘT câu hỏi: vùng này thuộc kiểu nào trong 46 kiểu.
  TẤT ĐỊNH  xác thực: normalizer chạy trên dạng nói phải tái tạo đúng dạng viết.

Hai nguồn vùng:
  1. khác về TỪ    — "55" <-> "năm mươi lăm"
  2. khác về HOA   — "Việt Nam" <-> "việt nam" (ASR trả về chữ thường)

Mỗi cụm chỉ hỏi API một lần rồi dùng lại cho mọi câu — "Việt Nam" xuất hiện
hàng trăm lần nhưng chỉ tốn một lần hỏi.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

from ..inference import normalize_with_gold_spans
from ..labels import SEMANTIC_TYPES
from ..registry import get_normalizer
from ..validators import validate
from .from_data_v1 import PUNCT_MAP, STRIP_CHARS
from .gemma_client import GemmaClient, RateLimiter, load_keys
from .schema import Sample, validate_sample, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
OUT_OF_TAXONOMY = "ORGANIZATION"
NO_TYPE = "NONE"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "type": {"type": "string"}},
                "required": ["index", "type"],
            },
        }
    },
    "required": ["items"],
}

PROMPT = """Bạn phân loại cụm cho hệ chuẩn hoá văn bản tiếng Việt sau nhận dạng tiếng nói.

Mỗi dòng dưới đây là một cụm, gồm dạng VIẾT và dạng NÓI của cùng nội dung, kèm
ngữ cảnh. Hãy cho biết cụm đó thuộc KIỂU nào.

Kiểu hợp lệ:
{types}
ORGANIZATION — tên tổ chức, cơ quan, đơn vị (Bộ Quốc phòng, Quân chủng Hải quân)
NONE — không thuộc kiểu nào, hoặc chỉ là từ thường

Trả về đúng một `type` cho mỗi `index`.

CÁC CỤM:
{items}"""

def is_capitalized(word):
    """Chữ cái đầu có viết hoa không.

    KHÔNG dùng lớp ký tự kiểu [A-ZĐÀ-Ỹ]: dải Unicode À-Ỹ là U+00C0..U+1EF8, bao
    trùm luôn toàn bộ chữ THƯỜNG tiếng Việt (à, á, đ, ạ...). Dùng nó thì "đôi",
    "đã", "ánh" đều bị coi là viết hoa và sinh ra hàng nghìn vùng ứng viên rác.
    """
    return bool(word) and word[0].isupper()


def norm_chunk(text):
    return text.strip().strip(STRIP_CHARS)


_DIGITS = re.compile(r"\d+")
_NON_LETTER = re.compile(r"[^a-zà-ỹđ]")


def _groups(text):
    return _DIGITS.findall(text)


def _letters(text):
    return _NON_LETTER.sub("", text.lower())


def same_value(ours, theirs):
    """Hai cách VIẾT của cùng một GIÁ TRỊ có được coi là khớp không.

    Bài báo viết 'Tháng 6 1968', bộ chuẩn hoá của ta cho '06/1968'. Trước đây
    cổng kiểm chứng so chuỗi thô nên coi đây là sai và loại cụm — 239 cụm DATE
    chết vì lý do này, mỗi cụm biến thành một mẫu âm dạy mô hình ĐỪNG đụng vào
    ngày tháng. Ta chỉ cần chắc LLM gán đúng KIỂU; cách trình bày thì lấy theo
    bộ chuẩn hoá của mình, vì đó mới là quy ước nhất quán.
    """
    a, b = norm_chunk(ours), norm_chunk(theirs)
    if a == b:
        return True
    ga, gb = _groups(a), _groups(b)
    if not ga or _letters(a) != _letters(b):
        return False
    if len(ga) == len(gb):
        strip = lambda g: g.lstrip("0") or "0"
        return [strip(g) for g in ga] == [strip(g) for g in gb]
    # dấu phân nhóm nghìn: '1.968' so với '1968'. Bên nhiều nhóm hơn phải có
    # mọi nhóm sau nhóm đầu đúng 3 chữ số, nếu không '1/6' sẽ khớp nhầm '16'.
    more, less = (ga, gb) if len(ga) > len(gb) else (gb, ga)
    if len(less) != 1 or not all(len(g) == 3 for g in more[1:]) or len(more[0]) > 3:
        return False
    return "".join(more) == less[0]


def candidate_regions(written, spoken):
    """Trả [(written_chunk, spoken_chunk, (j1, j2))] theo chỉ số TỪ của dạng nói."""
    w, s = written.split(), spoken.split()
    matcher = SequenceMatcher(None, [x.lower() for x in w], [x.lower() for x in s])
    regions = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            if j1 < j2 and i1 < i2:
                regions.append((" ".join(w[i1:i2]), " ".join(s[j1:j2]), (j1, j2)))
            continue
        # trong vùng giống nhau: bắt các cụm VIẾT HOA liên tiếp
        k = 0
        while k < i2 - i1:
            if not is_capitalized(w[i1 + k]):
                k += 1
                continue
            start = k
            while k < i2 - i1 and is_capitalized(w[i1 + k]):
                k += 1
            if i1 + start == 0:          # đầu câu: hoa do chính tả, không phải tên riêng
                continue
            regions.append((" ".join(w[i1 + start:i1 + k]),
                            " ".join(s[j1 + start:j1 + k]),
                            (j1 + start, j1 + k)))
    return regions


def verify(spoken_chunk, written_chunk, type_name):
    if type_name not in SEMANTIC_TYPES:
        return False
    normalizer = get_normalizer(type_name)
    if normalizer is None:
        return False
    result = normalizer(spoken_chunk, type_name)
    if not result.valid or not same_value(result.normalized, written_chunk):
        return False
    return validate(type_name, result.normalized)[0]


def split_with_punct(words_text):
    words, puncts = [], []
    for raw in words_text.split():
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
    return words, puncts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="datasets_v2/v1_pairs.jsonl")
    parser.add_argument("--out", default="datasets_v2/v1_labeled.jsonl")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--tpm", type=int, default=250_000)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--cache", default="datasets_v2/api_types_cache.json",
                        help="nhớ câu trả lời của API để chạy lại không tốn quota")
    parser.add_argument("--keep-rejected", action="store_true",
                        help="giữ câu có cụm bị loại (hành vi cũ: sinh mẫu âm giả)")
    parser.add_argument("--rejects-out", default="datasets_v2/v1_rejected.jsonl")
    parser.add_argument("--key-file", default="api.txt",
                        help="file chứa các khoá API, mỗi dòng một khoá")
    args = parser.parse_args(argv)

    pairs = [json.loads(l) for l in (ROOT / args.pairs).open(encoding="utf-8") if l.strip()]
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]
    print(f"{len(pairs)} cặp 1:1")

    # gom vùng, khử trùng lặp trước khi hỏi API
    per_pair, unique = [], {}
    for pair in pairs:
        regions = candidate_regions(pair["written"], pair["spoken"])
        per_pair.append(regions)
        for w_chunk, s_chunk, _ in regions:
            key = (norm_chunk(w_chunk), s_chunk.strip().lower())
            if key not in unique:
                unique[key] = pair["written"]
    total_regions = sum(len(r) for r in per_pair)
    print(f"{total_regions} vùng ứng viên, {len(unique)} cụm khác nhau "
          f"-> chỉ hỏi API {len(unique)} lần thay vì {total_regions}")

    cache_path = ROOT / args.cache
    cached = {}
    if cache_path.exists():
        for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items():
            written_chunk, _, spoken_chunk = k.partition("\u241f")
            cached[(written_chunk, spoken_chunk)] = v
        print(f"cache: {len(cached)} cụm đã có sẵn")

    keys = [k for k in unique if k not in cached]
    batches = [keys[i:i + args.batch] for i in range(0, len(keys), args.batch)]
    print(f"{len(batches)} lượt gọi, model {args.model}, {args.rpm} RPM")

    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm, rpd=10 ** 9)
    client = GemmaClient(model=args.model, limiter=limiter,
                         keys=load_keys(str(ROOT / args.key_file)))
    type_of, stats = dict(cached), Counter()

    def work(batch_keys):
        lines = []
        for i, key in enumerate(batch_keys):
            written_chunk, spoken_chunk = key
            ctx = unique[key]
            lines.append(f'[{i}] viết="{written_chunk}" nói="{spoken_chunk}" '
                         f'ngữ_cảnh="{ctx[:110]}"')
        prompt = PROMPT.format(types=" ".join(SEMANTIC_TYPES), items="\n".join(lines))
        text = client.generate(prompt, temperature=0.0, max_tokens=4096,
                               response_schema=RESPONSE_SCHEMA)
        out = {}
        try:
            payload = json.loads(text)
        except Exception:
            return out, Counter({"JSON hỏng": 1})
        local = Counter()
        for item in payload.get("items", []):
            idx = item.get("index")
            if not isinstance(idx, int) or not 0 <= idx < len(batch_keys):
                local["index sai"] += 1
                continue
            out[batch_keys[idx]] = str(item.get("type", "")).strip().upper()
        return out, local

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, b): b for b in batches}
        for future in as_completed(futures):
            done += 1
            try:
                mapping, local = future.result()
            except Exception as exc:
                stats[f"gọi API lỗi: {type(exc).__name__}"] += 1
                continue
            type_of.update(mapping)
            stats.update(local)
            if done % 10 == 0 or done == len(batches):
                print(f"  [{done}/{len(batches)}] đã phân loại {len(type_of)} cụm",
                      flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {f"{w}\u241f{sp}": t for (w, sp), t in type_of.items()},
        ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"cache: đã ghi {len(type_of)} cụm vào {args.cache}")

    # dựng mẫu
    kept, org_examples, rejected = {}, [], []
    for pair, regions in zip(pairs, per_pair):
        words, puncts = split_with_punct(pair["spoken"])
        verified, bad = [], []
        for w_chunk, s_chunk, (j1, j2) in regions:
            key = (norm_chunk(w_chunk), s_chunk.strip().lower())
            type_name = type_of.get(key, NO_TYPE)
            # Vùng chỉ khác hoa-thường thì bộ dựng câu tự xử lý được, bỏ qua
            # nó KHÔNG tạo mâu thuẫn nhãn. Vùng khác nhau ở TỪ thì có; nếu ta
            # không gán được kiểu cho nó mà vẫn giữ câu lại, câu đó dạy mô hình
            # rằng chỗ ấy phải để nguyên — trái với hàng trăm câu khác.
            words_differ = w_chunk.lower().split() != s_chunk.lower().split()
            if type_name == OUT_OF_TAXONOMY:
                stats["ngoài taxonomy: ORGANIZATION"] += 1
                org_examples.append({"written": w_chunk, "spoken": s_chunk,
                                     "context": pair["written"][:120]})
                continue
            if type_name in (NO_TYPE, ""):
                stats["API trả NONE (chỉ khác hoa-thường)" if not words_differ
                      else "API trả NONE (khác từ) -> BỎ CÂU"] += 1
                if words_differ:
                    bad.append((w_chunk, s_chunk, "NONE"))
                continue
            if type_name not in SEMANTIC_TYPES:
                stats["kiểu lạ"] += 1
                bad.append((w_chunk, s_chunk, f"kiểu lạ {type_name}"))
                continue
            if j2 > len(words):
                stats["lệch chỉ số"] += 1
                bad.append((w_chunk, s_chunk, "lệch chỉ số"))
                continue
            if not verify(" ".join(words[j1:j2]), w_chunk, type_name):
                stats[f"không xác thực: {type_name}"] += 1
                bad.append((w_chunk, s_chunk, f"không xác thực {type_name}"))
                continue
            verified.append((j1, j2 - 1, type_name))
        if bad and not args.keep_rejected:
            stats["BỎ CÂU vì có cụm không gán được"] += 1
            rejected.append({"id": pair["id"], "spoken": pair["spoken"],
                             "written": pair["written"],
                             "cụm_hỏng": [{"viết": w, "nói": sp, "lý_do": r}
                                          for w, sp, r in bad]})
            continue
        if not verified:
            stats["giữ làm hard negative (không span)"] += 1
        verified.sort()
        deduped, last_end = [], -1
        for start, end, type_name in verified:
            if start > last_end:
                deduped.append((start, end, type_name))
                last_end = end
        spoken_text = " ".join(words)
        try:
            written, outputs, _al, conflicts = normalize_with_gold_spans(
                spoken_text, deduped, puncts)
        except Exception as exc:
            stats[f"pipeline lỗi: {type(exc).__name__}"] += 1
            continue
        if conflicts or not all(o.emitted for o in outputs):
            stats["span không phát ra được"] += 1
            continue
        sample = Sample(
            id=hashlib.sha1(spoken_text.encode()).hexdigest()[:16],
            spoken=spoken_text, written=written, words=words,
            spans=deduped, punct=puncts,
            types=sorted({t for _, _, t in deduped}),
            leak_key="|".join(sorted(o.normalized for o in outputs)),
            source=f"v1pair:{pair['topic']}", written_source=pair["written"])
        if validate_sample(sample):
            stats["schema không hợp lệ"] += 1
            continue
        kept.setdefault(sample.id, sample)

    samples = list(kept.values())
    out_path = ROOT / args.out
    write_jsonl(out_path, samples)
    if org_examples:
        side = out_path.with_name("out_of_taxonomy.jsonl")
        with side.open("w", encoding="utf-8") as fh:
            for item in org_examples:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    if rejected:
        rej_path = ROOT / args.rejects_out
        with rej_path.open("w", encoding="utf-8") as fh:
            for item in rejected:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"loại {len(rejected)} câu (xem {args.rejects_out})")

    types = Counter(t for s in samples for t in s.types)
    print(f"\nghi {len(samples)} câu vào {out_path}")
    print(f"gọi API {client.calls} lần, {client.tokens} token, "
          f"chờ hạn mức {limiter.waited:.0f}s")
    print("\n--- phủ kiểu ---")
    for t in SEMANTIC_TYPES:
        if types.get(t):
            print(f"  {t:16s} {types[t]}")
    print("\n--- thống kê (15 đầu) ---")
    for why, n in stats.most_common(15):
        print(f"  {n:5d}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
