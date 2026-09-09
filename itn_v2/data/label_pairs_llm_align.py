"""Gán nhãn corpus 1:1 bằng MỘT lượt hỏi: API vừa chia span vừa xuất dạng viết.

Khác bản `label_pairs_with_api`: ở đó vùng cần chuẩn hoá do phép so chuỗi tìm
ra, API chỉ gán kiểu, rồi normalizer của ta phải tái tạo đúng dạng viết của bài
báo mới được nhận. Cách đó bị chặn bởi độ phủ của code — cụm nào normalizer
chưa biết đọc thì bị loại, và câu vẫn ở lại tập huấn luyện với cụm đó gán O,
tức là dạy mô hình điều ngược lại.

Ở đây văn bản ITN của bài báo đã là đáp án. Việc còn lại chỉ là CĂN HÀNG: cho
API cả câu viết lẫn câu nói cùng danh sách kiểu, bảo nó liệt kê từng span gồm
(chuỗi nói, chuỗi viết, kiểu). Không cần normalizer nào lúc gán nhãn.

Cổng kiểm chứng vẫn tất định và còn chặt hơn trước: ghép mọi span trở lại câu
nói phải tái tạo ĐÚNG văn bản của bài báo. Nếu còn sai một ký tự nghĩa là bộ
span chưa giải thích hết, câu bị loại chứ không lặng lẽ thành mẫu âm.
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
from .from_data_v1 import PUNCT_MAP, STRIP_CHARS
from .gemma_client import GemmaClient, RateLimiter, load_keys
from .schema import Sample, validate_sample, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
OUT_OF_TAXONOMY = "ORGANIZATION"
SEP = "␟"
CACHE_VERSION = 2

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
                                "noi": {"type": "string"},
                                "viet": {"type": "string"},
                                "kieu": {"type": "string"},
                            },
                            "required": ["noi", "viet", "kieu"],
                        },
                    },
                },
                "required": ["index", "spans"],
            },
        }
    },
    "required": ["items"],
}

PROMPT = """Bạn căn hàng dữ liệu cho hệ chuẩn hoá văn bản tiếng Việt sau nhận dạng tiếng nói.

Mỗi mục dưới đây có hai dòng của CÙNG một câu:
  NOI  = dạng nói, toàn chữ thường, mọi con số đã đọc thành chữ.
  VIET = dạng viết đã chuẩn hoá, lấy nguyên từ bài báo. Đây là ĐÁP ÁN.

Việc của bạn: liệt kê những đoạn của NOI cần đổi để thành VIET. Mỗi đoạn gồm:
  "noi"  = chuỗi con LIÊN TỤC, chép NGUYÊN VĂN từ NOI, không thêm bớt từ nào.
  "viet" = đoạn tương ứng, chép NGUYÊN VĂN từ VIET.
  "kieu" = một kiểu trong danh sách dưới.

Quy tắc bắt buộc:
- Thay lần lượt từng "noi" bằng "viet" trong câu NOI phải ra đúng câu VIET.
  Chỉ được bỏ qua khác biệt về dấu câu và về viết hoa đầu câu.
- Đoạn nào chỉ khác nhau ở chữ hoa (tên riêng, địa danh) VẪN phải liệt kê.
- Không liệt kê đoạn giống hệt nhau ở cả hai bên.
- Lấy đoạn NGẮN NHẤT đủ nghĩa. "mười sáu tuổi" -> chỉ lấy "mười sáu" = "16".
- Ngày tháng lấy trọn cụm: "tháng sáu một nghìn chín trăm sáu mươi tám" là MỘT
  đoạn kiểu DATE, không tách thành tháng riêng năm riêng.
- Tên cơ quan, đơn vị, tổ chức thì dùng kiểu ORGANIZATION.
- Nếu câu không có đoạn nào cần đổi, trả "spans": [].

Kiểu hợp lệ:
{types} ORGANIZATION

Các mục:
{items}

Trả JSON: {{"items":[{{"index":0,"spans":[{{"noi":"...","viet":"...","kieu":"..."}}]}}]}}
"""


# Kiểu BẮT BUỘC sinh ra chữ số. Nếu dạng viết không có chữ số nào thì LLM đã
# gán sai kiểu — hầu hết là nó thấy khác nhau ở chữ hoa đầu câu rồi chọn bừa
# ("rồi" -> "Rồi" gán CARDINAL, "tác" -> "Tác" gán HEADING).
MUST_HAVE_DIGIT = {
    "CARDINAL", "ORDINAL", "DIGIT_SEQ", "DECIMAL", "FRACTION", "PERCENT",
    "RANGE", "RATIO", "MONEY", "MEASURE", "VERSION", "DATE", "TIME", "DURATION",
    "ETA", "ETD", "QUARTER", "COORD", "HEADING", "BEARING", "SPEED", "DISTANCE",
    "DEPTH", "DRAFT", "FREQUENCY", "CHANNEL", "MMSI_ID", "IMO_ID", "VESSEL_ID",
    "TELEPHONE", "VEHICLE_PLATE", "EQUIPMENT_ID", "LEGAL_DOC_ID", "DOCUMENT_ID",
}

# Kiểu mà khác biệt CHỈ ở chữ hoa là hợp lệ: đó chính là việc khôi phục hoa
# cho tên riêng ("việt nam" -> "Việt Nam"), phần thật của bài toán.
CASE_ONLY_OK = {
    "PERSON_NAME", "LOCATION_NAME", "FOREIGN_NAME", "ORGANIZATION",
    "EQUIPMENT_NAME", "MARITIME_TERM", "RANK",
}

# Các kiểu mà casing là chính nội dung chuẩn hoá. Nếu formatter
# của ta không dựng đúng bề mặt mà bài báo dùng thì span đó không
# phải gold có thể đạt được khi suy luận.
STRICT_SOURCE_TYPES = {
    "PERSON_NAME", "LOCATION_NAME", "FOREIGN_NAME", "EQUIPMENT_NAME",
    "MARITIME_TERM", "RANK", "ACRONYM",
}

DIGIT_RE = re.compile(r"\d")


def span_is_sane(noi, viet, kieu):
    """Lọc tất định các span LLM gán sai kiểu. Trả (giữ?, lý_do_bỏ)."""
    if re.sub(r"\s+", " ", noi.strip()) == re.sub(r"\s+", " ", viet.strip()):
        return False, "dạng nói và dạng viết giống hệt nhau"
    has_digit = bool(DIGIT_RE.search(viet))
    if kieu in MUST_HAVE_DIGIT and not has_digit:
        return False, f"{kieu} nhưng dạng viết không có chữ số"
    case_only = strip_marks(noi) == strip_marks(viet)
    if case_only:
        if kieu not in CASE_ONLY_OK:
            return False, f"{kieu} nhưng chỉ khác chữ hoa"
        if kieu == "ACRONYM":
            return False, "ACRONYM nhưng chỉ khác chữ hoa"
    if kieu == "ACRONYM" and not any(c.isupper() for c in viet):
        return False, "ACRONYM nhưng dạng viết không có chữ hoa"
    return True, ""


def split_with_punct(words_text):
    """Tách thành từ + nhãn dấu câu đi kèm mỗi từ (giống bản gán nhãn cũ)."""
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


def strip_marks(text):
    """Rút gọn để so sánh: bỏ dấu câu, gộp khoảng trắng, hạ chữ thường."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s/:.,+-]", " ", text)
    text = re.sub(r"[,.;:!?]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


SENTENCE_INITIAL_RE = re.compile(r"(^|[.!?]\s*)([^\W\d_])", re.UNICODE)
PUNCT_CHAR = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}


def comparison_form(text):
    """Dạng so sánh bảo toàn casing, chỉ bỏ casing do đầu câu.

    ``strip_marks`` cũ hạ toàn bộ chuỗi nên "việt nam" và "Việt Nam"
    bị coi là khớp. Renderer tự viết hoa đầu câu, nên chỉ ký tự
    đầu câu được phép khác casing.
    """
    text = unicodedata.normalize("NFC", text)
    text = SENTENCE_INITIAL_RE.sub(
        lambda m: m.group(1) + m.group(2).lower(), text)
    text = re.sub(r"[^\w\s/:.,+-]", " ", text)
    text = re.sub(r"[,.;:!?]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def surface_form(text):
    """So sánh bề mặt một span, không nới casing ở ký tự đầu."""
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"[^\w\s/:.,+-]", " ", text)
    text = re.sub(r"[,.;:!?]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def only_sentence_initial_case(noi, viet):
    """Khác đúng chữ cái đầu: ``bà`` -> ``Bà``, không phải entity.

    ``chan phal`` -> ``Chan Phal`` trả False vì chữ P vẫn khác; span đó
    phải được giữ.
    """
    a = re.sub(r"\s+", " ", noi.strip())
    b = re.sub(r"\s+", " ", viet.strip())
    if not a or not b or a == b:
        return False
    return a == b[0].lower() + b[1:]


def find_run(words, phrase_words, taken):
    """Vị trí của một dãy từ liên tục trong câu nói, bỏ qua vùng đã dùng."""
    n = len(phrase_words)
    if not n:
        return None
    for i in range(len(words) - n + 1):
        if words[i:i + n] != phrase_words:
            continue
        if any(taken[i:i + n]):
            continue
        return i, i + n
    return None


def reconstruct(words, spans, written_of, puncts=None):
    """Ghép câu nói với các span đã thay bằng dạng viết."""
    out, i = [], 0
    by_start = {s: (e, t) for s, e, t in spans}
    while i < len(words):
        if i in by_start:
            end, _ = by_start[i]
            text = written_of[i]
            if puncts:
                text += PUNCT_CHAR.get(puncts[end], "")
            out.append(text)
            i = end + 1
        else:
            text = words[i]
            if puncts:
                text += PUNCT_CHAR.get(puncts[i], "")
            out.append(text)
            i += 1
    return " ".join(out)


def parse_spans(raw_spans, words, puncts, stats):
    """Đưa câu trả lời của API về (bắt_đầu, kết_thúc, kiểu) + dạng viết."""
    taken = [False] * len(words)
    placed, written_of = [], {}
    for item in raw_spans:
        noi = str(item.get("noi", "")).strip().lower()
        viet = str(item.get("viet", "")).strip()
        kieu = str(item.get("kieu", "")).strip().upper()
        if not noi or not viet or not kieu:
            stats["span thiếu trường"] += 1
            continue
        ok, why = span_is_sane(noi, viet, kieu)
        if not ok:
            # Bỏ RIÊNG span sai, giữ câu: dạng viết của span này bằng dạng nói
            # (chỉ khác chữ hoa) nên bộ dựng câu vẫn tái tạo đúng được.
            stats[f"span bỏ: {why}"] += 1
            continue
        if kieu != OUT_OF_TAXONOMY and kieu not in SEMANTIC_TYPES:
            stats[f"kiểu lạ: {kieu}"] += 1
            return None, None
        phrase = [w.strip(STRIP_CHARS) for w in noi.split()]
        phrase = [w for w in phrase if w]
        hit = find_run(words, phrase, taken)
        if hit is None:
            stats["không tìm thấy chuỗi nói trong câu"] += 1
            return None, None
        i, j = hit
        at_sentence_start = i == 0 or puncts[i - 1] in {"PERIOD", "QUESTION"}
        if strip_marks(noi) == strip_marks(viet) and at_sentence_start \
                and only_sentence_initial_case(noi, viet):
            stats[f"span bỏ: {kieu} nhưng chỉ là viết hoa đầu câu"] += 1
            continue
        for k in range(i, j):
            taken[k] = True
        placed.append((i, j - 1, kieu))
        written_of[i] = viet
    placed.sort()
    return placed, written_of


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="datasets_v2/v1_pairs.jsonl")
    parser.add_argument("--out", default="datasets_v2/v1_aligned.jsonl")
    parser.add_argument("--rejects-out", default="datasets_v2/v1_aligned_rejected.jsonl")
    parser.add_argument("--gaps-out", default="datasets_v2/normalizer_gaps.jsonl")
    parser.add_argument("--out-of-taxonomy-out",
                        default="datasets_v2/out_of_taxonomy.jsonl")
    parser.add_argument("--cache", default="datasets_v2/align_cache_v2.json")
    parser.add_argument("--legacy-cache", default="datasets_v2/align_cache.json",
                        help="chỉ tái dùng câu không biến thể có đầu vào y hệt")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--tpm", type=int, default=250_000)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--keep-organization", action="store_true",
                        help="giữ span ORGANIZATION làm nhãn thay vì bỏ qua")
    parser.add_argument("--key-file", default="api.txt",
                        help="file chứa các khoá API, mỗi dòng một khoá")
    args = parser.parse_args(argv)

    pairs = [json.loads(l) for l in (ROOT / args.pairs).open(encoding="utf-8") if l.strip()]
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]
    print(f"{len(pairs)} cặp 1:1")

    def fingerprint(pair):
        payload = f"{CACHE_VERSION}{SEP}{pair['spoken']}{SEP}{pair['written']}"
        return hashlib.sha256(payload.encode()).hexdigest()

    pair_by_id = {p["id"]: p for p in pairs}
    cache_path = ROOT / args.cache
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}
    if not isinstance(cache_payload, dict) or \
            cache_payload.get("version") != CACHE_VERSION:
        if cache_payload:
            print("cache cũ/không đúng version: không tái dùng")
        cache_payload = {"version": CACHE_VERSION, "items": {}}
    cached_items = cache_payload.get("items", {})
    cache = {
        pid: row for pid, row in cached_items.items()
        if pid in pair_by_id and isinstance(row, dict)
        and row.get("fingerprint") == fingerprint(pair_by_id[pid])
        and isinstance(row.get("spans"), list)
    }
    # Câu ``không_biến_thể`` đã là lowercase ở pipeline cũ, nên
    # prompt của nó không đổi. Chỉ nhóm có biến thể bị hạ casing mới
    # bắt buộc hỏi lại. Mục tái dùng vẫn được đóng gói bằng
    # fingerprint v2; những lần sau không còn phụ thuộc cache cũ.
    legacy_path = ROOT / args.legacy_cache if args.legacy_cache else None
    migrated = 0
    if legacy_path and legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        if isinstance(legacy, dict) and "items" not in legacy:
            for pair in pairs:
                raw = legacy.get(pair["id"])
                if pair.get("kind") == "không_biến_thể" \
                        and pair["spoken"] == pair["spoken"].lower() \
                        and isinstance(raw, list) and pair["id"] not in cache:
                    cache[pair["id"]] = {
                        "fingerprint": fingerprint(pair), "spans": raw}
                    migrated += 1
    if migrated:
        print(f"cache: tái dùng an toàn {migrated} câu không biến thể")
    print(f"cache: {len(cache)} câu đã có sẵn và đúng đầu vào")

    todo = [p for p in pairs if p["id"] not in cache]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"cần hỏi {len(todo)} câu -> {len(batches)} lượt gọi, "
          f"model {args.model}, {args.rpm} RPM")

    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm, rpd=10 ** 9)
    client = GemmaClient(model=args.model, limiter=limiter,
                         keys=load_keys(str(ROOT / args.key_file)))
    stats = Counter()

    def work(batch):
        lines = []
        for i, pair in enumerate(batch):
            lines.append(f'[{i}]\nNOI  = {pair["spoken"]}\nVIET = {pair["written"]}')
        prompt = PROMPT.format(types=" ".join(SEMANTIC_TYPES), items="\n".join(lines))
        text = client.generate(prompt, temperature=0.0, max_tokens=8192,
                               response_schema=RESPONSE_SCHEMA)
        out = {}
        payload = json.loads(text)
        for item in payload.get("items", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(batch):
                out[batch[idx]["id"]] = item.get("spans", [])
        return out

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, b) for b in batches]
        for future in as_completed(futures):
            done += 1
            try:
                for pid, spans in future.result().items():
                    cache[pid] = {
                        "fingerprint": fingerprint(pair_by_id[pid]),
                        "spans": spans,
                    }
            except Exception as exc:
                stats[f"gọi API lỗi: {type(exc).__name__}"] += 1
            if done % 10 == 0 or done == len(batches):
                print(f"  [{done}/{len(batches)}] đã có {len(cache)} câu", flush=True)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(
                    {"version": CACHE_VERSION, "items": cache},
                    ensure_ascii=False), encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"version": CACHE_VERSION, "items": cache},
        ensure_ascii=False), encoding="utf-8")

    # Xuất đủ mọi span ngoài taxonomy từ phản hồi API, kể cả khi
    # câu sau đó bị loại bởi một lỗi căn hàng khác.
    out_of_taxonomy = []
    for pair in pairs:
        cached = cache.get(pair["id"])
        for item in cached["spans"] if cached else []:
            if str(item.get("kieu", "")).strip().upper() == OUT_OF_TAXONOMY:
                out_of_taxonomy.append({
                    "written": str(item.get("viet", "")).strip(),
                    "spoken": str(item.get("noi", "")).strip().lower(),
                    "context": pair["written"],
                    "pair_id": pair["id"],
                })

    samples, rejected, gaps = [], [], []
    from ..registry import get_normalizer

    for pair in pairs:
        cached = cache.get(pair["id"])
        if cached is None:
            stats["API không trả lời"] += 1
            continue
        raw = cached["spans"]
        words, puncts = split_with_punct(pair["spoken"])
        placed, written_of = parse_spans(raw, words, puncts, stats)
        if placed is None:
            rejected.append({**pair, "lý_do": "span không khớp câu nói", "api": raw})
            continue

        got = comparison_form(reconstruct(words, placed, written_of, puncts))
        want = comparison_form(pair["written"])
        if got != want:
            stats["ghép lại KHÔNG ra văn bản gốc -> bỏ câu"] += 1
            rejected.append({**pair, "lý_do": "ghép lại không khớp",
                             "ghép_được": got, "cần": want, "api": raw})
            continue
        stats["ghép lại khớp văn bản gốc"] += 1

        organizations = [(start, end, written_of[start])
                         for start, end, kieu in placed
                         if kieu == OUT_OF_TAXONOMY]
        if organizations and not args.keep_organization:
            # Giữ câu nhưng bỏ span ORGANIZATION sẽ dạy nhãn O trên
            # "Quân tình nguyện Việt Nam" trong khi "Việt Nam" ở câu
            # khác lại là LOCATION_NAME. Đó chính là supervision mâu
            # thuẫn người dùng quan sát thấy. Taxonomy chưa có lớp này
            # thì phương án sạch duy nhất là loại cả câu.
            stats["bỏ câu: có ORGANIZATION ngoài taxonomy"] += 1
            rejected.append({**pair, "lý_do": "có ORGANIZATION ngoài taxonomy",
                             "organizations": organizations, "api": raw})
            continue

        # Đối chiếu với normalizer của ta -> danh sách lỗ hổng cần vá cho lúc suy luận
        keep = []
        strict_mismatches = []
        for start, end, kieu in placed:
            gold = written_of[start]
            spoken_chunk = " ".join(words[start:end + 1])
            if kieu == OUT_OF_TAXONOMY:
                if not args.keep_organization:
                    continue
                keep.append((start, end, kieu))
                continue
            keep.append((start, end, kieu))
            fn = get_normalizer(kieu)
            if fn is None:
                gaps.append({"pair_id": pair["id"], "kiểu": kieu,
                             "nói": spoken_chunk, "vàng": gold,
                             "ta": None, "lý_do": "chưa có normalizer"})
                if kieu in STRICT_SOURCE_TYPES:
                    strict_mismatches.append((spoken_chunk, kieu, None, gold))
                continue
            res = fn(spoken_chunk, kieu)
            ours = res.normalized if res.valid else None
            if ours is None or strip_marks(ours) != strip_marks(gold):
                gaps.append({"pair_id": pair["id"], "kiểu": kieu,
                             "nói": spoken_chunk, "vàng": gold,
                             "ta": ours, "lý_do": "khác đáp án" if ours else "không đọc được"})
            if kieu in STRICT_SOURCE_TYPES and \
                    surface_form(ours) != surface_form(gold):
                strict_mismatches.append((spoken_chunk, kieu, ours, gold))

        if strict_mismatches:
            stats["bỏ câu: normalizer entity không khớp casing gốc"] += 1
            rejected.append({**pair,
                             "lý_do": "normalizer entity không khớp casing gốc",
                             "mismatches": strict_mismatches, "api": raw})
            continue

        spoken_text = " ".join(words)
        # Đích huấn luyện là bộ nhãn span; phần dựng chuỗi là hậu xử lý tất định
        # của TA, nên `written` lấy theo quy ước của ta cho nhất quán ('06/1968'
        # chứ không phải 'Tháng 6 1968' như bài báo viết). Văn bản báo giữ ở
        # `written_source` để đo riêng. Nếu pipeline chưa dựng nổi thì vẫn giữ
        # câu — nhãn span đã được kiểm chứng bằng phép ghép lại rồi.
        try:
            rendered, outputs, _al, conflicts = normalize_with_gold_spans(
                spoken_text, keep, puncts)
            if conflicts or not all(o.emitted for o in outputs):
                raise ValueError("span chưa phát ra được")
        except Exception as exc:
            stats[f"bỏ câu: pipeline chưa dựng được: {type(exc).__name__}"] += 1
            rejected.append({**pair, "lý_do": "pipeline chưa dựng được",
                             "chi_tiết": str(exc), "api": raw})
            continue
        sample = Sample(
            id=hashlib.sha1(spoken_text.encode()).hexdigest()[:16],
            spoken=spoken_text,
            written=rendered,
            words=words,
            spans=keep,
            punct=puncts,
            types=sorted({t for _, _, t in keep}),
            leak_key="|".join(sorted(written_of[s] for s, _, _ in keep)),
            source=f"v1align:{pair['topic']}",
            written_source=pair["written"],
        )
        errors = validate_sample(sample)
        if errors:
            stats[f"mẫu hỏng: {errors[0]}"] += 1
            rejected.append({**pair, "lý_do": "; ".join(errors), "api": raw})
            continue
        samples.append(sample)

    out_path = ROOT / args.out
    write_jsonl(out_path, samples)
    for path, rows in ((args.rejects_out, rejected), (args.gaps_out, gaps),
                       (args.out_of_taxonomy_out, out_of_taxonomy)):
        p = ROOT / path
        with p.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    types = Counter(t for s in samples for t in s.types)
    print(f"\nghi {len(samples)} câu vào {args.out}")
    print(f"loại {len(rejected)} câu -> {args.rejects_out}")
    print(f"{len(gaps)} chỗ normalizer chưa khớp -> {args.gaps_out}")
    print(f"gọi API {client.calls} lần, {client.tokens} token, "
          f"chờ hạn mức {limiter.waited:.0f}s")
    print("\n--- phủ kiểu ---")
    for t, n in types.most_common():
        print(f"  {t:16s} {n}")
    print("\n--- thống kê ---")
    for why, n in stats.most_common(20):
        print(f"  {n:5d}  {why}")
    print("\n--- normalizer hụt nhiều nhất ---")
    for kieu, n in Counter(g["kiểu"] for g in gaps).most_common(12):
        print(f"  {n:5d}  {kieu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
