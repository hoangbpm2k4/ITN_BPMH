"""Mở rộng cách đọc trong danh mục bằng LLM — đề xuất bởi LLM, phán quyết bởi parser.

Vì sao cần: 630 dạng chuẩn trong sáu danh mục nhưng trung bình chỉ có 1,2-2,8
cách đọc mỗi cái. Đó là nút thắt thật của hệ thống. Mỗi mẻ dữ liệu mới lại lòi
ra một lối đọc chưa từng thấy — "sờ u", "mi gờ", "em ai gi", "dê ét u" — và ta
phát hiện chúng theo kiểu nhặt từng mẻ một.

Liệt kê cách người Việt phát âm một ký hiệu nước ngoài đúng là việc mô hình ngôn
ngữ làm giỏi. Nhưng nó cũng bịa rất giỏi, nên ở đây LLM CHỈ được quyền đề xuất;
quyền phán quyết thuộc về chính parser đang chạy trong pipeline. Bốn cổng:

  1. Hình thức — chỉ chữ thường có dấu, không chữ số, không dấu câu, 1-8 tiếng.
  2. Đụng độ — chạy qua danh mục HIỆN TẠI. Nếu đã ra đúng dạng chuẩn thì đề xuất
     là THỪA (bỏ, không làm danh mục phình). Nếu ra một dạng chuẩn KHÁC thì LOẠI:
     thêm vào sẽ tạo ra hai nghĩa cho cùng một cách đọc.
  3. Hợp lý — cổng thật sự, vì với danh mục tra bảng thì "thêm vào rồi tra lại"
     là vô nghĩa: thêm cái gì vào cũng tự khớp. Nên cổng này kiểm CẤU TRÚC của
     cách đọc so với dạng chuẩn, độc lập với danh mục:
       - dạng chuẩn CÓ chữ số (Su-22, AK-47): tách cách đọc thành phần chữ +
         phần số. Phần số phải đọc ra ĐÚNG chữ số của dạng chuẩn, và từ chỉ dấu
         gạch chỉ được nằm GIỮA hai phần. Cổng này bắt đúng thứ LLM hay sinh:
         "a ka bảy bốn gạch" — nội dung đúng nhưng dấu gạch bị gắn vào cuối.
       - dạng chuẩn KHÔNG chữ số: thử đọc như ĐÁNH VẦN tên chữ cái ("vi hát ép"
         -> VHF); không khớp thì mới đòi tương đồng ngữ âm đủ cao.
  4. Hồi quy toàn danh mục — sau khi thêm, MỌI cách đọc đã có của danh mục đó
     phải vẫn ra đúng dạng chuẩn cũ. Danh mục khớp gần đúng nên một mục mới có
     thể cướp kết quả của mục khác; không có cổng này thì mở rộng danh mục là
     làm hỏng nó. Trượt cổng này thì hoàn tác cả lô của dạng chuẩn đó.

Không có cổng nào ở đây tin vào LLM. Cái gì LLM bịa mà parser không xác nhận
được thì không vào tới danh mục.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .. import catalog as catalog_module
from difflib import SequenceMatcher

from ..catalog import (CATALOG_DIR, NUMBER_SPOKEN, get_catalog, load_raw,
                       phonetic_key)
from ..normalizers.number import read_number_auto
from ..registry import get_normalizer, reset_registry
from .gemma_client import GemmaClient, RateLimiter, load_keys

# Danh mục -> kiểu ngữ nghĩa dùng để nghiệm thu bằng chính normalizer.
CATALOG_TYPE = {
    "equipment": "EQUIPMENT_ID",
    "acronyms": "ACRONYM",
    "legal_docs": "LEGAL_DOC_ID",
    "foreign_names": "FOREIGN_NAME",
    "maritime_terms": "MARITIME_TERM",
}
# equipment.json để danh sách model dưới khoá "models".
ENTRY_KEY = {"equipment": "models"}

SPOKEN_RE = re.compile(r"^[a-zà-ỹ]+(?: [a-zà-ỹ]+){0,7}$")
MAX_SYLLABLES = 8
# Ngưỡng tương đồng ngữ âm cho dạng chuẩn không có chữ số. Đo trên các cách đọc
# thật: "pa tri ốt"/Patriot 1,00 · "hy mác"/Himars 0,73. Lối ĐÁNH VẦN tên chữ
# cái ("i xi đi ai ét"/ECDIS 0,29) đã được nhánh riêng bắt trước nên ngưỡng này
# giữ được chặt.
PHONETIC_MIN = 0.60
SEPARATOR_WORDS = {"gạch", "ngang", "nối", "gạch ngang", "gạch nối", "trừ",
                   "xẹt", "xẹc", "chéo", "dấu"}

PROMPT = """Bạn là chuyên gia phiên âm tiếng Việt cho hệ thống nhận dạng tiếng nói \
trong lĩnh vực quân sự và hàng hải.

Với mỗi ký hiệu dưới đây, hãy liệt kê những cách một người Việt ĐỌC THÀNH TIẾNG \
ký hiệu đó, viết lại bằng chữ thường có dấu.

Quy tắc bắt buộc:
- Chỉ chữ cái tiếng Việt và dấu cách. TUYỆT ĐỐI không có chữ số, không dấu câu, \
không dấu gạch.
- Chữ số phải viết thành chữ: 47 -> "bốn bảy" hoặc "bốn mươi bảy".
- Dấu gạch nối nếu có đọc thành tiếng thì viết "gạch ngang", "gạch nối" hoặc "gạch".
- Phủ nhiều lối: đọc tên chữ cái tiếng Anh ("ét u"), đọc tên chữ cái tiếng Việt \
("ét u khô"), đọc tắt kiểu Việt ("su"), đọc từng chữ số và đọc số nguyên, \
có đọc dấu gạch và không đọc dấu gạch.
- Chỉ đưa cách đọc mà người ta THẬT SỰ dùng. Không bịa.

Trả về JSON: một mảng các đối tượng {"canonical": "...", "readings": ["...", ...]}.
Mỗi ký hiệu tối đa %d cách đọc.

Các ký hiệu:
%s"""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "canonical": {"type": "STRING"},
            "readings": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["canonical", "readings"],
    },
}


def clear_caches():
    """Danh mục được nhớ đệm bằng lru_cache; đổi file thì phải xoá đệm."""
    load_raw.cache_clear()
    get_catalog.cache_clear()
    catalog_module.get_equipment_prefixes.cache_clear()
    # EquipmentIDParser dựng bảng model ngay trong __init__ nên phải dựng lại
    # cả registry, không chỉ xoá đệm danh mục.
    reset_registry()


def normalise_reading(text):
    text = unicodedata.normalize("NFC", str(text or "")).lower().strip()
    text = re.sub(r"[^\w\sà-ỹ]", " ", text)
    return " ".join(text.split())


def gate_form(reading):
    """Cổng 1: hình thức."""
    if not reading:
        return "rỗng"
    if any(ch.isdigit() for ch in reading):
        return "còn chữ số"
    if len(reading.split()) > MAX_SYLLABLES:
        return f"dài quá {MAX_SYLLABLES} tiếng"
    if not SPOKEN_RE.match(reading):
        return "có ký tự ngoài chữ cái tiếng Việt"
    return None


# Tên chữ cái TIẾNG ANH phiên âm bằng âm tiết Việt. Thuật ngữ hàng hải hầu hết
# được đánh vần theo lối này (ECDIS = "i xi đi ai ét", VHF = "vi hát ép"), mà
# bảng chữ cái sẵn có trong port_codes.json chỉ chứa tên chữ cái tiếng Việt.
# Một âm tiết có thể ứng với nhiều chữ ("a" = A hoặc R, "i" = E hoặc I) nên đây
# là quan hệ MỘT-NHIỀU; ta chỉ dùng nó để XÁC NHẬN một dạng chuẩn đã biết, không
# dùng để sinh, nên nhập nhằng không gây hại.
ENGLISH_LETTER_NAMES = {
    "ây": "A", "ê": "AE", "a": "AR", "bi": "B", "bê": "B", "xi": "C", "si": "C",
    "sê": "C", "đi": "D", "đê": "D", "i": "EI", "ép": "F", "ep": "F", "gi": "G",
    "giê": "G", "ết": "HS", "hát": "H", "ây chờ": "H", "ai": "I", "giay": "J",
    "giây": "J", "kây": "K", "ca": "K", "eo": "L", "en lờ": "L", "em": "M",
    "en": "N", "ô": "O", "âu": "O", "pi": "P", "pê": "P", "kiu": "Q",
    "quy": "Q", "rờ": "R", "e rờ": "R", "ơ": "R", "ét": "S", "ti": "T",
    "tê": "T", "diu": "U", "iu": "U", "u": "U", "vi": "V", "vê": "V",
    "đắp bờ liu": "W", "vê kép": "W", "ích": "X", "quai": "Y", "oai": "Y",
    "dét": "Z", "dét tờ": "Z",
}
MAX_SPELL_STATES = 4096


def spell_letters(reading, target):
    """Cách đọc này có phải là ĐÁNH VẦN của `target` không?

    Duyệt mọi cách gán âm tiết -> chữ cái (kể cả cụm hai tiếng như "e rờ") và
    trả True nếu có ít nhất một cách cho ra đúng `target`.
    """
    want = re.sub(r"[^A-Za-z]", "", target).upper()
    if not want:
        return False
    viet = {spoken: symbol.upper() for spoken, symbol in catalog_module.get_letters()}
    words = reading.split()
    # trạng thái = (vị trí trong words, số chữ đã khớp của want)
    states = {(0, 0)}
    for _ in range(len(words) + 1):
        nxt = set()
        for i, matched in states:
            if i >= len(words):
                continue
            for span in (2, 1):
                token = " ".join(words[i:i + span])
                if len(token.split()) != span:
                    continue
                options = set(ENGLISH_LETTER_NAMES.get(token, ""))
                if token in viet:
                    options |= set(viet[token])
                if want[matched:matched + 1] and want[matched] in options:
                    nxt.add((i + span, matched + 1))
        if not nxt:
            break
        states |= nxt
        if len(states) > MAX_SPELL_STATES:
            break
    return (len(words), len(want)) in states


def _digits_of(text):
    return "".join(ch for ch in text if ch.isdigit())


def gate_plausible(reading, canonical):
    """Cổng 3: cách đọc có hợp với dạng chuẩn về CẤU TRÚC không."""
    words = reading.split()
    if words[0] in SEPARATOR_WORDS or words[-1] in SEPARATOR_WORDS:
        return "từ chỉ dấu gạch nằm ở đầu hoặc cuối"

    want = _digits_of(canonical)
    if want:
        # Phần số phải là ĐUÔI của cách đọc.
        k = len(words)
        while k > 0 and words[k - 1] in NUMBER_SPOKEN:
            k -= 1
        number_words, head = words[k:], words[:k]
        if not number_words:
            return "dạng chuẩn có chữ số mà cách đọc không có phần số"
        if not head:
            return "thiếu phần chữ"
        try:
            got = _digits_of(str(read_number_auto(number_words)))
        except Exception:
            return "phần số không đọc được"
        digit_run = " ".join(number_words)
        if got != want and _digits_of(digit_run.replace(" ", "")) != want:
            # Đọc từng chữ số: "bốn bảy" = 47, khác read_number_auto.
            try:
                one_by_one = "".join(str(read_number_auto([w])) for w in number_words)
            except Exception:
                one_by_one = ""
            if one_by_one != want:
                return f"phần số đọc ra {got or one_by_one!r}, dạng chuẩn cần {want!r}"
        if any(w in SEPARATOR_WORDS for w in number_words):
            return "từ chỉ dấu gạch lẫn trong phần số"
        return None

    if spell_letters(reading, canonical):
        return None
    score = SequenceMatcher(None, phonetic_key(reading), phonetic_key(canonical)).ratio()
    if score < PHONETIC_MIN:
        return f"ngữ âm chỉ giống {score:.2f} < {PHONETIC_MIN}"
    return None


def resolve(type_name, reading):
    """Dạng chuẩn mà pipeline HIỆN TẠI cho ra, hoặc None nếu không đọc được."""
    normalizer = get_normalizer(type_name)
    if normalizer is None:
        return None
    result = normalizer(reading, type_name)
    return result.normalized if result.valid else None


def load_catalog_file(name):
    path = CATALOG_DIR / f"{name}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def existing_pairs(data, key):
    """[(cách đọc, dạng chuẩn)] của mọi mục đang có — dùng cho cổng hồi quy."""
    out = []
    for entry in data.get(key, []):
        for spoken in entry.get("spoken", []):
            out.append((normalise_reading(spoken), entry["canonical"]))
    return out


def regression_ok(type_name, pairs):
    """Cổng 4: mọi cách đọc cũ vẫn phải ra đúng dạng chuẩn cũ."""
    for reading, canonical in pairs:
        if resolve(type_name, reading) != canonical:
            return reading, canonical
    return None


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_catalog(name, client, args, stats):
    type_name = CATALOG_TYPE[name]
    key = ENTRY_KEY.get(name, "entries")
    path, data = load_catalog_file(name)
    entries = {e["canonical"]: e for e in data.get(key, [])}
    canonicals = sorted(entries)
    if args.limit:
        canonicals = canonicals[:args.limit]
    # Chỉ canh những cách đọc HIỆN ĐANG ĐÚNG. Nếu đưa cả cái đang hỏng sẵn vào
    # thì một mâu thuẫn có từ trước sẽ làm mọi lô đều bị hoàn tác — đúng như
    # "tìm kiếm cứu nạn" gán cho cả SAR lẫn TKCN đã chặn đứng lần chạy đầu.
    baseline = [(r, c) for r, c in existing_pairs(data, key)
                if resolve(type_name, r) == c]
    hong = len(existing_pairs(data, key)) - len(baseline)
    print(f"\n### {name}: {len(canonicals)} dạng chuẩn · canh {len(baseline)} cách đọc"
          + (f" · {hong} cách đọc VỐN ĐÃ hỏng, không canh" if hong else ""))

    accepted = Counter()
    for chunk in batches(canonicals, args.batch):
        if stats["gọi"] >= args.budget:
            print("  đã chạm trần số lượt gọi")
            break
        prompt = PROMPT % (args.per_canonical, "\n".join(f"- {c}" for c in chunk))
        try:
            raw = client.generate(prompt, temperature=args.temperature,
                                  max_tokens=2048, response_schema=SCHEMA)
        except Exception as exc:
            stats[f"lỗi gọi: {type(exc).__name__}"] += 1
            continue
        stats["gọi"] += 1
        try:
            proposals = json.loads(raw)
        except json.JSONDecodeError:
            stats["JSON hỏng"] += 1
            continue

        for item in proposals if isinstance(proposals, list) else []:
            canonical = str(item.get("canonical", "")).strip()
            entry = entries.get(canonical)
            if entry is None:
                stats["dạng chuẩn LLM bịa ra"] += 1
                continue
            added = []
            for reading in item.get("readings", []) or []:
                reading = normalise_reading(reading)
                stats["đề xuất"] += 1
                why = gate_form(reading)
                if why:
                    stats[f"cổng 1 hình thức: {why}"] += 1
                    continue
                current = resolve(type_name, reading)
                if current == canonical:
                    stats["cổng 2 thừa (đã đọc được)"] += 1
                    continue
                if current is not None:
                    stats["cổng 2 ĐỤNG ĐỘ dạng chuẩn khác"] += 1
                    continue
                why = gate_plausible(reading, canonical)
                if why:
                    stats[f"cổng 3 không hợp lý: {why.split(',')[0][:34]}"] += 1
                    continue
                if reading in set(entry.get("spoken", [])):
                    continue
                entry.setdefault("spoken", []).append(reading)
                added.append(reading)

            if not added:
                continue
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            clear_caches()

            kept = []
            for reading in added:
                if resolve(type_name, reading) == canonical:
                    kept.append(reading)
                else:
                    stats["cổng 3 thêm rồi vẫn không đọc được"] += 1
            entry["spoken"] = [s for s in entry["spoken"] if s not in added or s in kept]
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            clear_caches()

            broke = regression_ok(type_name, baseline) if kept else None
            if broke is not None:
                entry["spoken"] = [s for s in entry["spoken"] if s not in kept]
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
                clear_caches()
                stats["cổng 4 HOÀN TÁC vì làm hỏng mục cũ"] += len(kept)
                print(f"  hoàn tác {canonical}: {broke[0]!r} lẽ ra là {broke[1]}")
                continue

            for reading in kept:
                baseline.append((reading, canonical))
            accepted[canonical] += len(kept)
            stats["NHẬN"] += len(kept)

    total = sum(accepted.values())
    print(f"  nhận {total} cách đọc mới cho {len(accepted)} dạng chuẩn")
    for canonical, n in accepted.most_common(8):
        print(f"    +{n}  {canonical}")
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogs", default="equipment,acronyms,foreign_names,"
                                              "maritime_terms,legal_docs")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--batch", type=int, default=6,
                        help="số dạng chuẩn mỗi lượt gọi")
    parser.add_argument("--per-canonical", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=400,
                        help="trần số lượt gọi cho cả lần chạy")
    parser.add_argument("--limit", type=int, default=0,
                        help="chỉ xử lý N dạng chuẩn đầu (để thử)")
    parser.add_argument("--key-file", default="api.txt")
    args = parser.parse_args(argv)

    keys = load_keys(args.key_file if Path(args.key_file).exists() else None)
    client = GemmaClient(model=args.model, keys=keys, limiter=RateLimiter())
    print(f"{len(keys)} khoá · model {args.model} · trần {args.budget} lượt gọi")

    stats = Counter()
    for name in args.catalogs.split(","):
        name = name.strip()
        if name not in CATALOG_TYPE:
            print(f"bỏ qua danh mục không hỗ trợ: {name}")
            continue
        run_catalog(name, client, args, stats)

    print("\n=== TỔNG ===")
    for k, v in stats.most_common():
        print(f"  {v:6d}  {k}")
    print(f"\nlượt gọi API đã dùng: {client.calls} · token: {client.tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
