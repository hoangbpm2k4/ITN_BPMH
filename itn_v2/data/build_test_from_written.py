"""Dựng bộ đánh giá đầu-cuối từ các bộ test CHỈ CÓ DẠNG VIẾT.

`ITN_Test_20_Hoi_Thoai.zip` và `test_ITN_v2.zip` đánh dấu thực thể bằng dấu
nháy kép nhưng KHÔNG kèm dạng nói. Muốn đo đầu-cuối thì phải có dạng nói.

Sinh dạng nói bằng LLM chứ KHÔNG bằng verbalizer của kho này: verbalizer nội
bộ dùng đúng bộ cách đọc mà parser đang tra, nên chấm trên nó là tự chấm bài
mình — mọi cách đọc lạ sẽ biến mất khỏi bộ test đúng ở chỗ ta cần đo nhất.

Mỗi câu là một dòng đánh số; trả về theo số nên khớp cặp là tuyệt đối, không
phải dò bằng độ tương đồng như bộ data_v1.
"""

import argparse
import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from itn_v2.data.gemma_client import GemmaClient, RateLimiter

ROOT = Path(__file__).resolve().parents[2]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SPEAKER = re.compile(r"^[^:]{2,30}:\s*")

PROMPT = """Bạn là bộ nhận dạng tiếng nói tiếng Việt. Với MỖI dòng đánh số dưới
đây, hãy viết ra đúng những gì máy nhận dạng in ra khi có người ĐỌC TO câu đó.

Quy ước (bám đúng bản ghi thật của corpus tham chiếu):
- Mọi chữ số đọc thành chữ: "15/08/2026" -> "ngày mười lăm tháng tám năm hai
  nghìn không trăm hai mươi sáu"; "06:45" -> "sáu giờ bốn mươi lăm phút";
  "8,2 m" -> "tám phẩy hai mét"; "10°25′30″N" -> "mười độ hai mươi lăm phút
  ba mươi giây bắc"; "5.6.2" -> "năm chấm sáu chấm hai".
- TỪ TIẾNG ANH GIỮ NGUYÊN chữ Latinh, chỉ hạ chữ thường: "Chief Officer" ->
  "chief officer", "oil pressure" -> "oil pressure", "Bridge" -> "bridge".
  TUYỆT ĐỐI KHÔNG phiên âm ra tiếng Việt.
- Chữ viết tắt đọc thành tên chữ cái khi người Việt vẫn đọc vậy: "AIS" ->
  "ây ai ét", "VTS" -> "vê tê ét", "UTC" -> "u tê xê", "VHF" -> "vê hát ép".
- Đơn vị đọc thành lời: "kn" -> "hải lý trên giờ", "NM" -> "hải lý",
  "°C" -> "độ c", "V" -> "vôn", "%" -> "phần trăm".
- Bỏ hết dấu nháy kép, dấu chấm, dấu phẩy, chữ hoa. KHÔNG được còn chữ số nào.
- Giữ nguyên số dòng và thứ tự. Mỗi dòng ra một dòng, dạng "<số>|<lời đọc>".
- Không thêm giải thích.

{lines}"""


def clean_written(text):
    """Bỏ dấu nháy đánh dấu thực thể; chúng là markup, không phải dấu câu."""
    return re.sub(r"\s+", " ", text.replace('"', " ").replace("“", " ")
                  .replace("”", " ")).strip()


def sentences(text):
    out = []
    for block in text.splitlines():
        block = SPEAKER.sub("", block.strip())
        for part in SENT_SPLIT.split(block):
            part = clean_written(part)
            if len(part.split()) >= 3:
                out.append(part)
    return out


def parse_reply(reply, n):
    got = {}
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)\s*[|.)：:]\s*(.+)", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        spoken = m.group(2).strip().strip('"')
        if 1 <= idx <= n:
            got[idx] = spoken
    return got


def spoken_ok(text):
    """Chỉ nhận dạng nói THẬT: không chữ số, đủ dài."""
    if any(c.isdigit() for c in text):
        return False
    return len(text.split()) >= 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục .txt dạng viết")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="tên bộ, ghi vào cột chủ đề")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rpm", type=int, default=4)
    args = ap.parse_args(argv)

    files = sorted(Path(args.root).rglob("*.txt"))
    print(f"{len(files)} file dạng viết")

    done, rows = {}, []
    out_path = ROOT / args.out
    if args.resume and out_path.exists():
        # Hạn mức miễn phí đụng trần giữa chừng là chuyện thường; chạy lại
        # phải nối tiếp chứ không gọi lại từ đầu, nếu không thì mỗi lần chạy
        # lại là một lần đốt sạch hạn mức vào phần đã có.
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[(r["file"], r["sent"])] = r
        rows = list(done.values())
        print(f"nối tiếp: đã có {len(rows)} câu")

    jobs = []          # (tên file, chỉ số câu, câu viết)
    for path in files:
        for i, s in enumerate(sentences(path.read_text(encoding="utf-8",
                                                       errors="replace"))):
            if (path.stem, i) not in done:
                jobs.append((path.stem, i, s))
    print(f"{len(jobs)} câu viết còn phải đọc")

    client = GemmaClient(model=args.model, limiter=RateLimiter(rpm=args.rpm, tpm=900_000),
                         key_file=str(ROOT / "api.txt"))

    batches = [jobs[i:i + args.batch] for i in range(0, len(jobs), args.batch)]

    def run(batch):
        lines = "\n".join(f"{i+1}| {w}" for i, (_, _, w) in enumerate(batch))
        try:
            reply = client.generate(PROMPT.format(lines=lines), temperature=0.3,
                                    max_tokens=8192, thinking_level="low")
        except Exception as exc:
            print(f"  lỗi: {exc}")
            return []
        got = parse_reply(reply, len(batch))
        out = []
        for i, (stem, si, written) in enumerate(batch):
            spoken = got.get(i + 1, "")
            if spoken_ok(spoken):
                out.append({"topic": args.name, "file": stem, "sent": si,
                            "spoken": spoken, "written": written})
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k, part in enumerate(pool.map(run, batches), 1):
            rows.extend(part)
            print(f"  lô {k}/{len(batches)}: cộng dồn {len(rows)} câu")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nghi {len(rows)} câu -> {args.out}")
    print(f"lượt gọi {client.calls} · thử lại {client.retries}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
