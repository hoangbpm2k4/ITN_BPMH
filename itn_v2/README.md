# ITN V2 — trạng thái triển khai

Triển khai theo [README_ITN_V2_FINAL_UPDATED.md](../README_ITN_V2_FINAL_UPDATED.md).
Cập nhật 2026-09-09. **223 test, tất cả đều qua.** Đã train thật trên GPU.

```bash
cd /home/hoangbpm/inverse_text_norm
/home/hoangbpm/Intekcom/.venv_nemo/bin/python -m unittest discover -s itn_v2/tests -t .
```

Môi trường máy này thiếu `underthesea`, `torchcrf`, `pytest`. Đã xử lý mà không
cần cài thêm: tách từ dùng chính từ điển từ ghép của PhoBERT (21.405 từ), CRF tự
cài và đã đối chiếu với vét cạn, test dùng `unittest` chuẩn.

## Cập nhật casing và corpus 2026-09-09

Các bảng dữ liệu/huấn luyện cũ ở phía dưới là lịch sử thí nghiệm; artifact hiện
hành là bộ này.

Tìm thấy hai lỗi gốc: nhánh có biến thể phát âm giữ casing của bài báo trong
`spoken`, nhưng khi tạo `Sample.words` lại hạ chữ thường; và cổng ghép lại cũ hạ cả
hai chuỗi trước khi so sánh. Vì thế `Norng Chan Phal` có thể không được gán span
nhưng cổng vẫn báo đúng. Nay:

- mọi `spoken` đều hạ chữ thường ngay khi dựng cặp;
- cổng ghép lại bảo toàn casing, chỉ nới đúng chữ cái đầu câu;
- cache API có version + fingerprint của hai câu, không tái dùng phản hồi sinh trên
  input khác;
- span giống hệt nhau (`bà` -> `bà`) và casing thuần đầu câu (`bà` -> `Bà`) bị loại;
- câu có entity mà normalizer không dựng đúng casing nguồn, hoặc có span không
  phát ra được, bị loại thay vì fallback sang một `written` không thể đạt khi suy luận;
- taxonomy 46 lớp chưa có `ORGANIZATION`, nên 442 câu có lớp này bị loại khỏi
  train. Nếu xóa riêng span mà giữ câu, `Việt Nam` sẽ lúc mang `LOCATION_NAME`, lúc
  mang `O` bên trong tên tổ chức.

Kết quả audit cuối: 1.395 câu thật sạch; 78/78 lần `Việt Nam` và 9/9 lần
`Chan Phal` còn trong corpus đều được phủ span; không còn `bà` nào bị gán
`PERSON_NAME`/`RANK`. 927 span tổ chức được lưu riêng để quyết định taxonomy sau.

```text
1.395 câu thật + 524 câu seed = 1.919
  train                    1.427
  random_dev                 118
  entity_holdout_dev         164
  template_holdout_dev       210
  rò rỉ                         0
```

Checkpoint `checkpoints_v2/standardized_e15.pt` được train 15 epoch với
`--punct-weight 1.0`. Kết quả trên ba dev mới:

| Chỉ số | random | entity | template |
|---|---:|---:|---:|
| boundary F1 | 0,8868 | 0,9362 | 0,8385 |
| type F1 | 0,4465 | 0,4397 | 0,2484 |
| utterance exact | 0,0254 | 0,0305 | 0,1429 |
| punctuation macro-F1 bỏ O | 0,4246 | 0,6365 | 0,4472 |

CSV để duyệt tay là `datasets_v2/csv/v1_labeled.csv`; exporter nay mặc định đọc
`v1_aligned.jsonl`, đúng artifact được dùng để train.

---

## Đã xong

**Phase 1 — hợp đồng dữ liệu và gióng hàng (bước 1-7)**

| Thành phần | File |
|---|---|
| 46 kiểu + O, nhãn BIOES, mã hoá/giải mã span | `labels.py` |
| Tách từ tiếng Việt, ánh xạ ngược được về token ASR | `segmentation_alignment.py` |
| Gióng subword, gộp trung bình, không ghi đè position id | `tokenizer_alignment.py` |

**Phase 2 — chuẩn hoá tất định (bước 8-23)**

46/46 kiểu đều có bộ chuẩn hoá. Danh mục nằm ngoài mã nguồn ở `catalogs/*.json`.
Validator cho mọi lớp rủi ro cao. Oracle và hồi quy 33 nhóm đều qua.

**Phase 3 — mô hình (bước 24-31, trừ vòng huấn luyện)**

| Thành phần | File | Kiểm chứng |
|---|---|---|
| CRF ranh giới 5 trạng thái có ràng buộc | `crf.py`, `crf_constraints.py` | hàm phân hoạch, biên duyên, Viterbi đều khớp vét cạn |
| Đầu ra kiểu 47 chiều, gộp trung bình logit trên span | `model.py` | test hình dạng + tích hợp |
| Đầu ra dấu câu + focal loss | `model.py`, `losses.py` | |
| Độ tin cậy theo span từ biên duyên CRF | `confidence.py` | |

**Phase 4 — câu dài (bước 32-34)** — `windowing.py`, cửa sổ chồng lấn theo ranh
giới từ, ghép theo chỉ số từ gốc, ưu tiên dự đoán gần tâm cửa sổ.

**Phase 5 — chỉ số (bước 35-41, phần thuần hàm)** — `evaluate.py`, gồm cả
False Normalization Rate và Critical Utterance Exact Match.

---

**Dữ liệu và huấn luyện (§20-25, bước 31)** — có dữ liệu hạt giống đủ để chạy code.

| Thành phần | File |
|---|---|
| Lược đồ bản ghi, đọc/ghi JSONL | `data/schema.py` |
| Phân tích đánh dấu `[[nội dung\|KIỂU]]` | `data/markup.py` |
| Sinh dữ liệu bằng Gemma trên AI Studio | `data/generate_seed_data.py` |
| Lọc chất lượng | `data/clean_generated.py` |
| Chia tập chống rò rỉ | `data/build_splits.py` |
| Dataset + collate | `dataset.py` |
| Vòng huấn luyện, chọn checkpoint theo §27 | `train.py` |

### Dữ liệu hạt giống hiện có

Sinh bằng `gemma-4-31b-it`, 43 lượt gọi API. **Nguyên tắc: mô hình sinh lỏng,
pipeline tất định lọc chặt** — một câu chỉ được giữ khi mọi span trong đó chạy
lọt normalizer + validator + cổng ngưỡng. Nhờ vậy nhãn mô hình bịa tự bị loại.

```
844 câu qua được cổng tất định
 -> 540 câu sau bộ lọc chất lượng (64%)
      train                380 câu · 46/46 kiểu
      random_dev            44 câu
      entity_holdout_dev    59 câu
      template_holdout_dev  57 câu
      rò rỉ giữa các tập: 0
```

### Dữ liệu thật từ `data_v1/`

103 bài báo Quân đội / Hàng hải, 3.639 chunk, 1.654 biến thể phát âm. Đây là
**văn bản nghiệp vụ thật**, nhưng nhãn của pipeline Gemma v14 phía trên thì
không dùng được: `EQUIPMENT_CODE` (328 mục) phần lớn là từ tiếng Việt thường bị
đánh vần bậy (`"qua 80"` → `"Q U A tám mươi"`), 542/3633 mục không có nhãn, lẫn
cả rác do cắt chunk sai (`"000 tấn"` → `"không tấn"`).

`data/from_data_v1.py` chỉ lấy **cặp (dạng viết, dạng nói)**, căn hai chuỗi để
tìm vùng khác nhau, đoán kiểu từ HÌNH DẠNG của dạng viết, rồi **xác thực bằng
chính bộ chuẩn hoá**: chỉ giữ nhãn nào mà normalizer chạy trên dạng nói tái tạo
đúng dạng viết. Nhãn sai không thể lọt.

Kết quả **1.153 câu**: CARDINAL 770 · ACRONYM 288 · DATE 278.

Chính dữ liệu này chỉ ra hai lỗi và được sửa:

- **Năm trần ra sai.** `"một nghìn chín trăm linh sáu mươi tám"` → `1.968` thay
  vì `1968`, vì CardinalParser chèn dấu phân nhóm nghìn. `DateParser` nay đọc
  được năm đứng một mình → thêm 278 câu.
- **Viết tắt thật chưa có trong danh mục.** `data/harvest_acronyms.py` nạp
  LLVT, QĐND, CHQS, QNCN… từ dữ liệu thật vào `catalogs/acronyms.json`, đúng
  nguyên tắc "danh mục là tài sản dữ liệu, không phải danh sách tự suy đoán".
  ACRONYM từ 12 lên 288.

> Bộ thu hoạch lần chạy đầu nạp nhầm `XI ← "mười một"`, `XII ← "mười hai"` —
> đó là **số La Mã** trong "Đại hội XIII", không phải viết tắt. Nếu để nguyên
> thì mọi cụm "mười hai" trong văn bản sẽ biến thành `XII`. Bộ lọc nay loại số
> La Mã và chuỗi có chữ số (M41, Z133 là mã khí tài, không phải viết tắt).

Mỗi bản ghi giữ **hai dạng viết**: `written` là đầu ra của pipeline tất định
(đích huấn luyện nhất quán) và `written_source` là văn bản gốc của bài báo.
Đo trên `written` là đo vòng tròn; `written_source` mới cho biết khoảng cách thật.

### Kết quả huấn luyện thật

Train 15 epoch trên 1.127 câu (1.153 thật + 524 sinh máy), RTX 3060, vài phút.

| Chỉ số | random_dev | entity_holdout | template_holdout |
|---|---|---|---|
| boundary strict F1 | 0,888 | 0,903 | 0,906 |
| type strict F1 | 0,747 | 0,780 | 0,722 |
| dấu câu macro-F1 (bỏ O) | 0,395 | 0,393 | 0,387 |
| **utterance exact** | **0,082** | **0,093** | **0,046** |
| utterance exact *nếu bỏ cổng ngưỡng* | 0,117 | 0,172 | 0,103 |
| khớp **văn bản gốc** của bài báo | 0,016 | 0,042 | 0,017 |
| tỷ lệ chuyển trạng thái BIOES sai | 0 | 0 | 0 |

Chỉ dùng dữ liệu sinh máy (367 câu) thì type F1 là 0,49-0,64; thêm dữ liệu thật
đưa lên **0,72-0,78**. Dữ liệu thật là đòn bẩy, không phải kiến trúc.

**Khoảng cách so với văn bản gốc phân rã ra sao** (đếm trên random_dev):

| Loại khác biệt | Số lượng | Tỷ lệ |
|---|---|---|
| chỉ khác hoa-thường | 325 | 43% |
| chỉ khác dấu phẩy | 217 | 29% |
| khác nội dung | 154 | 20% |
| số chưa chuẩn hoá | 60 | 8% |

### Gán nhãn từ CÂU GỐC bằng Gemini

`data/annotate_with_gemini.py`. Cách trên (`from_data_v1.py`) đi ngược từ bản
augment của pipeline v14 — mà bản đó đã hỏng sẵn — nên phải đoán kiểu bằng luật
hình dạng và chỉ vớt được 3 kiểu. Ở đây quay về nguồn: đưa **câu gốc dạng viết**
cho Gemini, yêu cầu chỉ ra từng cụm kèm kiểu và cách đọc, dùng JSON có lược đồ.

Cổng xác thực giữ nguyên: chỉ nhận span nào mà normalizer chạy trên cách đọc
tái tạo ĐÚNG dạng viết trong câu gốc. Gemini gán sai vẫn bị loại.

```
3.087 câu gốc -> 124 lượt gọi -> 1.459 câu, 11 kiểu
   PERSON_NAME 702 · LOCATION_NAME 646 · CARDINAL 482 · DATE 146
   ACRONYM 54 · RANK 85 · ORDINAL 4 · DIGIT_SEQ 3 · ...
```

**Chọn model theo hạn mức, không theo chất lượng thuần tuý.**
`gemini-2.5-flash` cho nhãn tốt nhất nhưng chỉ **20 request/ngày** — không dùng
được ở quy mô này. `gemini-3.5-flash-lite` (15 RPM, 500 RPD) kèm **gộp 25 câu
mỗi lượt gọi** thì toàn bộ 3.087 câu chỉ tốn 124 lượt.

**Khoảng trống taxonomy đã có bằng chứng số.** Gemini gán `ORGANIZATION` cho
**691 cụm, 366 tên tổ chức khác nhau** — "Quân chủng Hải quân", "Bộ Quốc phòng",
"Quân đội nhân dân Việt Nam", "Đài Truyền hình Việt Nam". Taxonomy 46 kiểu không
có lớp nào cho chúng nên chúng không bao giờ được viết hoa đúng. Đã lưu vào
`datasets_v2/out_of_taxonomy.jsonl`, KHÔNG đưa vào huấn luyện (§5 đóng băng
taxonomy). Đây là quyết định của người sở hữu spec.

**Tác động, đo cùng một cách trên khoảng cách so với văn bản gốc:**

| Loại khác biệt | nhãn hình dạng (3 kiểu) | nhãn Gemini (11 kiểu) |
|---|---|---|
| chỉ khác hoa-thường | 325 (43%) | **167 (26%)** |
| chỉ khác dấu phẩy | 217 (29%) | 319 (50%) |
| khác nội dung | 154 (20%) | 83 (13%) |
| số chưa chuẩn hoá | 60 (8%) | 72 (11%) |

Lỗi viết hoa giảm gần một nửa nhờ có nhãn PERSON_NAME / LOCATION_NAME / RANK.
Nút thắt còn lại là **dấu phẩy**: precision 0,47 — cứ hai dấu phẩy mô hình đặt
thì một cái sai. Đây là bài toán vừa khó vừa mang tính văn phong.

> Lưu ý khi so sánh: hai lần chạy dùng tập dev khác nhau (dữ liệu đổi thì phép
> chia đổi theo), nên không so trực tiếp boundary/type F1 giữa hai cột được.
> Bảng trên thì so được vì đo cùng một cách trên cùng loại văn bản gốc.

### Ablation: trọng số nhánh dấu câu

Spec §8 cố định `0.30 * L_PUNCT`. Trên dữ liệu này con số đó chặn mất phần lớn
dấu phẩy. Cùng cấu hình, chỉ đổi trọng số:

| Chỉ số (random_dev) | 0,30 (spec) | 1,00 |
|---|---|---|
| COMMA recall | 0,18 | **0,76** |
| COMMA F1 | 0,27 | **0,65** |
| PERIOD F1 | 0,91 | **0,98** |
| dấu câu macro-F1 (bỏ O) | 0,395 | **0,546** |
| boundary strict F1 | 0,888 | **0,917** |
| type strict F1 | 0,747 | **0,772** |
| utterance exact | 0,082 | **0,129** |
| khớp văn bản gốc | 0,016 | **0,040** |

**Không một chỉ số nào tụt** — kể cả ranh giới và kiểu cũng tốt lên, có vẻ do
lợi ích đa nhiệm. Mặc định trong `config.py` vẫn để 0,30 đúng spec; bật 1,00
bằng cờ `--punct-weight 1.0`. Đây là quyết định của người sở hữu spec, không
phải của tôi.

`QUESTION` F1 bằng 0 ở mọi cấu hình vì dữ liệu gần như không có câu hỏi:
`data_v1` có 3.510 dấu phẩy, 1.202 dấu chấm và **không một dấu hỏi nào**.

**72% khoảng cách không phải do ITN.** Viết hoa tên riêng và tổ chức
("Trường Cao đẳng Y tế Sơn La", "Nguyễn Quốc Trị") không được gán nhãn vì
pipeline thượng nguồn chỉ chú thích thay đổi cách đọc, không chú thích viết hoa
— và taxonomy 46 kiểu đã đóng băng thì không có lớp cho tên tổ chức. Dấu phẩy
thì COMMA recall chỉ 0,15-0,18.

Ba điều đọc được từ bảng này:

1. **Ranh giới đã học được** (F1 0,90-0,96) và giữ nguyên trên cả hai tập
   holdout. CRF có ràng buộc chưa từng sinh ra một chuỗi BIOES sai cấu trúc nào.
2. **Kiểu chưa học được** (F1 0,49-0,64). 367 câu chia cho 46 lớp là quá ít;
   các lớp phân biệt được bằng dấu hiệu bề mặt (ELECTRONIC, IMO_ID, MEASURE,
   VERSION) đạt gần tuyệt đối, còn các lớp phải dựa vào ngữ cảnh — CARDINAL với
   DIGIT_SEQ với HEADING với MMSI_ID đều là dãy chữ số — thì hỏng.
3. **Khoảng cách 0,10 so với 0,63 chính là cổng tin cậy đang làm việc.**
   Span chuẩn hoá đúng nhưng mô hình chỉ tự tin 0,07-0,24 nên hệ thống giữ dạng
   nói thay vì phát ra giá trị có thể sai. Đây là hành vi ĐÚNG theo spec §16.
   Ngưỡng hiện tại là giá trị tạm; §9.4 yêu cầu tinh chỉnh trên `real_dev`.

**Báo cáo dấu câu phải bỏ lớp O.** Accuracy thô 0,986 trông rất đẹp nhưng vô
nghĩa vì 94% số từ mang nhãn "không có dấu câu" — đúng cái bẫy V1 mắc phải.
Macro-F1 bỏ O cho thấy sự thật: PERIOD 0,93 nhưng QUESTION 0,00.

`real_dev` và `real_test` để TRỐNG có chủ đích — hai tập này phải là dữ liệu
thật do người cung cấp, không được sinh máy (spec §22).

Lý do bị loại nhiều nhất, theo đúng thứ tự: mô hình viết chữ số thay vì chữ
(377), bịa kiểu ngoài taxonomy (45), bịa số hiệu khí tài không có trong danh
mục (25), sai số chữ số IMO (validator bắt). Đây là bằng chứng cổng chất lượng
đang làm việc, không phải lỗi.

```bash
export GOOGLE_API_KEY=...              # không bao giờ ghi key vào kho mã
python -m itn_v2.data.generate_seed_data --rounds 5 --per-prompt 14 --workers 5
python -m itn_v2.data.clean_generated
python -m itn_v2.data.build_splits --input datasets_v2/seed_clean.jsonl
python -m itn_v2.train --smoke --device cpu   # kiểm tra đường chạy
python -m itn_v2.train                        # huấn luyện thật
```

## Chưa xong

| Việc | Bước trong spec | Vì sao chưa làm |
|---|---|---|
| Trộn 30% sạch / 20% biến thể / 50% lỗi ASR | §20 | cần transcript ASR thật |
| `real_dev`, `real_test` | §22 | phải là dữ liệu thật, không sinh máy |
| Huấn luyện đầy đủ + tinh chỉnh ngưỡng | §24, §9.4 | cần dữ liệu thật để có ý nghĩa |
| Nhật ký chẩn đoán sản xuất | Phase 6 | cần hệ thống chạy thật |

---

## Đối chiếu checklist §35

Đạt **28/35**. Bảy mục còn lại đều thuộc nhóm cần dữ liệu ở trên.

| ✔ | Mục |
|---|---|
| ✔ | Đầu vào PhoBERT đã tách từ tiếng Việt |
| ✔ | Ánh xạ ngược về token ASR gốc, có test |
| ✔ | Dùng vị trí subword gốc của PhoBERT |
| ✔ | Biểu diễn từ bằng gộp trung bình subword |
| ✔ | Ranh giới dùng CRF BIOES 5 trạng thái |
| ✔ | Kiểu là đầu ra riêng 46 + O |
| ✔ | Dấu câu độc lập với ranh giới ITN |
| ✔ | Cả 33 nhóm test cũ đều ánh xạ vào taxonomy |
| ✔ | MONEY · MEASURE · TIMEZONE · TELEPHONE · ELECTRONIC · VERSION · ADDRESS · VEHICLE_PLATE |
| ✔ | COORD có bộ chuẩn hoá riêng |
| ✔ | HEADING giữ số 0 đầu |
| ✔ | MMSI giữ nguyên dãy chữ số |
| ✔ | EQUIPMENT_ID hỗ trợ khuôn chữ-số hỗn hợp |
| ✔ | Giữ hoa-thường chuẩn Su / MiG / Kh |
| ✔ | Số âm là modifier, không phải lớp riêng |
| ✔ | EMAIL / URL / IP / MAC hoạt động |
| ✔ | Tra danh mục có ngưỡng và đường lui |
| ✔ | Có biên duyên CRF |
| ✔ | Độ tin cậy span được định nghĩa tường minh |
| ✔ | Ngưỡng theo từng lớp, cấu hình được |
| ✔ | Lớp rủi ro cao có validator |
| ✔ | Chuẩn hoá thất bại thì trả về dạng nói |
| ✔ | Regex toàn cục không ghi đè được span đã chuẩn hoá |
| ✔ | Câu dài dùng cửa sổ chồng lấn |
| ✔ | Test oracle qua hết |
| ✔ | 33 nhóm đều có test hồi quy |
| ✔ | Chỉ số exact-match theo từng lớp |
| ✔ | Có False Normalization Rate |
| ✔ | Suy luận trả vết gỡ lỗi có cấu trúc |
| ✘ | Biến thể cùng nguồn không rò rỉ giữa các tập |
| ✘ | Có `entity_holdout_dev` |
| ✘ | Có `template_holdout_dev` |
| ✘ | `real_test` được đóng băng |

---

## Sáu lỗi V1 mà spec §29 yêu cầu phải biến mất

Tất cả đều đã có test hồi quy trong `tests/test_oracle_pipeline.py`.

| Lỗi V1 | V1 | V2 |
|---|---|---|
| DATE mất ngày khi từ khoá nằm ngoài span | `hai mươi tám tháng bốn` → `04` | → `28/04` |
| Nhãn COORD không điều khiển chuyển đổi | có nhãn hay không đều ra `19°20'N` | không nhãn thì giữ nguyên dạng nói |
| Quân hàm bị luật dọn hạ chữ | `Đại tá` → `đại tá` | `Đại tá` được bảo vệ |
| Khí tài mất hoa-thường chuẩn | `Su-30` → `SU-30` | `Su-30`, `MiG-21`, `Kh-35` |
| Dấu câu quyết định ranh giới span | hai cụm DATE liền nhau bị dính | BIOES tách được, không cần dấu câu |
| Cắt cứng làm mất đuôi câu | vứt phần quá 256 subword | cửa sổ chồng lấn, phủ đủ mọi từ |

Ngoài ra V2 sửa thêm ba lỗi phát hiện trong quá trình triển khai:

- **Số điện thoại đọc theo nhóm bị phá huỷ.** `không tám tám hai mươi mốt ba bốn năm sáu`
  ra `6` ở V1 (mất 8/9 chữ số). Nguyên nhân: bộ số V1 tự đoán chế độ đọc bằng
  heuristic từ khoá. V2 để **kiểu ngữ nghĩa quyết định chế độ đọc**, nên cả hai
  cách đọc đều ra `088213456`.
- **`không trăm` bị tính thành 100.** `một nghìn không trăm hai mươi` ra `1.120`
  ở V1, đúng phải là `1.020`.
- **Không viết hoa sau dấu chấm giữa đoạn.** V2 chạy bước viết hoa sau khi khôi
  phục dấu câu, đúng như spec §17 quy tắc 4.

---

## Điểm cần quyết định

Spec không quy định định dạng cho vài trường hợp, tôi đã chọn mặc định sau và
đánh dấu ở đây để dễ đổi:

| Trường hợp | Đang chọn |
|---|---|
| `RANGE` | `10-20` (gạch nối) |
| `DURATION` | `2 giờ 30 phút` (giữ từ tiếng Việt) |
| `CHANNEL` | `kênh 16` nếu span có từ "kênh", không thì chỉ `16` |
| Khoảng trắng trước đơn vị | có, trừ đơn vị bắt đầu bằng `°` → `24 V` nhưng `-10°C` |
| Phân nhóm hàng nghìn | dấu chấm, và **không áp dụng** cho phần nguyên đọc rời (`121,5` chứ không `1.215`) |
