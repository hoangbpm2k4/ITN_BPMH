# ITN V2 — trạng thái, chẩn đoán và hướng đi

Cập nhật 2026-09-09. Tài liệu này ghi lại **toàn bộ** những gì đã đo được: kiến
trúc hiện tại, dữ liệu, các lỗi đã tìm ra và sửa, cách đánh giá, và việc cần làm
tiếp — kèm bằng chứng số cho từng kết luận.

Triển khai theo [README_ITN_V2_FINAL_UPDATED.md](README_ITN_V2_FINAL_UPDATED.md).
Tài liệu triển khai chi tiết ở [itn_v2/README.md](itn_v2/README.md).

Đối chiếu với hệ cũ V1 và đánh giá gói dữ liệu v4 ở [README_BASELINE_V1_VA_DANH_GIA_V4.md](README_BASELINE_V1_VA_DANH_GIA_V4.md).

---

## 1. Kết quả hiện tại

### Trên tập test thật `data_v1/data` — thước đo chính

728 câu, 33 chủ đề, do người viết, chưa từng đưa vào huấn luyện.

| checkpoint | khớp hoàn toàn | F1 theo từ | span phát ra | câu bị ngưỡng chặn |
|---|---:|---:|---:|---:|
| `standardized_e15` (trước khi có bundle) | 12,4% | 0,727 | **0** | 243 |
| `mil_clean` (+ bundle v2) | 25,3% | 0,792 | 330 | 126 |
| **`mil_v3`** (+ bundle v2 + v3) | **25,4%** | **0,804** | **490** | **83** |

### Trên dev tin tức thật (trích từ `data_v1`)

| chỉ số | random (115) | entity (157) | template (206) |
|---|---:|---:|---:|
| boundary_f1 | 0,869 | 0,949 | 0,881 |
| type_f1 | 0,831 | 0,929 | 0,817 |
| utterance_exact | 0,652 | 0,586 | 0,660 |
| khớp văn bản gốc | 0,486 | 0,434 | 0,582 |
| dấu câu macro-F1 (bỏ O) | 0,949 | 0,596 | 0,830 |

**37/42 kiểu có span dev đạt F1 ≥ 0,50** (trước bundle: 1/42).

### Trên test tổng hợp — KHÔNG dùng làm thước đo

`mil_v2_te` type_f1 0,966 · `mil_v3_te` type_f1 0,999. Cao vì dữ liệu sinh theo
khuôn. Chỉ dùng để đo mức lệch miền, xem §6.

---

## 2. Kiến trúc — không phải chỗ cần sửa

Backbone `vinai/phobert-base-v2` theo đúng spec §2. Không có số liệu nào chỉ vào
kiến trúc: `boundary_f1` đã 0,87–0,95 và `type_f1` 0,83–0,93 trên dev thật. Mô
hình **nhận diện tốt**; chỗ mất điểm nằm ở dữ liệu, ở bộ chuẩn hoá tất định và ở
cổng ngưỡng phía sau.

Hai trục tách rời (spec §6): biên do CRF ràng buộc BIOES lo, kiểu do softmax 47
lớp lo. Không sinh văn bản tự hồi quy (spec §33).

---

## 3. Dữ liệu

### 3.1 Corpus hiện tại

| nguồn | số câu | vai trò |
|---|---:|---|
| tin tức thật (`data_v1`, đã căn hàng) | 1.405 | phân bố đích |
| bundle quân sự v2 | 5.615 | cân bằng lớp |
| bundle quân sự v3 | 9.750 | lấp 13 kiểu đói + câu tương phản |
| **train** (tin tức nhân 3 + tổng hợp) | **19.580** | |
| random_dev / entity_dev / template_dev | 115 / 157 / 206 | dev thật |
| `dv1_eval` (`data_v1/data`) | 728 | **test thật** |

Tin tức thật được nhân 3 để không bị chìm giữa 15.365 câu tổng hợp — 22% tỉ
trọng mà không vứt dữ liệu nào.

### 3.2 Đường đi của nhãn — hai thế hệ

**Thế hệ 1 (bỏ):** căn hàng bằng phép so chuỗi để tìm vùng khác nhau, API chỉ
gán kiểu, rồi bắt normalizer của ta tái tạo đúng dạng viết của bài báo mới nhận.

Cách này hỏng vì bị chặn bởi **độ phủ của code**: cụm nào normalizer chưa đọc
được thì bị loại, mà câu vẫn ở lại tập huấn luyện với cụm đó gán `O` — tức dạy
mô hình điều ngược lại. Đo được: **239 cụm `DATE` bị loại âm thầm** theo cách
này.

**Thế hệ 2 (đang dùng):** văn bản ITN của bài báo đã là đáp án, nên việc còn lại
chỉ là **căn hàng**. Cho API cả câu viết lẫn câu nói cùng 46 kiểu, bảo nó liệt kê
từng span gồm (chuỗi nói, chuỗi viết, kiểu). Không cần normalizer nào lúc gán
nhãn.

Cổng kiểm chứng vẫn tất định và chặt hơn: **ghép các span trở lại câu nói phải
tái tạo đúng văn bản bài báo**. Không khớp thì bỏ cả câu, không thể lặng lẽ thành
mẫu âm. Kết quả: 2.480/2.559 câu ghép lại khớp.

Mã: [itn_v2/data/label_pairs_llm_align.py](itn_v2/data/label_pairs_llm_align.py)

### 3.3 Ba bộ lọc tất định bắt buộc

Đặt trong bộ gán nhãn, không phải bước hậu kiểm:

1. **Kiểu bắt buộc sinh chữ số** mà dạng viết không có chữ số nào → bỏ span.
   Bắt được 415 span: `rồi → Rồi` gán `CARDINAL`, `tác → Tác` gán `HEADING`.
2. **Viết hoa đầu câu** bị gán thành tên riêng → bỏ span. Dùng chính corpus làm
   bằng chứng: từ nào giữa câu luôn viết thường thì không phải danh từ riêng.
   Bắt được 134 span: `bà → Bà` gán `PERSON_NAME`.
3. **Span vô nghĩa** — normalizer trả về chuỗi y hệt đầu vào → bỏ cả câu.
   Xem §5.1, đây là lỗi tốn kém nhất.

Mã: [itn_v2/data/drop_identity_spans.py](itn_v2/data/drop_identity_spans.py)

---

## 4. Đánh giá bundle tổng hợp

Ba mẻ do ChatGPT sinh. Mỗi mẻ đều được kiểm chứng độc lập, không tin changelog.

| tiêu chí | v1 | v2 | v3 |
|---|---|---|---|
| số câu | 3.360 | 8.340 | 15.960 |
| số span | 4.478 | 17.342 | 33.328 |
| giá trị khác nhau mỗi kiểu | **16** | 95–120 | 90–130 |
| rò rỉ thực thể dev/test | **100%** | 0% | 0% |
| rò rỉ câu nền | **62%** | 0% | 0% |
| span vô nghĩa (ra ≡ vào) | 36 | **0** | **0** |
| `FOREIGN_NAME` dịch sang tiếng Anh | **9/16** | 0 | 0 |
| lỗi BIO / lệch token | 0 | 0 | 0 |
| độ phủ normalizer của ta | 84% | 98,9% | 90,3% |

**v1 không dùng được**: `pháp → France`, `thái lan → Thailand` là *dịch nghĩa*
chứ không phải chuẩn hoá; văn bản tiếng Việt viết "Pháp", "Thái Lan". Cộng thêm
100% thực thể dev/test có trong train nên điểm đo được là trí nhớ.

**v3 nhắm đúng ba nhóm đã chỉ ra**: 13 kiểu đói dữ liệu (từ dưới 20 lên hơn 1.200
mẫu mỗi kiểu), 4 kiểu chưa có dev, và **4.750 câu tương phản** — trong đó
4.750/4.750 thực sự chứa cả hai lớp gây lẫn trong cùng một câu:

```
sau khi nhắc tên mai đức tùng, nhân viên radar mới chuyển sang thông tin về ireland
   PERSON_NAME  +  FOREIGN_NAME  cùng một câu
```

Lỗi còn lại của v3: 1.685 câu (10,6%) để `MMSI`/`AIS` nguyên chữ hoa trong câu
nền — ASR thật không xuất ra như vậy, và nó thành **đường tắt** vì "MMSI" luôn
đứng trước span `MMSI_ID`. Đã hạ chữ thường khi chuyển đổi.

---

## 5. Các lỗi đã tìm ra

### 5.1 Họ lỗi "âm thầm nuốt chữ" — nguy hiểm nhất

Bộ đọc số trả về kết quả **hợp lệ nhưng sai** thay vì báo lỗi, nên không có gì
phát hiện được. Tất cả cùng một cơ chế.

| nơi | ví dụ | trước | sau |
|---|---|---|---|
| `DateParser` thiếu dạng *tháng + năm* | `tháng sáu một nghìn chín trăm sáu mươi tám` | HỎNG | `06/1968` |
| `DateParser` tưởng "năm" trong *năm mươi* là mốc năm | `một nghìn chín trăm năm mươi ba` | HỎNG | `1953` |
| `read_year` bỏ qua phần ngày-tháng | `tám chín một nghìn chín trăm sáu mươi chín` | `1969` | `08/09/1969` |
| `read_number_auto` bỏ từ thừa | `sáu mười một một nghìn…` | `10/01/1978` | `06/11/1978` |
| đọc tham lam một cách duy nhất | `ngày ba mươi tám…` (38 hay 30+8?) | `1984` | `30/08/1984` |
| `EquipmentIDParser` đứt tại "trăm" | `ét ba trăm` | HỎNG | `S-300` |
| `ElectronicParser` chỉ đọc rời chữ số | `mười tám chấm một trăm chín mươi sáu…` | HỎNG | `18.196.58.20` |
| `AddressParser` đọc số theo cụm rời | `số một trăm năm mươi tám đường trường chinh` | `Số 1 trăm 58 …` | `số 158 đường Trường Chinh` |

Số cụm `DATE` đọc sai: **107 → 34**.

Cách chữa chung: bộ đọc phải **báo đã tiêu thụ bao nhiêu từ**, và khi có nhiều
cách đọc thì thử tất cả rồi chọn cách ghép được với phần còn lại — thay vì đọc
tham lam một lần.

### 5.2 Lỗi hạ tầng

- **`Catalog` chỉ đọc khoá `entries`** mà `equipment.json` để dữ liệu dưới khoá
  `models` → toàn bộ 223 model **vô hình** với `EquipmentNameResolver`. Đó là lý
  do `EQUIPMENT_NAME` đứng yên ở 0/746 dù đã nạp catalog.
- **`lẻ` thiếu trong `UNIT_DIGITS`** dù `linh` (đồng nghĩa hoàn toàn) có sẵn →
  `hai trăm lẻ chín` ra `200le9`.
- **Bảng chữ cái thiếu 14 âm đọc thông tục**: `gờ` (G), `nờ` (N), `mờ` (M),
  `o` (O), `e rờ` (R)… → `e rờ ba tám` ra `ER38` thay vì `R38`, và `gờ o vê`
  (gov) không đọc được.
- **`SPEED` đóng cứng mẫu số `/h`** → `chín mét trên giây` không ra `9 m/s`.
- **`hai chấm` luôn hiểu là dấu `:`** → `mười chấm một chấm hai chấm ba` ra
  `10.1.:3`. Nay thử cả hai cách rồi lấy cách cho ra địa chỉ hợp lệ.

### 5.3 Lỗi tự che giấu — bài học đắt nhất

**`ElectronicParser` có đường tắt**: nếu đầu vào đã là URL thì trả lại y nguyên
và báo hợp lệ. Cổng kiểm chứng so `normalizer(nói) == viết`, nên
`www.vinamarine.gov.vn → www.vinamarine.gov.vn` **luôn khớp**.

Hệ quả dây chuyền: dữ liệu seed sinh bằng Gemma không đọc URL thành chữ → 37
span lọt cổng → nằm trong cả train lẫn dev → `ELECTRONIC` báo **F1 0,929** trong
khi **14/15 span dev là loại đầu vào đã chứa đáp án**. Con số đó hoàn toàn ảo.

Sau khi dọn: `ELECTRONIC` còn **1 span dev, F1 0,00**.

**Và việc dọn 36 câu rác (0,5% tập train) làm MỌI chỉ số tăng:**

| | chưa dọn | đã dọn |
|---|---:|---:|
| random type_f1 | 0,8153 | **0,8516** |
| entity boundary_f1 | 0,9349 | **0,9557** |
| test khớp hoàn toàn | 22,9% | **25,3%** |

Dữ liệu mà **đầu vào đã chứa đáp án không trung tính** — nó dạy mô hình chép
lại, và thói quen đó lan sang cả chỗ cần chuẩn hoá thật.

### 5.4 Lỗi trong nhãn vàng còn tồn đọng

- **43 câu bị xoá mất từ dẫn ngày tháng**: span `DATE` nuốt cả chữ "tháng" rồi
  `DateParser` bỏ đi. `đến tháng 4` thành `đến 04` — câu tiếng Việt hỏng, mà nằm
  trong nhãn vàng nên đang **dạy** mô hình xuất như vậy.
- **`EQUIPMENT_NAME` trong dev thật có nhãn sai**: `qua sinh tơn → Washington`
  bị gán `EQUIPMENT_NAME` thay vì `FOREIGN_NAME`. Cần sửa nhãn trước khi kết
  luận F1 0,00 của lớp này.

---

## 6. Cách đánh giá — và cái bẫy phải tránh

### Ba thước đo, độ tin cậy giảm dần

1. **`data_v1/data`** (728 câu) — người viết, không dùng nhãn span nào của ta,
   chạy pipeline trên dạng nói rồi so thẳng với văn bản gốc. **Đây là thước đo
   duy nhất đáng tin để kết luận.**
2. **dev tin tức thật** (115/157/206 câu) — nhãn do API căn hàng, đã kiểm chứng
   bằng phép ghép lại. Dùng để chẩn đoán theo lớp.
3. **dev/test tổng hợp** — chỉ dùng để **đo mức lệch miền**, không bao giờ làm
   con số công bố.

### Cái bẫy: F1 tổng hợp ≈ 1,00 nhưng F1 thật = 0,00

Cột `f1_dev_tong_hop` trong báo cáo theo kiểu gần như 1,00 ở mọi lớp. Khoảng
cách giữa nó và F1 thật chính là mức mô hình học **câu nền** làm đường tắt thay
vì học đặc trưng của cụm.

Ví dụ rõ nhất — `VEHICLE_PLATE`:

| nguồn | dạng nói |
|---|---|
| dữ liệu tổng hợp (877 mẫu) | `một bốn ca bốn không tám tám chín` |
| dữ liệu thật (test) | `một năm xê **gạch ngang** một hai ba **chấm** bốn năm` |

Người Việt đọc biển `15C-123.45` có cả "gạch ngang" và "chấm". Dữ liệu tổng hợp
bỏ hết dấu phân cách, tức dạy 877 mẫu của một dạng **không tồn tại trong thực
tế**. F1 dev tổng hợp 1,00; trên test mô hình **không tìm ra span nào**.

### Lưu ý về tính trinh nguyên của test

`data_v1/data` đã được dùng một lần để so ba mức ngưỡng. Không quyết định huấn
luyện nào dựa vào nó, nhưng nghiêm ngặt thì 25,4% là *test đã nhìn một lần*.

---

## 7. Phân rã lỗi trên test — lỗi nằm ở đâu

728 câu:

| loại | số câu | tỉ lệ |
|---|---:|---:|
| khớp hoàn toàn | 181 | 24,9% |
| **chỉ khác hoa-thường** | 44 | 6,0% |
| **chỉ khác dấu câu** | 77 | 10,6% |
| khác nội dung thật sự | 426 | 58,5% |

> Sửa xong hai nhóm giữa: **24,9% → 41,5%** mà không đụng gì tới ITN.

Trong 426 câu sai nội dung:

| nguyên nhân | số câu | tỉ lệ |
|---|---:|---:|
| mô hình không tìm ra span nào | 182 | 43% |
| viết tắt đã bị mở rộng, không co lại được | 144 | 34% |
| còn span bị ngưỡng chặn | 68 | 16% |

Chủ đề kém nhất: `viết tắt hàng hải` 0,54 · `địa chỉ mạng` 0,65 · `ETA/ETD` 0,67
· `nhận dạng tàu` 0,69. Ba trong bốn vướng đúng vấn đề viết tắt:

```
gốc : Thiết bị AIS có MAC address 00:1A:2B:3C:4D:5E.
nói : thiết bị hệ thống nhận dạng tự động có địa chỉ kiểm soát truy cập môi trường address…
```

---

## 8. Việc cần làm, xếp theo lợi ích trên công sức

### 1. Dấu câu và hoa-thường — 16,6% số câu, gần như miễn phí

Không cần dữ liệu mới; cả hai đã là đầu ra sẵn có của mô hình. Lợi nhất trên mỗi
giờ công. Bắt đầu từ `entity_holdout_dev` vì dấu câu ở đó chỉ 0,596 trong khi
`random_dev` đạt 0,949 — chênh lệch này chưa được giải thích.

### 2. Catalog *cụm nghĩa tiếng Việt → viết tắt* — 144 câu, 20% toàn test

Tra bảng thuần tuý, không cần mô hình. Catalog hiện chỉ có chiều đánh vần
(`a i ét → AIS`), thiếu chiều nghĩa (`hệ thống nhận dạng tự động → AIS`).
Gỡ được ba chủ đề kém nhất.

### 3. Sửa ĐỘ THẬT của dạng nói trong dữ liệu tổng hợp

Không phải sinh thêm — v2 → v3 thêm 9.750 câu mà test chỉ nhích 0,792 → 0,804.
Thêm dữ liệu tổng hợp đã tới hạn.

Quy trình đề xuất cho v4: lấy dạng nói của mỗi kiểu trong `data_v1/data` làm
**chuẩn đối chiếu**, kiểu nào lệch thì sinh lại. Bắt đầu với `VEHICLE_PLATE`
(phải có "gạch ngang"/"chấm"), và vá `VehiclePlateParser` vì nó cũng không đọc
được dạng thật.

### 4. Ngưỡng theo lớp — 68 câu

Để sau cùng. Hạ ngưỡng đồng loạt chỉ được +0,3% và làm 4 chủ đề đi lùi
(`đọc tiền` −0,036, `đọc số âm` −0,027). Phải dò riêng từng lớp trên dev, đúng
như spec §9.4 yêu cầu — spec **cấm** dùng một ngưỡng chung.

### 5. Mở rộng dev thật

Nút thắt đã đổi chỗ: không còn ở thiếu dữ liệu train mà ở **dev quá nhỏ**. 25
trong 42 kiểu có dưới 6 span dev thật, nên F1 1,00 của `HEADING` (3 span) hay
`PERCENT` (1 span) chưa nói được gì.

### KHÔNG nên làm lúc này

Đổi backbone, thêm lớp, đổi hàm mất mát. Không có số liệu nào chỉ vào đó, và cả
bốn việc trên đều rẻ hơn nhiều.

---

## 9. Tệp và công cụ

### Dữ liệu

| tệp | nội dung |
|---|---|
| `datasets_v2/train.jsonl` | 19.580 câu huấn luyện |
| `datasets_v2/dv1_eval.jsonl` | **728 câu test thật** |
| `datasets_v2/v1_aligned.jsonl` | corpus tin tức đã căn hàng |
| `datasets_v2/out_of_taxonomy.jsonl` | 927 span `ORGANIZATION` (taxonomy chưa có lớp này) |
| `datasets_v2/normalizer_gaps.jsonl` | chỗ normalizer chưa khớp — danh sách việc |

### Báo cáo (CSV, UTF-8 BOM, mở Excel được)

| tệp | nội dung |
|---|---|
| `csv/bao_cao_theo_kieu_v3.csv` | 46 kiểu × 24 cột: F1, so với bản trước, nguyên nhân, việc cần làm, ví dụ nói→viết |
| `csv/test_mil_v3.csv` | 728 câu test: vào → ra → gốc, kèm span và span bị ngưỡng chặn |
| `csv/v1_labeled.csv` | corpus tin tức kèm nhãn span |

Cột `vi_du_noi`/`vi_du_viet` trong báo cáo theo kiểu **chỉ nhận cặp mà chuẩn hoá
thực sự đổi nội dung** — vì CSV này dùng làm mẫu sinh thêm dữ liệu, một dòng
`www.x.vn → www.x.vn` sẽ nhân bản đúng lỗi đã giết lớp `ELECTRONIC`.

### Lệnh

```bash
PY=/home/hoangbpm/data_vung_mien/.venv/bin/python

# kiểm thử (223 test)
$PY -m unittest discover -s itn_v2/tests -t .

# huấn luyện
$PY -m itn_v2.train --punct-weight 1.0 --epochs 20 --out-name <tên>.pt

# đánh giá trên dev thật
$PY -m itn_v2.run_eval --checkpoint checkpoints_v2/<tên>.pt

# đánh giá ĐẦU-CUỐI trên test thật
$PY -m itn_v2.eval_dv1 --checkpoint checkpoints_v2/<tên>.pt --out <đường dẫn>.csv

# báo cáo theo từng kiểu span
$PY -m itn_v2.eval_per_type --checkpoint checkpoints_v2/<tên>.pt \
    --baseline <csv bản trước> --extra-split mil_v3_dev --csv <đường dẫn>.csv

# BẮT BUỘC trước mỗi lần train: dọn span vô nghĩa
$PY -m itn_v2.data.drop_identity_spans datasets_v2/train.jsonl datasets_v2/*_dev.jsonl

# chạy trên văn bản thô bất kỳ
$PY -m itn_v2.infer_text <file.csv|txt> --checkpoint checkpoints_v2/mil_v3.pt
```

### Checkpoint

| tệp | mô tả |
|---|---|
| `mil_v3.pt` | **hiện hành** — tin tức + bundle v2 + v3 |
| `mil_clean.pt` | tin tức + bundle v2, đã dọn |
| `standardized_e15.pt` | trước khi có bundle |

Không ghi đè checkpoint cũ để còn so sánh.

---

## 10. Nguyên tắc rút ra

**Cổng kiểm chứng phải kiểm được chính nó.** Ba lỗi lớn nhất đều tự che giấu:
đường tắt của `ElectronicParser` làm cổng luôn báo khớp; `strip_marks` hạ chữ
thường cả hai vế nên casing sai vẫn "đúng"; bộ đọc số trả kết quả hợp lệ nhưng
sai thay vì báo lỗi. Mỗi lần thêm một cổng, phải hỏi *nếu dữ liệu hỏng theo cách
X thì cổng này có bắt được không*.

**Loại một span thì phải loại cả câu.** Giữ câu lại với span bị bỏ nghĩa là gán
`O` cho chỗ đó — dạy mô hình điều ngược lại với hàng trăm câu khác. Đây là lỗi
tốn kém nhất trong dự án.

**Đo trên hai tập là bắt buộc.** Một con số đơn lẻ không phân biệt được "mô hình
giỏi" với "bài thi dễ". Cặp `f1_thật` / `f1_dev_tong_hop` mới nói được sự thật.

**Nhiều dữ liệu hơn không luôn tốt hơn.** Bỏ 36 câu làm mọi chỉ số tăng. Thêm
9.750 câu tổng hợp gần như không nhích test. Độ thật quan trọng hơn số lượng.
