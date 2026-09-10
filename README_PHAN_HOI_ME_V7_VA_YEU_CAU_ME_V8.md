# Phản hồi mẻ v7 / v7.1, và đặc tả mẻ v8

Tài liệu này nối tiếp `README_VAN_DE_SO_VA_YEU_CAU_DU_LIEU.md`. Mọi quy ước
định dạng, hợp đồng nhãn và cổng kiểm tra ở tài liệu đó **vẫn giữ nguyên** —
đây chỉ nói phần cần đổi, dựa trên kết quả đo được sau khi nhận hai mẻ.

Đo trên ba bộ test, cùng ngưỡng 0,5, cùng mã nguồn đã vá.

---

## 1. Kết quả hai mẻ

| checkpoint | câu train | span | khớp hoàn toàn | F1 từ |
|---|---|---|---|---|
| trước v7 | 20.063 | 27.000 | 34,5% | 0,843 |
| **v7** | 31.038 | 42.384 | **38,9%** | **0,873** |
| v7.1 | 34.345 | 46.136 | 35,2% | 0,863 |

**Mẻ v7 tốt: +4,4 điểm.** Giữ nguyên cách làm đó.

**Mẻ v7.1 làm tụt 3,7 điểm.** Không phải vì chất lượng câu kém — vì tỉ lệ.

---

## 2. Vì sao v7.1 làm hỏng: dồn hết vào một lớp

Phần thêm của v7.1 so với v7:

```
+3.307 câu, +3.752 span
   EQUIPMENT_ID   454 -> 3.126   +2.672  = 71% toàn bộ phần thêm  (×6,9 lần)
   ACRONYM      2.468 -> 3.204     +736
   ELECTRONIC   1.771 -> 1.955     +184
   ADDRESS        470 ->   630     +160
   RANK            90 ->    90       +0
   CHANNEL        195 ->   195       +0
   DECIMAL         15 ->    15       +0
```

Kết quả trên bộ test cuối, so từng lớp:

| lớp | v7 | v7.1 | |
|---|---|---|---|
| **EQUIPMENT_ID** | 11/12 | **24/26** | **+13** |
| ACRONYM | 109/111 | 79/81 | −30 |
| RANK | 9/9 | **0/0** | −9 |
| VERSION | 28/30 | 22/22 | −6 |
| QUARTER | 21/21 | 16/16 | −5 |
| CALLSIGN | 3/5 | **0/0** | −3 |
| VESSEL_ID | 3/6 | **0/0** | −3 |
| CHANNEL | 8/12 | 5/9 | −3 |
| ELECTRONIC | 15/15 | 12/12 | −3 |
| **tổng** | **627/729** | **573/668** | **−54** |

Một lớp lên 13, mười lớp xuống 67.

Chú ý cột "0/0": `RANK`, `CALLSIGN`, `VESSEL_ID` không đoán sai — chúng **ngừng
phát ra hoàn toàn**. Khi một lớp chiếm 6,8% tổng số span còn lớp khác chỉ 0,2%,
mô hình học được rằng đoán lớp lớn thì đáng, và im lặng ở lớp nhỏ.

> **Quy tắc số 1 cho mẻ v8: KHÔNG lớp nào được nhận quá 15% tổng số span của
> mẻ.** Thà sinh ít mà đều còn hơn sinh nhiều mà lệch.

---

## 3. Chỗ nghẽn thật: mật độ thực thể trên một câu

Đây là phát hiện quan trọng nhất của đợt này, và nó **không phải chuyện độ dài
câu**.

```
                        thực thể/câu
TRAIN hiện tại                  1,35
test data_v1                    2,40
test hội thoại 20               2,31
test ITN v2                     5,26
```

Tỉ lệ khớp hoàn toàn theo số thực thể có trong câu:

| thực thể/câu | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| data_v1 | 64% | 26% | 33% | 23% | 25% | 25% |
| hội thoại 20 | 83% | 17% | 18% | 22% | **0%** | **0%** |
| ITN v2 | 38% | 21% | 36% | 19% | 9% | **1%** |

Câu 5 thực thể trở lên gần như **không bao giờ** ra đúng trọn vẹn. Mà bộ test
thật lại đầy loại câu đó — đây là câu báo cáo hàng hải điển hình:

```
Lúc 05:08 ngày 01/09/2026, tàu CSB 8005, IMO 9876543, MMSI 574123456,
call sign XVH1234, báo vị trí 10°25′05″N, 107°08′50″E và đang giữ
course 090°, heading 092,5°, tốc độ 10,05 kn.
```

Câu này có 11 thực thể thuộc 8 lớp khác nhau. Dữ liệu train hiện tại hầu như
không có câu nào như vậy — trung bình 1,35 thực thể/câu, tức phần lớn câu chỉ
có **một** chỗ cần chuẩn hoá.

> **Quy tắc số 2: ít nhất 40% số câu mẻ v8 phải có TỪ 4 THỰC THỂ TRỞ LÊN,
> thuộc TỪ 3 LỚP KHÁC NHAU trở lên, trong cùng một câu.**

Đây không phải yêu cầu sinh thêm câu — là yêu cầu sinh **câu dày hơn**. Cùng
một ngân sách, 1.000 câu 5 thực thể có giá trị hơn 5.000 câu 1 thực thể.

**Đính chính:** ở phản hồi miệng trước đó tôi có nói mô hình yếu ở câu ngắn.
Sai. Đo ra thì câu 1–5 từ đạt 66% (t20), cao nhất trong mọi nhóm độ dài; câu
21+ từ mới là 0%. Nguyên nhân không phải độ dài mà là mật độ — câu dài thường
dày thực thể, và chỉ cần sai một chỗ là hỏng cả câu.

---

## 4. Hạn ngạch từng lớp cho mẻ v8

Số span hiện có sau khi cân lại (`train_v72.jsonl`, 32.716 câu, 44.213 span,
46 lớp). Cột "xin thêm" là yêu cầu cho mẻ v8.

### Nhóm A — đói nghiêm trọng, mô hình CHƯA BAO GIỜ phát ra lớp này

| lớp | hiện có | xin thêm | ghi chú |
|---|---|---|---|
| FREQUENCY | 12 | **+600** | Hz, kHz, MHz, GHz — tần số VHF, radar |
| DECIMAL | 15 | **+600** | số thập phân đứng một mình, không kèm đơn vị |
| EQUIPMENT_NAME | 18 | **+600** | tên khí tài dạng chữ, phân biệt với EQUIPMENT_ID |
| DOCUMENT_ID | 21 | **+600** | số hiệu giấy tờ không phải văn bản pháp quy |
| DIGIT_SEQ | 27 | **+600** | chuỗi chữ số đọc rời, không phải số đếm |
| FOREIGN_NAME | 33 | **+600** | tên tàu/cảng/người nước ngoài |

Sáu lớp này dưới 35 span. Mô hình không có đủ ví dụ để học ranh giới, nên nó
không bao giờ chọn chúng. Mỗi lớp cần tối thiểu 600 span thì mới ra khỏi vùng
im lặng.

### Nhóm B — có phát ra nhưng quá thưa

| lớp | hiện có | xin thêm |
|---|---|---|
| RANK | 90 | **+700** |
| QUARTER | 189 | **+500** |
| FRACTION | 189 | **+500** |
| CHANNEL | 195 | **+500** |
| MARITIME_TERM | 233 | **+500** |
| IMO_ID | 246 | **+500** |
| VESSEL_ID | 303 | **+400** |
| DATE | 315 | **+400** |
| RATIO | 366 | **+350** |
| VERSION | 375 | **+350** |
| BEARING | 378 | **+350** |
| RANGE | 384 | **+350** |
| HEADING | 387 | **+350** |
| PERCENT | 390 | **+350** |
| ORDINAL | 408 | **+300** |
| PERSON_NAME | 516 | **+300** |
| ADDRESS | 630 | **+300** |

### Nhóm C — KHÔNG sinh thêm

`ACRONYM` (3.204), `MEASURE` (3.094), `COORD` (3.048), `MONEY` (2.507),
`TIME` (2.460), `TIMEZONE` (2.404), `VEHICLE_PLATE` (2.135), `ETA` (2.097),
`ETD` (2.090), `ELECTRONIC` (1.958), `DISTANCE` (1.727), `EQUIPMENT_ID` (1.200).

Mười hai lớp này đã đủ. Sinh thêm chỉ làm lệch tỉ lệ như mẻ v7.1.

**Tổng xin thêm ≈ 10.500 span.** Nếu mỗi câu mang 4 thực thể như quy tắc số 2
thì khoảng **2.600–3.000 câu** là đủ. Không cần mẻ to.

---

## 5. Bốn cặp lớp bị lẫn — cần câu đối chiếu

Mô hình đọc số đúng nhưng gán sai KIỂU. Đây là loại lỗi không sửa được bằng
luật, chỉ sửa được bằng ví dụ có ngữ cảnh phân biệt.

### 5.1 TIME ↔ DURATION

```
NÓI : ... kéo dài quá mười ba giờ bốn mươi lăm phút
RA  : 13 giờ 45 phút          <- đọc DURATION
GỐC : 13:45                   <- phải là TIME
```

Từ dẫn phân biệt là `sau`, `trước`, `quá`, `chuyển sang`, `chậm hơn`,
`muộn hơn`, `tới`. **Năm cụm đầu xuất hiện 0 lần** trong toàn bộ dữ liệu train
trước một cụm thời gian. Xin sinh ít nhất 300 cặp câu đối chiếu, mỗi cặp dùng
đúng một từ dẫn, một câu ra TIME một câu ra DURATION:

```
Ca trực chuyển sang 13:45.                     -> TIME
Sự cố kéo dài 13 giờ 45 phút.                  -> DURATION
Tàu tới muộn hơn 02:30 so với ETA.             -> TIME
Máy chính chạy liên tục 2 giờ 30 phút.         -> DURATION
```

### 5.2 CARDINAL ↔ VESSEL_ID

```
NÓI : tàu cảnh sát biển tám nghìn không trăm lẻ năm
RA  : 8.005                   <- CARDINAL, có dấu phân nhóm nghìn
GỐC : CSB 8005                <- VESSEL_ID, số hiệu thân tàu, KHÔNG phân nhóm
```

Số hiệu tàu đọc y hệt số đếm. Chỉ ngữ cảnh phân biệt được. Xin 300 cặp:

```
Tàu CSB 8005 đang tuần tra.                    -> VESSEL_ID
Tổng cộng 8.005 tấn hàng đã bốc dỡ.            -> CARDINAL
```

### 5.3 EQUIPMENT_ID ↔ EQUIPMENT_NAME

Đây là lý do `EQUIPMENT_NAME` chỉ có 18 span: mẻ v7.1 gán mọi thứ về
`EQUIPMENT_ID`. Quy ước cần theo:

```
Su-30MK2, AK-630, RDR-5000X, M60      -> EQUIPMENT_ID  (có phần số/mã)
radar hàng hải, máy đo sâu, la bàn từ -> EQUIPMENT_NAME (thuần chữ)
```

### 5.4 DOCUMENT_ID ↔ LEGAL_DOC_ID

```
Số 12/2024/NĐ-CP                       -> LEGAL_DOC_ID  (có ký hiệu cơ quan)
Số 12/2024                             -> DOCUMENT_ID   (không có)
```

---

## 6. Quy ước đã CHỐT — sinh theo đúng thế này

Ba quy ước dưới đây vừa được sửa trong mã và đo có hiệu quả. Dữ liệu mẻ v8
phải khớp, nếu không sẽ dạy ngược lại:

| lớp | trước | sau | đo được |
|---|---|---|---|
| RANK | `Chief officer` | **`Chief Officer`** | 0/9 → 9/9 |
| CHANNEL | `kênh 16` | **`Channel 16`** | 1/12 → 8/12 |
| ADDRESS | `Số 18 Trên 5` | **`Số 18/5`** | 7/20 → 9/20 |

Cụ thể:

- **RANK**: chức danh tiếng Anh hoa MỌI tiếng (`Chief Officer`, `Second
  Engineer`, `Chief Engineer`). Quân hàm tiếng Việt hoa MỘT tiếng đầu
  (`Đại tá`, `Thượng úy`). Không trộn hai quy tắc.
- **CHANNEL**: giữ nguyên từ dẫn người nói, **không dịch**. Nói `channel` thì
  viết `Channel`; nói `kênh` thì viết `Kênh`; nói `ch` thì viết `Ch.`. Số hiệu
  hai chữ số có số 0 dẫn đầu giữ nguyên: `channel không sáu` → `Channel 06`.
- **ADDRESS**: `trên` / `xẹt` / `phần` giữa hai số nhà là dấu `/`, viết sát
  không có khoảng trắng: `số mười tám trên năm lê lợi` → `Số 18/5 Lê Lợi`.
  Tên đường luôn viết hoa từng tiếng.

---

## 7. Dạng nói: quy ước phải theo corpus thật

Đo trên corpus gốc (728 câu, dạng nói do người làm):

- **Từ tiếng Anh GIỮ NGUYÊN chữ Latinh, chỉ hạ chữ thường.** Không phiên âm.

  ```
  ĐÚNG : chief officer yêu cầu kiểm tra lại tải trọng
  SAI  : chíp óp phi xơ yêu cầu kiểm tra lại tải trọng
  ```

- **Chữ viết tắt đọc thành tên chữ cái** khi người Việt đọc vậy:
  `VTS` → `vê tê ét`, `UTC` → `u tê xê`, `AIS` → `ây ai ét`,
  `ECDIS` → `i xi đi ai ét`, `VHF` → `vê hát ép`.

- **Đơn vị đọc thành lời**: `kn` → `hải lý trên giờ`, `NM` → `hải lý`,
  `°C` → `độ c`, `V` → `vôn`, `%` → `phần trăm`, `TP` → `thành phố`.

- **Không còn chữ số nào** trong dạng nói. Không dấu câu, không chữ hoa.

Bộ sinh dạng nói bằng LLM của đợt này đạt 97–99% đúng ở phần số, cao hơn
corpus thật (94,8%), nên cách làm này đáng tin. Chỗ hay sai nhất là **số thập
phân**: `1,25` phải đọc `một phẩy hai lăm` (đọc rời từng chữ số sau dấu phẩy),
KHÔNG phải `một phẩy hai mươi lăm`.

---

## 8. Bảng kiểm trước khi gửi mẻ v8

Mẻ sẽ bị bộ chuyển đổi loại nếu không qua. Tự kiểm trước cho đỡ mất vòng:

1. **Tỉ lệ**: không lớp nào quá 15% tổng số span.
2. **Mật độ**: ≥ 40% số câu có ≥ 4 thực thể thuộc ≥ 3 lớp.
3. **Dựng lại được**: chạy ngược từ nhãn span ra bản viết phải trùng bản gốc
   từng ký tự. Lệch một dấu cách cũng là hỏng.
4. **Biên cực đại**: từ đứng sát ngoài một span không được là từ chỉ số, trừ
   khi nó thuộc một span khác.
5. **Không chữ số trong dạng nói.**
6. **Câu toàn nhãn O được giữ** — đó là mẫu âm hợp lệ, đừng lọc bỏ.
7. Sáu lớp nhóm A mỗi lớp ≥ 600 span. Đây là điều kiện đủ để mẻ có ích.

---

## 9. Tóm tắt cho người sinh dữ liệu

Nếu chỉ đọc một đoạn thì đọc đoạn này:

> Mẻ v7 tốt, mẻ v7.1 làm tụt điểm vì dồn 71% dữ liệu mới vào một lớp.
> Mẻ v8 cần **ít câu hơn nhưng dày hơn**: khoảng 2.600–3.000 câu, mỗi câu
> 4 thực thể trở lên thuộc 3 lớp trở lên, phân bố đều theo hạn ngạch mục 4,
> không lớp nào quá 15%. Ưu tiên tuyệt đối sáu lớp nhóm A đang dưới 35 span.
> Thêm 600 cặp câu đối chiếu cho bốn cặp lớp hay lẫn ở mục 5. Bám ba quy ước
> đã chốt ở mục 6 và quy ước dạng nói ở mục 7.
