# Vấn đề: số vẫn sai, và vì sao thêm mô hình không cứu được

Tài liệu này ghi lại kết quả mổ xẻ lỗi SỐ của ITN V2 trên bộ test cuối, chỉ ra
nguyên nhân gốc, và đặc tả chính xác **dữ liệu cần sinh thêm**.

Đo trên checkpoint tốt nhất hiện có, ngưỡng 0,5, bộ test cuối 728 câu.

---

## 1. Bối cảnh: đổi kiến trúc không thu được gì

| checkpoint | khớp hoàn toàn | F1 từ | đủ số |
|---|---|---|---|
| V1 cũ (PhoBERT, 93 nhãn BIO gộp) | 14,0% | 0,765 | 61,8% |
| mil_v3 | 25,4% | 0,804 | 39,2% |
| mil_v52 (+corpus V1) | 31,9% | 0,838 | 54,1% |
| mil_v51 (PhoBERT base, 135 tr tham số) | 34,2% | 0,843 | 56,8% |
| xlmr (XLM-R base, 278 tr tham số) | **34,5%** | **0,843** | 56,4% |

Gấp đôi tham số, đổi hẳn vốn từ (64 nghìn → 250 nghìn) và dữ liệu tiền huấn
luyện (thuần Việt → 100 ngôn ngữ) chỉ đổi được **2 câu trên 728**, F1 giống hệt,
khôi phục số còn kém hơn. Phân rã lỗi của hai backbone gần như trùng khít.

**Kết luận: kiến trúc không phải chỗ nghẽn.** Chỗ nghẽn nằm ở dữ liệu.

---

## 2. Số đi về đâu

Đếm từng token chứa chữ số trong bản gốc (đã bóc dấu câu bám ngoài):

```
611 token số trong bản gốc
  326 (53,4%)  ra đúng
  285 (46,6%)  mất
   91          thừa (bịa hoặc sai khuôn)
```

Ba nguyên nhân của 285 token mất:

| nguyên nhân | số ca | tỉ lệ |
|---|---|---|
| **D — mô hình không sinh span nào** | 182 | 63,9% |
| **A — có span, đủ chữ số, sai khuôn** | 60 | 21,1% |
| **B — span đúng, bị validator/ngưỡng chặn** | 43 | 15,1% |

Trong nhóm A, **9 ca là lỗi của thước đo chứ không phải của hệ thống**: bản gốc
viết `4.8 bar`, `0.8 NM`, `-0.5` bằng dấu **chấm** thập phân nhưng cùng file lại
viết `72,5%` bằng dấu **phẩy** — dùng chấm 88 lần, phẩy 29 lần. Bản gốc tự mâu
thuẫn nên không có đầu ra nào đúng được cả hai. Nhóm A thật sự là 51 ca.

### Theo chủ đề (A / B / D)

```
chủ đề                                 gold  thiếu       %   A/B/D
doc vu khi hai quan ngu loi              33     32   97,0%   3/5/24
doc vu khi trang bi quan su              34     32   94,1%   0/0/32
doc so am                                43     24   55,8%   3/1/20
doc so dien thoai                        26     22   84,6%   0/19/3
doc version phien ban                    35     21   60,0%  20/0/1
doc dia chi mang                         25     20   80,0%   0/1/19
doc dia chi                              31     17   54,8%   5/3/9
doc eta etd                              22     17   77,3%   3/1/13
doc van ban phap ly                      11     11  100,0%   0/0/11
doc gio                                  36     10   27,8%   4/1/5
doc thong tin nhan dang tau              23     10   43,5%   3/4/3
doc bien so xe                           36      9   25,0%   0/0/9
doc toa do                               28      9   32,1%   1/7/1
doc ty le phan so                        21      9   42,9%   5/0/4
doc don vi do luong                      47      8   17,0%   1/0/7
doc mui gio                              23      8   34,8%   0/0/8
doc lenh dieu dong tieng anh hang hai     9      7   77,8%   0/0/7
doc tu viet tat quan su canh sat bien    10      7   70,0%   5/0/2
doc quy so la ma                          6      5   83,3%   4/0/1
doc vhf kenh lien lac                     7      3   42,9%   0/1/2
doc tien                                 47      2    4,3%   2/0/0
doc ngay                                 29      0    0,0%   —
doc phan tram                            21      0    0,0%   —
```

---

## 3. Nguyên nhân gốc của 64% lỗi: train và test đọc theo hai kiểu khác nhau

Đây **không phải** mô hình yếu, cũng **không phải** thiếu câu. Đây là những cách
đọc mà mô hình chưa từng nhìn thấy một lần nào.

Đếm trên toàn bộ 20.063 câu huấn luyện của `train_v51.jsonl`:

| cụm đọc xuất hiện trong test | số câu train chứa nó | tỉ lệ mất số của chủ đề |
|---|---|---|
| `nghị định chính phủ` | **0 / 20.063** | 100% |
| `thông tư bộ` | **0 / 20.063** | 100% |
| `i p v bốn` / `i p v sáu` | **0 / 20.063** | 80% |
| `ét u gạch ngang` (Su-30) | **0 / 20.063** | 94% |
| `a ca gạch ngang` (AK-47) | 8 | 94% |

Cùng một thực thể, train và test đọc khác hẳn:

```
VĂN BẢN PHÁP LÝ
  train : số mười hai xẹt hai không hai tư nờ đê cê pê
  test  : số hai mươi hai năm hai nghìn không trăm hai mươi tư nghị định chính phủ
          ────────┬────────  ─┬─  ────────────┬──────────────  ────────┬────────
                  │           │               │                        │
          đọc số đầy đủ    "năm" thay "xẹt"  năm đọc đầy đủ    đọc NGHĨA thay vì đánh vần

VŨ KHÍ / KHÍ TÀI
  train : a ka bốn bảy      | mờ bốn a một  | su ba mươi
  test  : a ca gạch ngang bốn bảy | em bốn a một | ét u gạch ngang ba không em ca hai
          ──────┬──────           ──┬─          ──────┬──────
        CÓ đọc dấu gạch      tên chữ cái      tên chữ cái + có dấu gạch
```

Cộng thêm việc các kiểu liên quan **đói dữ liệu trầm trọng**:

```
TIMEZONE      2397 span         EQUIPMENT_ID     54 span
MEASURE       2084              DIGIT_SEQ        27
TIME          1830              DOCUMENT_ID      21
DISTANCE      1727              EQUIPMENT_NAME   18
COORD         1688              DECIMAL          15
                                FREQUENCY        12
```

Hai chủ đề vũ khí đóng góp 56 trong 182 ca không nhận ra — được huấn luyện bằng
đúng **54 span**, mà lại theo cách đọc khác.

Nguồn dữ liệu train hiện tại: **79% đến từ một bộ sinh duy nhất** (`military_v2`),
16% `v1align`, 5% `gemma`. Một bộ sinh thì chỉ có một văn phong.

> Đây là lời giải cho việc XLM-R hoà PhoBERT: không backbone nào đoán được
> `nghị định chính phủ` phải ra `NĐ-CP` khi chưa từng thấy cụm đó.

---

## 4. Ba lỗi kỹ thuật độc lập (không liên quan dữ liệu)

**4.1 — Điện thoại bị gán nhầm thành MMSI: 19 số.**
Mô hình dự đoán `MMSI_ID` cho `0908123456`, validator từ chối đúng (MMSI phải 9
chữ số), pipeline giữ dạng nói. `TELEPHONE` 369 span và `MMSI_ID` 381 span đều là
chuỗi chữ số trần, không gì phân biệt ngoài ngữ cảnh.
*Hướng sửa tổng quát:* khi validator của kiểu dự đoán từ chối, thử các **kiểu anh
em cùng hình dạng** trước khi bỏ cuộc. Chuỗi 10 chữ số mở đầu bằng `0` chỉ có thể
là điện thoại. Cùng cơ chế này gỡ luôn `MK46 → PORT_CODE` và `IPV6 → PORT_CODE`.

**4.2 — Toạ độ dạng độ + phút thập phân (DDM): 7 số.**
Bản gốc dùng `10°25.500′N`, đọc là *"mười độ hai mươi lăm nghìn năm trăm phút"*.
Parser hiện chỉ biết độ-phút-giây nên ra `10°25500′N` và bị validator chặn.
Cần thêm nhánh phút thập phân.

**4.3 — Nhầm `TIME` thành `DURATION`: ~10 số, và `VERSION` sai khuôn: 20 số.**
`mười bảy giờ mười phút` ra `17 giờ 10 phút` thay vì `17:10`. Đúng về số, sai về
khuôn. Đây là lỗi phân loại kiểu, sửa được bằng dữ liệu phân biệt ngữ cảnh.

### Đã sửa — kết quả đo lại trên cùng bộ test

`4.1` và `4.2` đã sửa xong, cùng với khuôn số điện thoại quốc tế và đầu số dịch vụ:

| | trước | sau |
|---|---|---|
| khớp hoàn toàn (XLM-R) | 34,5% | **34,9%** |
| F1 theo từ | 0,843 | **0,847** |
| khôi phục số | 56,4% | **59,8%** |
| token số mất | 285 | **266** |
| nhóm B (bị chặn) | 43 | **25** |
| token số **thừa** (bịa) | 91 | **91** — không phát sinh |

Cùng mức cải thiện trên PhoBERT `mil_v51`: 34,2% → 34,5%, F1 0,843 → 0,846.
Số bịa đứng yên, tức là phần thu hồi được là thu hồi thật chứ không phải nới cổng.

**Cơ chế đã thêm — cổng dự phòng giữa kiểu anh em** (`pipeline.FALLBACK_TYPES`):
khi validator của kiểu mô hình dự đoán bác bỏ, pipeline thử các kiểu cùng hình
dạng dạng nói. Điều kiện an toàn bắt buộc: **chỉ lùi sang kiểu CÓ validator
riêng**. Kiểu không validator sẽ nhận mọi thứ, biến cổng dự phòng thành đường
bịa giá trị — đúng thứ spec §16 cấm. Vì vậy `DIGIT_SEQ`, `VESSEL_ID`,
`CALLSIGN` không nằm trong bảng, và `MK46 → PORT_CODE` vẫn chưa gỡ được bằng
cách này.

### Chưa sửa — và cố ý không sửa bằng code

**`4.3` TIME vs DURATION không được sửa bằng luật cứng.** Đã tra cụm dẫn đứng
ngay trước span trong toàn bộ dữ liệu huấn luyện:

| cụm dẫn | tổng | TIME | DURATION |
|---|---|---|---|
| `lúc` | 87 | 87 | 0 |
| `vào` | 61 | 61 | 0 |
| `từ` | 61 | 61 | 0 |
| `khoảng` | 120 | 0 | 120 |
| `dài` | 64 | 0 | 64 |
| **`sau`** | **0** | 0 | 0 |
| **`trước`** | **0** | 0 | 0 |
| **`quá`** | **0** | 0 | 0 |
| **`sang`** | **0** | 0 | 0 |
| **`hơn`** | **0** | 0 | 0 |

Đúng những cụm dẫn mà bộ test dùng (`sau 23:30`, `quá 13:45`, `chuyển sang
17:10`, `chậm hơn 16:45`) **xuất hiện 0 lần** trong huấn luyện. Mô hình đoán
`DURATION` là nhất quán với thứ nó được dạy.

Viết một bảng cứng `sau/trước/quá/sang/hơn → TIME` sẽ nâng điểm, nhưng đó là
**chỉnh tay theo đáp án của bộ test cuối** — cùng loại vi phạm với việc chọn
ngưỡng trên test, thứ `tune_thresholds.py` đã chặn bằng `SystemExit`. Lỗi này
thuộc về dữ liệu và phải sửa ở §5.4.

Ba số quốc tế còn lại cũng không sửa bằng code: mô hình cắt biên span **bỏ mất
từ `cộng`**, nên parser chỉ nhận `tám bốn chín không tám…`. Suy ra dấu `+` từ
việc chuỗi bắt đầu bằng `84` là bịa ra một ký tự không có trong span. Đây là lỗi
biên, phải sửa bằng dữ liệu (§5.4).

---

## 5. Đặc tả dữ liệu cần sinh

Mục tiêu: **bổ sung biến thể CÁCH ĐỌC, không phải tăng số lượng câu.** Sinh thêm
20 nghìn câu cùng văn phong cũ sẽ không cải thiện gì.

### 5.1 Định dạng — JSONL, mỗi dòng một câu

```json
{
  "id": "chuỗi 16 hex, duy nhất",
  "spoken": "phát tín hiệu sê cu ri tê để yêu cầu cứu nạn khẩn cấp",
  "written": "Phát tín hiệu SECURITE để yêu cầu cứu nạn khẩn cấp.",
  "words": ["phát","tín","hiệu","sê","cu","ri","tê","để","yêu","cầu","cứu","nạn","khẩn","cấp"],
  "spans": [[3, 6, "MARITIME_TERM"]],
  "types": ["MARITIME_TERM"],
  "punct": ["O","O","O","O","O","O","O","O","O","O","O","O","O","PERIOD"],
  "written_source": "",
  "leak_key": "SECURITE",
  "source": "tên_bộ_sinh:tên_lớp"
}
```

Ràng buộc bắt buộc:

- `words` = `spoken.split()`, **không tách từ ghép**, không dấu gạch dưới.
- `spans` = danh sách `[đầu, cuối, KIỂU]`, chỉ số **tính trên `words`**, `cuối`
  **bao gồm** (inclusive). Các span không được chồng lấn, phải sắp tăng dần.
- `types` = danh sách kiểu theo span, **cùng thứ tự và cùng độ dài với `spans`**.
- `punct` = một nhãn cho **mỗi** phần tử của `words`, thuộc
  `{"O", "COMMA", "PERIOD", "QUESTION"}`. Dấu câu nằm ở từ **cuối** của cụm nó theo sau.
- `written` phải là kết quả đúng khi áp span lên `spoken` — bao gồm cả hoa/thường
  và dấu câu.
- `spoken` **không chứa chữ số và không chứa dấu câu**. Chỉ chữ thường.
- `leak_key` = dạng chuẩn của span chính, dùng để chống rò rỉ giữa train và test.

### 5.2 Bộ 46 kiểu hợp lệ

```
số học   : CARDINAL ORDINAL DIGIT_SEQ DECIMAL FRACTION PERCENT RANGE RATIO
           MONEY MEASURE VERSION
ngày giờ : DATE TIME TIMEZONE DURATION ETA ETD QUARTER
hàng hải : COORD HEADING BEARING SPEED DISTANCE DEPTH DRAFT FREQUENCY CHANNEL
định danh: MMSI_ID IMO_ID CALLSIGN VESSEL_ID PORT_CODE DOCUMENT_ID LEGAL_DOC_ID
           TELEPHONE ELECTRONIC VEHICLE_PLATE ADDRESS
thực thể : EQUIPMENT_ID EQUIPMENT_NAME FOREIGN_NAME PERSON_NAME LOCATION_NAME
           ACRONYM MARITIME_TERM RANK
```

Kiểu ngoài danh sách này sẽ bị loại khi nạp.

### 5.3 Ưu tiên — số câu cần sinh theo từng lớp

Con số ở cột cuối là ước lượng số token số thu hồi được nếu làm đúng.

| ưu tiên | lớp | số câu | thu hồi |
|---|---|---|---|
| 1 | `LEGAL_DOC_ID` đọc theo NGHĨA | 800 | ~11 |
| 1 | `EQUIPMENT_ID` đủ biến thể đọc | 2.500 | ~56 |
| 1 | `ELECTRONIC` — IPv4, IPv6, MAC | 1.200 | ~20 |
| 2 | `CARDINAL`/`MEASURE` số âm | 800 | ~20 |
| 2 | `TELEPHONE` phân biệt với `MMSI_ID` | 600 | ~19 |
| 2 | `VERSION` | 600 | ~20 |
| 3 | `COORD` phút thập phân | 400 | ~7 |
| 3 | `TIME` phân biệt với `DURATION` | 600 | ~10 |
| 3 | `ADDRESS` có dấu `/` | 400 | ~9 |

### 5.4 Yêu cầu chi tiết từng lớp

#### LEGAL_DOC_ID — bắt buộc sinh cả hai lối đọc

```
đánh vần (đã có):  số mười hai xẹt hai không hai tư nờ đê cê pê   -> Số 12/2024/NĐ-CP
đọc nghĩa (THIẾU): số hai mươi hai năm hai nghìn không trăm hai mươi tư nghị định chính phủ
                                                                  -> Số 22/2024/NĐ-CP
```

Cần phủ các ký hiệu: `NĐ-CP` (nghị định chính phủ), `TT-BTC` (thông tư bộ tài
chính), `TT-BQP` (thông tư bộ quốc phòng), `QĐ-TTg` (quyết định thủ tướng),
`TT-BCA`, `NQ-CP`, `CT-TTg`, `QĐ-BQP`.

Biến thể dấu `/`: `xẹt`, `xẹc`, `trên`, `gạch chéo`, **`năm`** (khi phần sau là năm).
Biến thể năm: `hai không hai tư` và `hai nghìn không trăm hai mươi tư`.
Tỉ lệ hai lối đọc: **50/50**.

#### EQUIPMENT_ID — lớp đói nhất, chỉ có 54 span

Sinh theo tổ hợp của bốn trục, mỗi số hiệu ít nhất 8 biến thể:

1. **Tên chữ cái vs âm đọc tắt**: `a ca` / `a ka` / `ây kây`; `em` / `mờ` / `em mờ`;
   `ét u` / `su` / `xu`; `ca hát` / `khát`; `e rờ pê gờ` / `rợt pi ji`.
2. **Có hoặc không đọc dấu gạch**: `a ca gạch ngang bốn bảy` vs `a ca bốn bảy`.
   Biến thể dấu: `gạch ngang`, `gạch nối`, `gạch`, `trừ`.
3. **Cách đọc số**: `bốn bảy` (từng chữ số) vs `bốn mươi bảy` (số nguyên);
   `sáu ba không` vs `sáu trăm ba mươi`; `tám không không` vs `tám trăm`.
4. **Hậu tố chữ**: `Su-30MK2` = `ét u gạch ngang ba không em ca hai`,
   `M4A1` = `em bốn a một`, `53-65KE` = `năm ba gạch ngang sáu năm ca e`.

Danh sách số hiệu tối thiểu cần phủ:
```
AK-47  AK-74  AK-101  AK-630   M16  M4A1  M60  M249
RPG-7  RPG-29  PKM  SVD  IGLA
Su-27  Su-30  Su-30MK2  Su-34  MiG-21  MiG-29  MiG-31
F-16  F-22  F-35  B-52  UH-60  AH-64  CH-47
Mi-8  Mi-17  Mi-24  Ka-27  Ka-28  Ka-52
T-54  T-55  T-72  T-90  BMP-1  BMP-2  BTR-80
53-65KE  SET-65  MU90  MK 46  Kh-35  P-800  P-15  3M-54
S-125  S-300  S-400  Pantsir-S1
```

#### ELECTRONIC — IPv4, IPv6, MAC

```
IPv4 : i p v bốn một trăm chín mươi hai chấm một trăm sáu mươi tám chấm mười chấm hai mươi lăm
       -> IPv4 192.168.10.25
IPv6 : i p v sáu hai không không một hai chấm đê bê tám hai chấm một không không hai chấm hai chấm hai năm
       -> IPv6 2001:db8:100::25
MAC  : không không hai chấm một a hai chấm hai bê hai chấm ba xê hai chấm bốn đê hai chấm năm e
       -> 00:1A:2B:3C:4D:5E
```

Điểm mấu chốt: **`hai chấm` = dấu `:`** còn **`chấm` = dấu `.`**. Phải phủ dày cả
hai, vì hiện tại `i p v bốn` xuất hiện 0 lần. Thêm dạng `link-local`
(`ép e tám không hai chấm hai chấm một không không` → `fe80::100`), cổng
(`chấm hai mươi lăm hai chấm tám không` → `.25:80`), và dải CIDR.

Biến thể đọc `IPv4`: `i p v bốn`, `ai pi vi bốn`, `ip vờ bốn`, `địa chỉ i p`.

#### Số âm

```
âm mười hai              -> -12       [CARDINAL]
âm không phẩy tám        -> -0,8      [CARDINAL]
âm hai mươi tư vôn       -> -24 V     [MEASURE]
âm tám phẩy năm đề xi ben -> -8,5 dB  [MEASURE]
```

Hiện có 798 câu chứa `âm + số` nhưng **650/798 mang kiểu `MEASURE`**, chỉ 148
mang `CARDINAL`. Test lại chủ yếu là `CARDINAL` trần. Cần đảo tỉ lệ về khoảng
50/50, và bắt buộc có câu chứa **nhiều số âm liên tiếp**
(`âm một âm năm âm mười hai và âm không phẩy tám`) — dạng này hiện hỏng hoàn toàn.

#### TELEPHONE tách khỏi MMSI_ID

```
không chín không tám một hai ba bốn năm sáu        -> 0908123456     [TELEPHONE]
cộng tám bốn chín không tám một hai ba bốn năm sáu -> +84 908 123 456 [TELEPHONE]
một chín không không một hai ba bốn                -> 19001234       [TELEPHONE]
không hai tám ba tám hai ba bốn năm sáu bảy        -> 02838234567    [TELEPHONE]
```

Quy tắc phân biệt phải học được từ ngữ cảnh: cụm dẫn `số điện thoại`, `số trực
ban`, `hotline`, `số của đại lý`, `liên lạc qua` → `TELEPHONE`; cụm dẫn `MMSI`,
`em em ét i`, `nhận dạng tàu` → `MMSI_ID`. Sinh **cả câu chứa đồng thời hai loại**
để mô hình buộc phải dùng ngữ cảnh.

**Bắt buộc về biên span**: với số quốc tế, từ `cộng` phải nằm **bên trong** span
`TELEPHONE`, không được để ngoài. Hiện mô hình cắt biên bỏ mất `cộng` ở cả 3/3
số quốc tế của bộ test, khiến parser không dựng được dấu `+`. Cần ít nhất 150
câu có `cộng` mở đầu span.

#### COORD phút thập phân (DDM)

```
mười độ hai mươi lăm nghìn năm trăm phút bắc      -> 10°25.500′N
một trăm lẻ bảy độ tám nghìn hai trăm năm mươi phút đông -> 107°08.250′E
```

Chú ý: `hai mươi lăm nghìn năm trăm phút` = **25.500 phút** (25 phút và 500 phần
nghìn), không phải hai mươi lăm nghìn phút. Phải sinh kèm dạng độ-phút-giây đã có
để mô hình phân biệt được hai khuôn.

#### TIME tách khỏi DURATION

```
lúc mười bảy giờ mười phút        -> 17:10          [TIME]
kéo dài mười bảy giờ mười phút    -> 17 giờ 10 phút [DURATION]
```

Cụm dẫn quyết định kiểu: `lúc`, `vào`, `từ`, `đến`, `sau`, `trước`, `chậm hơn`,
`quá`, `chuyển sang`, `muộn hơn` → `TIME`; `kéo dài`, `trong vòng`, `mất`, `hết`,
`thêm` → `DURATION`. Sinh cặp câu chỉ khác nhau ở cụm dẫn.

**Bắt buộc**: năm cụm dẫn `sau`, `trước`, `quá`, `sang`, `hơn` hiện xuất hiện
**0 lần** trong dữ liệu huấn luyện. Mỗi cụm cần tối thiểu 60 câu `TIME`.
Cũng cần câu `khoảng 10:10` mang kiểu `TIME` — hiện `khoảng` gắn với `DURATION`
120/120 lần nên mô hình không thể học được nghĩa còn lại.

#### ADDRESS có dấu `/`

```
số mười tám trên năm lê lợi -> Số 18/5 Lê Lợi
số tám trên mười hai nguyễn du -> Số 8/12 Nguyễn Du
```

Hiện `trên` trong địa chỉ bị đọc thành chữ `Trên` (`Số 18 Trên 5`). Cần dạy
`trên` giữa hai số nhà = dấu `/`. Đồng thời phân biệt với `RATIO`
(`hai mươi mốt trên ba` → `21:3`) bằng ngữ cảnh có `số nhà`/tên đường theo sau.

### 5.5 Cổng kiểm tra bắt buộc trước khi nộp

Mỗi câu sinh ra phải qua hết bốn cổng, trượt cổng nào thì **bỏ cả câu**, không
được chỉ bỏ span (giữ lại sẽ dạy mô hình nhãn `O` ở đúng vị trí đó):

1. `len(words) == len(punct)` và `len(spans) == len(types)`.
2. Mọi chỉ số span nằm trong `[0, len(words)-1]`, `đầu <= cuối`, không chồng lấn.
3. **Biên cực đại**: từ ngay trước `đầu` và ngay sau `cuối` không được là từ chỉ
   số (`không một hai ba bốn năm sáu bảy tám chín mười mươi trăm nghìn ngàn triệu
   tỷ lẻ linh mốt tư lăm nhăm bẩy tỉ`). Nếu là, biên đã bị cắt cụt.
4. **Tái dựng**: chạy pipeline ở chế độ oracle (dùng đúng span vàng) phải cho ra
   `written`. Lệch một ký tự cũng loại.

### 5.6 Chống rò rỉ

Bộ test cuối là `data_v1/data` (728 câu). **Không** được lấy câu, cụm đọc, hay
giá trị nào từ đó làm dữ liệu sinh. `leak_key` của câu sinh phải được đối chiếu
với tập `leak_key` của test và loại nếu trùng.

### 5.7 Đa dạng văn phong

79% dữ liệu hiện tại đến từ một bộ sinh, nên mọi câu có cùng nhịp và cùng bộ cụm
dẫn. Mỗi lớp cần tối thiểu **6 khung câu khác nhau** và cụm dẫn lấy ngẫu nhiên từ
danh sách ít nhất 10 phần tử. Cần có cả câu **nhiều span cùng loại**
(`các ký hiệu AK-47, M16, M4A1 và RPG-7`) — dạng liệt kê này hiện hỏng gần như
hoàn toàn.

---

## 6. Cách kiểm tra sau khi có dữ liệu

```bash
python -m itn_v2.train --train datasets_v2/train_v6.jsonl \
    --dev datasets_v2/random_dev.jsonl --punct-weight 1.0 --epochs 20 \
    --out-name mil_v6.pt

python -m itn_v2.eval_dv1 --checkpoint checkpoints_v2/mil_v6.pt \
    --threshold 0.5 --out datasets_v2/csv/test_v6.csv
```

Chỉ số cần theo dõi, xếp theo mức quan trọng:

1. **Khôi phục số** — hiện 59,8%. Đây là chỉ số phản ánh đúng vấn đề này nhất.
2. **Tỉ lệ nhóm D** — hiện 181/266 = 68,0%. Nếu dữ liệu đúng, nhóm này phải co lại.
3. Khớp hoàn toàn — hiện 34,9%.
4. F1 từ — hiện 0,847.

Cảnh báo: **tuyệt đối không chọn ngưỡng trên bộ test cuối.** `tune_thresholds.py`
đã chặn cứng bằng `SystemExit` nếu đường dẫn chứa `dv1_eval`.
