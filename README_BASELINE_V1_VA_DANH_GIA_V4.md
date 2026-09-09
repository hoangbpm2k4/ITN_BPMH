# Đối chiếu V1 và đánh giá gói dữ liệu v4

_Đo ngày 09/09/2026. Bổ sung cho [README_ITN_V2_TRANG_THAI.md](README_ITN_V2_TRANG_THAI.md)._
_Mọi con số dưới đây là kết quả chạy thật, không phải ước lượng._

---

## 1. V1 (hệ cũ) chạy trên đúng bộ test của V2

Từ trước tới nay V1 và V2 chưa bao giờ được đo trên cùng một thước. Lần này V1 —
checkpoint `v11_joint_fullattn-epoch=14-val_punc_f1=0.8440-val_itn_f1=0.9794` —
chạy trên đúng 728 câu của `data_v1/data`, dùng lại nguyên `canon()` và
`token_f1()` của `itn_v2/eval_dv1.py`. Mã: [itn_v2/eval_v1_legacy.py](itn_v2/eval_v1_legacy.py).

| | V1 (v11) | V2 (mil_v3) |
|---|---|---|
| câu khớp hoàn toàn | 102/728 = **14,0%** | 185/728 = **25,4%** |
| F1 theo từ | 0,765 | 0,804 |
| span phát ra | 2.134 | 490 |

Phân rã lỗi:

| | V1 | V2 |
|---|---|---|
| khớp hoàn toàn | 14,0% | 25,4% |
| chỉ sai hoa-thường | 12,2% | 6,0% |
| chỉ sai dấu câu | 9,1% | 9,9% |
| sai nội dung | 64,7% | 58,7% |

Theo từng câu: V2 tốt hơn ở 389 câu, V1 tốt hơn ở 160, hoà 179.

### 1.1 Nhưng ở việc cốt lõi, V1 vẫn đang mạnh hơn

Xét riêng 296 câu mà **bản gốc có chữ số** — tức phần việc ITN thật sự:

| | V1 | V2 |
|---|---|---|
| khôi phục **đủ mọi** cụm số trong câu | 183 = **61,8%** | 116 = 39,2% |
| đầu ra có ít nhất một chữ số | **97,6%** | 67,2% |

V2 hụt số ở **180/296 câu**, chia gần đều ba nguyên nhân:

| nguyên nhân | số câu |
|---|---|
| bị ngưỡng tin cậy chặn | 59 |
| không tìm thấy span | 59 |
| tìm ra span nhưng chuyển sai | 62 |

Trong 180 câu đó, **V1 lấy đúng đủ số ở 87 câu**.

Kết luận: V2 hơn V1 chủ yếu nhờ *không làm bậy* — viết hoa đúng, không tự ý viết
hoa từ tiếng Anh, đặt dấu câu tốt hơn. Còn ở việc biến chữ thành số thì đầu NUM
của V1 chuyển thẳng tay và ăn đứt cơ chế ngưỡng của V2.

Hai đầu bảng chủ đề:

```
doc tien              F1 V1 0,891  V2 0,822   -0,069
doc dia chi           F1 V1 0,809  V2 0,727   -0,082
...
doc don vi do luong   F1 V1 0,784  V2 0,915   +0,131
doc lenh dieu dong    F1 V1 0,657  V2 0,805   +0,148
```

V1 đúng, V2 để nguyên dạng nói:
```
gốc: … như 123.456 đồng, 1.234.567 đồng, 12.345.678 đồng …
V1 : … như 123.456 đồng 1.234.567 đồng 12.345.678 đồng …
V2 : … như một trăm hai mươi ba nghìn bốn trăm năm mươi sáu đồng, …
```

V2 đúng, V1 viết hoa bừa:
```
gốc: Các destination và port name sau đó được nhập vào voyage plan.
V1 : Các Destination và Port Name sau đó được nhập vào Voyage Plan
V2 : Các destination và port name sau đó được nhập vào voyage plan.
```

### 1.2 Một lỗi trong mã V1 phải sửa mới chạy được

Lần chạy đầu ra 0,0% và F1 0,000 ở **mọi** chủ đề — đó là bug, không phải kết quả.
`ITNPostProcessor.decode` gộp subword theo hậu tố `</w>`, nhưng tokenizer PhoBERT
trên máy này (`PhobertTokenizer`, bản chậm) đánh dấu mảnh giữa bằng `@@`. Không
token nào kết thúc bằng `</w>` nên `merged_toks` rỗng và decode trả chuỗi rỗng.

Nghĩa là `inference.py` của V1 **đúng như đang có trong repo thì cho đầu ra rỗng**
với môi trường hiện tại. Cách vòng qua trong `eval_v1_legacy.py`: gộp theo
`position_ids` — biết chắc ranh giới từ, lấy nhãn nội dung ở mảnh đầu và nhãn dấu
câu ở mảnh cuối, đúng như `get_word_level_labels` vẫn làm.

### 1.3 Lưu ý khi đọc con số

V1 chỉ có 15 nhãn và chưa từng được huấn luyện cho phần lớn chủ đề trong bộ test
này (email, website, VHF, mã cảng, múi giờ). 14,0% là "hệ cũ nguyên trạng đặt vào
bài toán mới", không phải trần của kiến trúc đó.

---

## 2. Đánh giá gói `ITN_BPMH_v4_plan_data_catalog.zip`

Đã giải nén, chạy đủ ba script kiểm tra của gói, và đối chiếu với bộ test thật.

**Kết luận: khả thi — nhưng nó là bản *hợp đồng nhãn + kế hoạch*, không phải gói
data đẩy được điểm test.** Chất lượng kỹ thuật cao hơn hẳn v2/v3, nhưng phần dữ
liệu chỉ chạm khoảng một nửa chỗ đang thua, và ở chỗ thua nặng nhất thì vẫn sai
dạng nói y như v3.

### 2.1 Phần làm tốt

```
scripts/validate_v4.py    -> 6760 dòng, 0 lỗi offset/BIO/reconstruction
scripts/test_reference.py -> 17 regression + 4 KEEP case, 0 failure
```

- **Taxonomy khớp tuyệt đối**: 46/46 nhãn trùng `SEMANTIC_TYPES` trong
  [itn_v2/labels.py](itn_v2/labels.py), không thừa không thiếu.
- **Định dạng tốt hơn v3 nhiều**: char offset (`end` exclusive, NFC), BIO trên cả
  dạng nói lẫn dạng viết, `entity_group` để chia split, `template_id`, `provenance`.
- **540 câu all-O** — 24.300 dòng cũ không có câu nào, nên chưa bao giờ đo được
  tỷ lệ sửa thừa.
- **Vai trò tương phản có ngữ cảnh thật sự phân biệt**, không phải gán nhãn bừa:
  ```
  [DEPTH]    …báo lại độ sâu luồng mười phẩy tám một mét…
  [DRAFT]    …mười phẩy tám một mét là mớn nước không phải độ sâu…
  [DISTANCE] …cự ly quan sát mười phẩy tám một mét…
  ```
- **Đính chính đúng một kết luận sai trước đây của chúng tôi**: "0 leakage" chỉ
  đúng khi so cặp (nhãn, canonical); gom theo họ thực thể thì `AK-105` ở
  EQUIPMENT_ID trùng `súng AK-105` ở EQUIPMENT_NAME — 18 nhóm train/dev,
  26 train/test, 4 dev/test.
- **Trung thực về giới hạn**: ghi rõ `production_model_run: false`,
  `repository_source_reviewed: false`, không tuyên bố F1.

### 2.2 Đối chiếu với chỗ đang thua thì hụt

180 câu V2 hụt số, map sang nhãn:

| | số câu |
|---|---|
| nhãn v4 **có** sinh thêm | 86 |
| nhãn v4 **không** đụng đến | 94 |

v4 sinh span cho 24/46 nhãn. 22 nhãn không được bù, gồm đúng những chỗ thua nặng:

```
doc so am        15/19 hụt   CARDINAL     — v4 không sinh
doc eta etd      11/15       ETA/ETD      — không sinh
doc mui gio      11/11       TIMEZONE     — không sinh
doc toa do       10/10       COORD        — không sinh
doc dia chi mang  9/11       ELECTRONIC   — không sinh
doc tien          8/17       MONEY        — không sinh
```

### 2.3 Ba lỗ thủng cụ thể

**a. VEHICLE_PLATE — khối thua lớn nhất (17/17 câu), v4 không sửa.**
```
thật : "ba không a gạch ngang một hai ba chấm bốn năm"  ->  30A-123.45
v4   : "hai chín a hai ba năm ba hai"                    ->  29A-23532
```
Đếm trong toàn bộ `v4_train.jsonl`: `"gạch ngang"` xuất hiện **0 lần**, `"chấm"`
**0 lần**. Cả 550 span biển số đều không có dấu chấm. Đúng lỗi thực-tế-hoá đã nêu
với v3, lặp lại nguyên vẹn.

**b. ACRONYM — chủ đề tệ nhất (F1 0,488) là bài toán khác hẳn cái v4 sinh.**
```
thật : "hệ thống nhận dạng tự động, hệ thống định vị toàn cầu…"  ->  AIS, GPS, ECDIS
v4   : "tê xê xê tê"                                              ->  TCCT
```
Thật là **tra cụm tiếng Việt → viết tắt tiếng Anh**; v4 sinh **đánh vần chữ cái**.
Hai việc không liên quan. TIMEZONE cũng vậy: "giờ phối hợp quốc tế" → `UTC`.

**c. Nhãn "có bù" vẫn bù quá hẹp.** Toàn bộ MEASURE/DISTANCE/DEPTH/DRAFT của v4
chỉ dùng đúng một đơn vị là `m`. Bản gốc thật có `6,5 km`, `850 m²`, `2.400 m³`.
`"ki lô mét"`, `"mét vuông"`, `"mét khối"`: 0 lần trong v4.

### 2.4 Hai mâu thuẫn nội tại

- **Không câu nào kết thúc bằng dấu chấm** — 6.760/6.760. Có phẩy (2.985) và hỏi
  (612), nhưng PERIOD bằng không. Trộn thẳng vào train là dạy đầu dấu câu rằng câu
  không bao giờ có chấm cuối, đúng chỗ đã tốn công chỉnh trọng số (xem
  README_ITN_V2_TRANG_THAI mục 8.1). Lỗi câm — không validator nào của gói bắt được.
- **Phê phán EQUIPMENT_NAME lệch contract ở §2.1 rồi tự lặp lại**: `hamburg`,
  `liverpool`, `darwin` gán EQUIPMENT_NAME (nghĩa "tên riêng con tàu"), đồng thời
  `"địa danh liverpool"` lại gán FOREIGN_NAME — trong khi contract của chính gói
  định nghĩa LOCATION_NAME = "địa danh". v4 sinh **0 span LOCATION_NAME**.
- PORT_CODE bị từ chối sinh vì "thiếu registry được xác minh". Lý do không đứng
  vững: dạng nói là đánh vần thuần ("vê en hát pê hát" → `VNHPH`), dạy được mà
  không cần registry thật, và UN/LOCODE vốn công khai.

### 2.5 Nên làm gì với gói này

**Lấy ngay:** hợp đồng 46 nhãn (`schema/label_contract.csv`), phát hiện leakage
theo họ thực thể, 540 câu all-O, và cách tách hai tầng `normalization_family` /
`semantic_role`.

**Chưa trộn vào train.** 26/46 nhãn còn ở trạng thái `PROPOSED_REVIEW_BEFORE_TRAIN`;
chính gói cũng ghi "không trộn tự động v2+v3+v4". Nếu trộn, tối thiểu phải vá dấu
chấm cuối câu trước.

**Bốn việc cần sinh thêm, xếp theo lợi ích đo được:**

1. Sinh lại VEHICLE_PLATE có `gạch ngang` + `chấm`, đích dạng `30A-123.45` — 17 câu.
2. Mở nhãn cho **cụm tiếng Việt → viết tắt** (AIS/GPS/ECDIS/ARPA/VHF/UTC/LT) —
   chủ đề tệ nhất, hiện không nhãn nào phụ trách.
3. Sinh COORD (`mười độ hai mươi lăm phút ba mươi giây bắc` → `10°25′30″N`) và
   ELECTRONIC dạng nói thật (`vê kép vê kép vê kép chấm…`) — 19 câu.
4. Mở rộng đơn vị MEASURE ra `km`/`m²`/`m³`, thêm dấu phân nhóm nghìn.

---

## 3. Điều rút ra chung

Hai kết quả trên chỉ về cùng một chỗ: **nút thắt không nằm ở kiến trúc mà ở độ
phủ và độ thật của dạng nói.** V1 kiến trúc thô sơ hơn nhưng thắng ở khôi phục số
vì nó chuyển thẳng tay; v4 dữ liệu sạch hơn nhiều nhưng không chạm vào những kiểu
đang hỏng. Việc tiếp theo đáng làm nhất vẫn là nới ngưỡng theo lớp cho nhóm số, và
sinh dạng nói đúng như bộ test thật phát ra — không phải đổi backbone.
