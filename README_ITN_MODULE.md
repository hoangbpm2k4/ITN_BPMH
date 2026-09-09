# Module ITN + Chấm câu (PhoBERT) — Tài liệu kiến trúc

_Chốt ngày 2026-09-06. Mô tả đúng hiện trạng, không phải đề xuất._
_Mọi tỷ lệ đo trên tập huấn luyện hiện hành; mọi ví dụ chuyển đổi là kết quả chạy thật._

Bản rút gọn cho người ngoài nhóm: https://claude.ai/code/artifact/6b035c03-97f4-439a-8f77-9373fcaa8298
Tài liệu liên quan: [README.md](README.md) (baseline 23/07/2026) · [V12_PLAN.md](V12_PLAN.md) (nhánh fusion âm thanh — mô hình khác)

---

## 1. Module làm gì

Mô hình nhận dạng tiếng nói được huấn luyện để nghe ra âm, nên thứ nó trả về là chuỗi
chữ đúng như người phát âm: chữ thường, không dấu câu, số viết bằng chữ. Module này
biến chuỗi đó thành văn bản dạng viết.

```
vào : lúc mười bốn giờ ngày mười lăm tháng tám biên đội phát hiện
      hai chiếc su ba mươi tại mười chín độ hai mươi phút bắc
ra  : Lúc 14 giờ 15/08 biên đội phát hiện 2 chiếc SU-30 tại 19°20'N.
```

Module nằm ngay sau ASR, trước mọi bước phân tích nội dung. Nó đảm nhận bốn việc trong
một lần chạy: chuẩn hoá ngược (ITN), viết hoa, đặt dấu câu, và tra ký hiệu chuyên ngành.

---

## 2. Nguyên tắc: mô hình phân loại, luật biến đổi

Module không dùng mô hình sinh chữ để viết lại câu. Nó tách làm hai việc tách bạch.

**Mô hình quyết định *cái gì*.** Với mỗi từ, mô hình gắn một nhãn nội dung (từ này thuộc
một con số, một ngày tháng, một tên khí tài, hay không thuộc gì cả) và một nhãn dấu câu
(sau từ này có phẩy, chấm, hỏi hay không). Mô hình không sinh ra ký tự nào.

**Luật cố định quyết định *viết ra sao*.** Biết cụm "a ka bốn bảy" là tên vũ khí rồi thì
một bộ luật viết tay chuyển nó thành `AK-47`. Luật là mã nguồn và bảng tra, chạy giống
hệt nhau mỗi lần, và mọi phép biến đổi đều truy ngược được về một dòng luật cụ thể.

Lý do đánh đổi: mô hình sinh chữ có thể tạo ra một con số không có trong âm thanh mà
không để lại dấu vết. Với toạ độ, tần số hay số hiệu, một giá trị bịa nguy hiểm hơn
nhiều so với việc giữ nguyên dạng nói.

---

## 3. Kiến trúc mô hình

### 3.1 Đầu vào và cách gióng nhãn

Câu được tách thành từ bằng khoảng trắng — không dùng bộ tách từ ghép tiếng Việt. Mỗi từ
sau đó được cắt thành các mảnh subword theo bộ từ vựng BPE của PhoBERT (66.119 mục, mảnh
cuối của mỗi từ mang hậu tố `</w>`, chính là dấu hiệu để ghép ngược lại về từ ở bước decode).

Điểm khác biệt đáng chú ý so với cách dùng PhoBERT thông thường: **chỉ số vị trí được
đánh theo từ, không theo mảnh**. Mọi mảnh của từ thứ *i* đều mang cùng một `position_id`
bằng *i+1*. Một từ bị cắt thành ba mảnh vẫn chiếm đúng một vị trí trong mắt mô hình, nên
mô hình luôn nhìn câu theo đơn vị từ.

Về nhãn: chỉ mảnh **đầu tiên** của mỗi từ được gán nhãn thật; các mảnh sau nhận giá trị
bỏ qua nên không đóng góp vào hàm mất mát. Lúc suy luận, nhãn của cả từ cũng lấy từ mảnh
đầu — nhất quán với lúc huấn luyện.

Độ dài tối đa 256 mảnh, cắt cứng, không có cửa sổ trượt. Câu dài hơn bị mất phần đuôi.

### 3.2 Backbone

PhoBERT-base-v2, 12 tầng, chiều ẩn 768, nạp từ bản sao cục bộ trong
`runtime_phobert_asr_bilstm/models/phobert_hf`. Trọng số gốc bị checkpoint warm-start
ghi đè khi huấn luyện.

Mô hình gọi thẳng hai thành phần `embeddings` và `encoder` của RoBERTa thay vì gọi hàm
forward của thư viện, để tránh phần kiểm tra kích thước của HuggingFace. Mặt nạ attention
dạng nhị phân được chuyển sang dạng cộng trước khi đưa vào encoder.

Cấu hình hiện tại chạy **attention hai chiều đầy đủ**: mọi từ nhìn thấy mọi từ, chỉ còn
mặt nạ padding. Nhánh streaming (chia khối 8/12/16 mảnh với ngữ cảnh phải 2/4) vẫn còn
trong mã nguồn nhưng đã tắt từ phiên bản v11 — module hướng tới xử lý câu dài ngoại tuyến.

### 3.3 Sau encoder thì qua lớp gì

Đầu ra của encoder là một vector 768 chiều cho mỗi mảnh. Từ đó tới nhãn chỉ có **một
tầng Dropout và một phép chiếu tuyến tính cho mỗi đầu ra**:

```
vector từ encoder (768 chiều)
        │
   Dropout (p = 0,1)
        │
        ├──►  Linear 768 → 15   : nhãn nội dung (ITN)
        └──►  Linear 768 → 4    : nhãn dấu câu
```

Không có CRF. Không có BiLSTM, Transformer hay bất kỳ tầng trung gian nào. Không có
activation giữa chừng. Hai đầu ra chạy **song song**, cùng đọc chung một vector, độc lập
hoàn toàn với nhau, phân loại ở mức từng token.

Softmax nằm ẩn trong hàm mất mát khi huấn luyện; lúc suy luận lấy argmax trực tiếp trên
logits. Không có beam search, **không có ràng buộc nào giữa nhãn của các từ liền kề** —
mỗi từ được quyết định độc lập, và việc gom cụm hoàn toàn do bước hậu xử lý suy ra sau.

Đầu ra dấu câu chỉ được tạo khi bật cờ tương ứng; nếu tắt, mô hình chạy chế độ chỉ-ITN.

> Cần phân biệt: mô hình BiLSTM 1809 chiều hay được nhắc trong các báo cáo v11/v12 nằm
> trong thư mục `EfficientPunct/`. Đó là nhánh chấm câu bằng fusion âm thanh–văn bản,
> một mô hình khác. Module trong tài liệu này thuần văn bản.

### 3.4 Hàm mất mát

Nhánh ITN dùng cross-entropy có trọng số theo lớp. Nhánh dấu câu dùng focal loss
(gamma = 2,0) đặt trên cross-entropy đã có trọng số — thiết kế này để bù cho việc 94% số
từ mang nhãn "không có dấu câu". Tổng mất mát là tổng đơn giản của hai nhánh, hệ số 1,0
cho mỗi bên.

Trọng số lớp cho nhánh dấu câu khi tính mất mát: không-dấu 2,0 · phẩy 1,5 · chấm 2,0 ·
hỏi 10,0. Trọng số cho nhánh ITN nằm ở bảng mục 4.1.

Ngoài trọng số trong hàm mất mát, tập huấn luyện còn được **lấy mẫu có trọng số**: câu
chứa nhãn hiếm được rút ra nhiều lần hơn. Riêng dấu chấm có logic theo vị trí, vì dấu
chấm cuối câu quá dễ so với dấu chấm giữa đoạn:

| Trường hợp của câu | Trọng số lấy mẫu |
|---|---|
| Không có dấu chấm nào | 1,0 |
| Chỉ có dấu chấm cuối câu | 6,0 |
| Có từ hai dấu chấm trở lên | 20,0 + 4,0 cho mỗi dấu chấm (tối đa 6) |
| Có dấu chấm ở **giữa** đoạn | 35,0 + 6,0 mỗi dấu chấm giữa (tối đa 5) + 4,0 mỗi dấu chấm |

Trọng số cuối cùng của một câu là giá trị lớn nhất trong ba nguồn: nhãn ITN hiếm nhất,
nhãn dấu câu hiếm nhất, và điểm dấu chấm theo vị trí.

### 3.5 Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---|
| Tối ưu | AdamW, learning rate 1e-5, weight decay 0,01 |
| Lịch learning rate | Tuyến tính, warmup 5% tổng số bước, cập nhật theo bước |
| Batch | 32 · 15 epoch · gradient clip 1,0 |
| Độ chính xác số | bf16-mixed trên GPU |
| Warm start | `checkpoints/warm_start_v11_merged.ckpt` |
| Tập huấn luyện | 244.688 câu / 23.473.819 từ |
| Tập kiểm định | 61.191 câu |
| Chọn checkpoint | Theo F1 dấu câu trên tập kiểm định, giữ 3 bản tốt nhất |
| Số worker nạp dữ liệu | 2 — đã giảm từ 4 vì máy 23 GB RAM bị OOM khi worker fork |

Hai biến môi trường: `SMOKE=1` chạy thử 30 batch huấn luyện và 20 batch kiểm định trong
1 epoch; `RESUME=1` tiếp tục từ `checkpoints/last.ckpt` giữ nguyên trạng thái optimizer.

---

## 4. Bộ nhãn và chuẩn chuyển đổi

### 4.1 Mười lăm nhãn nội dung

Mười lăm nhãn phục vụ **ba loại việc khác hẳn nhau**. Cách chia này quan trọng vì ba nhóm
có cơ chế nhận diện, mức độ khó và tỷ lệ dữ liệu rất chênh lệch.

| Nhãn | Việc phải làm | Tỷ lệ từ | Trọng số | Bộ chuyển đổi |
|---|---|---|---|---|
| `O` | không biến đổi | 83,07 % | 0,1 | — |
| **Nhóm 1 — chỉ viết hoa** | | **8,12 %** | | |
| `CASE` | tên riêng, hoa mọi từ | 5,35 % | 1,0 | bộ viết hoa, chế độ `all` |
| `TITLE` | chức danh, hoa chữ đầu cụm | 1,22 % | 5,0 | bộ viết hoa, chế độ `title` |
| `UNIT` | phiên hiệu đơn vị | 1,13 % | 5,0 | bộ viết hoa, chế độ `title` |
| `RANK` | quân hàm | 0,42 % | 20,0 | bộ viết hoa, chế độ `title` |
| **Nhóm 2 — đổi chữ thành số** | | **8,00 %** | | |
| `NUM` | số các loại | 4,41 % | 1,5 | bộ số |
| `DATE` | ngày tháng năm | 2,89 % | 5,0 | bộ ngày tháng |
| `COORD` | toạ độ | 0,65 % | 20,0 | **không có** — xem mục 7.4 |
| `MEASURE` | đơn vị đo | 0,03 % | 120,0 | bộ đơn vị |
| `DATE_QUARTER` | quý | 0,02 % | 150,0 | bộ quý |
| **Nhóm 3 — tra danh mục** | | **0,80 %** | | |
| `EQUIP` | khí tài | 0,53 % | 20,0 | bộ khí tài (dùng chung với `WEAPON_NAME`) |
| `DOC` | số hiệu văn bản | 0,13 % | 25,0 | bộ mã văn bản |
| `LEGAL_DOC` | ký hiệu văn bản pháp quy | 0,06 % | 100,0 | bộ pháp quy |
| `FOREIGN_NAME` | tên riêng nước ngoài | 0,05 % | 100,0 | bộ tên nước ngoài |
| `WEAPON_NAME` | tên vũ khí | 0,03 % | 150,0 | bộ khí tài |

Nhãn **phẳng, không có tiền tố `B-`/`I-`**. Ranh giới cụm được suy ra bằng quy tắc "các
từ liền nhau mang cùng nhãn thì thuộc cùng một cụm", cộng thêm một luật cắt cụm khi gặp
dấu câu — chính luật cắt này khiến bước ITN phụ thuộc vào đầu ra dấu câu.

Nhóm 3 đáng chú ý: khó nhất về mặt kỹ thuật nhưng chỉ chiếm 0,80% số từ, nên phải bù bằng
trọng số gấp hàng trăm lần. Trọng số này được khai báo trùng lặp ở hai nơi — một bản cho
hàm mất mát, một bản cho việc lấy mẫu — và hai bản hiện giống nhau.

Bốn loại từng được cân nhắc nhưng không đưa vào bộ nhãn, xử lý hoàn toàn bằng luật:
phần trăm, tiền tệ, tên tổ chức, và quý (dạng rời).

### 4.2 Bốn nhãn dấu câu

| Nhãn | Ký tự | Tỷ lệ từ | Trọng số mất mát | Trọng số lấy mẫu |
|---|---|---|---|---|
| `O` | (không có) | 94,27 % | 2,0 | 1,0 |
| `COMMA` | `,` | 3,65 % | 1,5 | 20,0 |
| `PERIOD` | `.` | 2,06 % | 2,0 | theo vị trí, xem mục 3.4 |
| `QUESTION` | `?` | 0,02 % | 10,0 | 120,0 |

Bảng ánh xạ ký tự trong mã nguồn còn khai báo thêm dấu chấm than và dấu hai chấm, nhưng
bộ nhãn không có hai loại đó nên chúng không bao giờ được kích hoạt.

### 4.3 Chuẩn chuyển đổi

Toàn bộ bảng dưới đây là kết quả chạy thật của bộ hậu xử lý, không phải ví dụ minh hoạ.

**Nhóm 1 — chỉ viết hoa**

| Nhãn | Dạng nói | Dạng viết |
|---|---|---|
| `CASE` | nguyễn phú trọng | Nguyễn Phú Trọng |
| `CASE` | việt nam | Việt Nam |
| `TITLE` | chủ tịch | Chủ tịch |
| `UNIT` | lữ đoàn công binh | Lữ đoàn công binh |
| `RANK` | thiếu tá | Thiếu tá |

**Nhóm 2 — đổi chữ thành số**

| Nhãn | Dạng nói | Dạng viết |
|---|---|---|
| `NUM` | hai trăm năm mươi | 250 |
| `NUM` | một nghìn không trăm hai mươi | 1.120 |
| `NUM` | hai trăm năm mươi triệu | 250.000.000 |
| `NUM` | ba phẩy năm | 3,5 |
| `NUM` | hai mươi lăm phần trăm | 25% |
| `NUM` | không ba một hai ba bốn năm sáu bảy tám | 0312345678 |
| `DATE` | ngày mười lăm tháng tám năm hai không hai lăm | 15/08/2025 |
| `DATE` | ngày mùng một tháng năm | 01/05 |
| `DATE` | tháng mười hai năm một nghìn chín trăm tám mươi | 12/1980 |
| `COORD` | mười chín độ hai mươi phút bắc | 19°20'N |
| `MEASURE` | ki lô mét | km |
| `MEASURE` | mét vuông | m² |
| `DATE_QUARTER` | quý ba | Quý III |

**Nhóm 3 — tra danh mục**

| Nhãn | Dạng nói | Dạng viết |
|---|---|---|
| `WEAPON_NAME` | a ka bốn bảy | AK-47 |
| `WEAPON_NAME` | ép mười sáu | F-16 |
| `WEAPON_NAME` | bê năm hai | B-52 |
| `EQUIP` | mi mười bảy | MI-17 |
| `EQUIP` | tê chín mươi | T-90 |
| `LEGAL_DOC` | nờ đê cê pê | NĐ-CP |
| `LEGAL_DOC` | nờ đê gạch cê pê | NĐ-CP |
| `LEGAL_DOC` | tê tê bê tê xê | TT-BTC |
| `DOC` | số mười hai xẹt hai không hai tư | Số 12/2024 |
| `FOREIGN_NAME` | oa sinh tơn | Washington |
| `FOREIGN_NAME` | mát cơ va | Moscow |
| `FOREIGN_NAME` | phây búc | Facebook |

Khác biệt cốt lõi giữa nhóm 2 và nhóm 3: nhóm 2 **phân tích cấu trúc** nên xử lý được cả
những cụm chưa từng gặp; nhóm 3 **tra bảng** nên thứ gì không có trong bảng sẽ đi qua
nguyên dạng nói mà không báo lỗi.

---

## 5. Quy tắc của từng bộ chuyển đổi

### 5.1 Bộ số — nền của bốn bộ khác

Đây là bộ được bốn bộ khác gọi lại, nên quy tắc của nó ảnh hưởng gián tiếp tới ngày
tháng, toạ độ, số hiệu và khí tài.

Bảng chữ số cơ sở có 14 mục, gồm cả các biến thể phát âm: `không · một · mốt · hai · ba ·
bốn · tư · năm · lăm · nhăm · sáu · bảy · tám · chín`. Bảng đơn vị có 5 mục: `trăm` ·
`nghìn` và `ngàn` (cùng giá trị) · `triệu` · `tỷ`.

Bộ số làm việc ở **hai chế độ, tự chọn theo nội dung**:

- **Đọc rời.** Nếu cụm không chứa từ khoá cấu trúc nào (`mười`, `mươi`, `trăm`, `nghìn`,
  `ngàn`, `triệu`, `tỷ`) thì nó nối thẳng từng chữ số. Đây là cách số điện thoại và số
  hiệu ra đúng: "không ba một hai ba" thành `03123` chứ không phải một giá trị đếm.
- **Đọc theo cấu trúc.** Ngược lại, nó cộng dồn theo ba mức (tổng, khối trăm, giá trị
  tạm) để dựng số đếm. Chế độ này cũng xử lý được lối đọc ghép năm: "một chín tám mươi"
  thành `1980`. Hai từ `lẻ` và `linh` bị bỏ qua khi tính.

Ba quy ước định dạng cố định: từ `phẩy` sinh ra dấu **phẩy** thập phân; từ `chấm` sinh ra
dấu **chấm** (dùng cho số phiên bản, số hiệu); cụm `phần trăm` bị tách ra thành hậu tố `%`.
Riêng nhãn `NUM` bật thêm dấu chấm phân nhóm nghìn, nên `250000000` được viết `250.000.000`.

### 5.2 Bộ ngày tháng

Nhận diện ba thành phần bằng ba từ khoá: phần ngày phải đứng sau `ngày`, `mùng` hoặc
`mồng`; phần tháng sau `tháng`; phần năm sau `năm`. Ngày và tháng được đệm số 0 cho đủ
hai chữ số, rồi nối bằng dấu gạch chéo.

Trước khi xử lý có một bước chống nhầm lẫn: cụm `tháng năm` ở cuối chuỗi được đổi thẳng
thành `tháng 05`, vì nếu không thì từ "năm" sẽ bị hiểu là từ khoá chỉ năm.

Phần năm có bốn luật riêng, xếp theo thứ tự ưu tiên: bốn chữ số đọc rời thì ghép thẳng;
`hai không` cộng đuôi 0–99 thành `20xx`; `một chín` cộng đuôi thành `19xx`; `hai nghìn`
hoặc `hai ngàn` (có thể kèm `không trăm`) cộng đuôi thành `20xx`.

**Điểm yếu đã biết:** nếu cụm không chứa từ khoá chỉ ngày thì phần ngày bị bỏ hoàn toàn.
Xem mục 7.3.

### 5.3 Bộ viết hoa

Hai chế độ. Chế độ `title` chỉ viết hoa chữ đầu của cả cụm — dùng cho chức danh, đơn vị,
quân hàm ("Trung úy", "Lữ đoàn công binh"). Chế độ `all` viết hoa chữ đầu của mọi từ —
dùng cho tên riêng ("Nguyễn Phú Trọng").

Cả hai chế độ đều tra trước một danh sách 11 từ viết tắt luôn viết hoa toàn bộ:
`UBND · HĐND · CP · KH · NQ · TT · BQP · CA · UAV · VNN · QDND`.

Chuỗi được chuẩn hoá Unicode NFC và thay gạch dưới bằng khoảng trắng trước khi xử lý.

### 5.4 Bộ tên nước ngoài

Bảng 53 cách đọc phiên âm ánh xạ về 32 tên chuẩn, chia ba mảng:

- **Khí tài, công nghệ:** đờrôn→Drone · patriốt, batriốt→Patriot · tômahốc→Tomahawk ·
  himác→Himars · bairắcta→Bayraktar · giavêlin→Javelin · leoôpát→Leopard ·
  hapun, habun→Harpoon · sờtalinh→Starlink
- **Văn phòng, mạng xã hội:** phâybúc, phâybút→Facebook · gugồmít→Google Meet ·
  cờrôsóptim, cờrôsốptim→Microsoft Teams · đétlai→Deadline · cayôeo, kayôeo→KOL ·
  laisờtrim→Livestream · pótcát, bótcát→Podcast · quốcsóp→Workshop · semina→Seminar
- **Địa danh, chính trị:** oasinhtơn, quasinhtơn→Washington · mátcơva→Moscow ·
  bốn biến thể xinggapo→Singapore · ucraina, ucờraina→Ukraine ·
  philíppin, philípbin→Philippines · inđônêxia→Indonesia · malaixia, malayxia→Malaysia ·
  campuchia, cambuchia→Cambodia · bốn biến thể xơun→Seoul · tôkiô, tôkyô→Tokyo ·
  isareo→Israel · bốn biến thể paléctin→Palestine · bờradiu→Brazil · átgentina→Argentina

Cơ chế khớp: nối các từ trong cụm thành **một chuỗi liền không khoảng trắng**, thay thế
theo bảng (ưu tiên khớp chuỗi dài trước), rồi tách lại thành từ dựa trên ranh giới
hoa–thường. Cách này chịu được việc ASR cắt cụm thành số lượng từ khác nhau, nhưng cũng
có nghĩa là mọi biến thể phát âm đều phải liệt kê sẵn thành một dòng riêng.

### 5.5 Bộ pháp quy

28 cách đọc ánh xạ về 7 ký hiệu. Phần lớn khối lượng nằm ở `NĐ-CP` với 16 biến thể, sinh
ra từ tổ hợp của ba trục: `cê` hay `xê`, `pê` hay `bê`, và cách đọc dấu gạch (không đọc,
`gạch`, `xẹc`, `xẹt`). Sáu ký hiệu còn lại: `QĐ` · `CT` (2 biến thể) · `CP` (4 biến thể) ·
`NQ` (2) · `TT-BTC` (2) · `TT`. Cùng cơ chế nối–thay–tách như bộ tên nước ngoài.

### 5.6 Bộ mã văn bản

Khác các bộ trên, bộ này **phân tích cấu trúc** chứ không tra nguyên cụm. Nó duyệt từng
từ và phân loại vào bốn nhóm: từ chỉ dấu phân cách (`xẹt`, `trên`, `phần`) sinh ra dấu
gạch chéo; từ chỉ dấu nối (`gạch`) sinh ra dấu gạch ngang; 10 tên chữ cái
(`quy→Q · hát→H · nờ→N · en→N · đê→Đ · cê→C · xê→C · pê→P · bê→B · tê→T`) sinh ra chữ cái
tương ứng; còn lại là chữ số thì gom vào bộ đệm rồi đưa cho bộ số.

Hai chuẩn hoá đầu vào: `xẹc` và `gạch chéo` đều quy về `xẹt`; từ `số` bị bỏ. Có thêm một
luật năm: bộ đệm bắt đầu bằng `hai không` cộng đuôi 0–99 được hiểu là năm 20xx — đây là
cách `số mười hai xẹt hai không hai tư` ra được `Số 12/2024`.

### 5.7 Bộ khí tài

Dùng chung cho cả `WEAPON_NAME` và `EQUIP`. Thuật toán: tìm từ chỉ số đầu tiên trong cụm,
cắt đôi thành phần tiền tố và phần số; tra tiền tố trong bảng 24 mục; đưa phần số cho bộ
số; ghép lại theo khuôn `TIỀN TỐ-số`.

24 tiền tố: `a ka→AK · xu, su→Su · míc→MiG · mi→Mi · bê→B · tê→T · át→S · áp→AP · en→N ·
mờ→M · đê→D · hát→H · vê→V · ép→F · quy→Q · ec→EC · ét→S · ích→X · xê→C · xá→X · ka→K ·
ca→C · em→M`.

Vì phần số đi qua bộ số nên bộ này xử lý được cả những số hiệu chưa từng gặp: chỉ cần
tiền tố có trong bảng, `su` cộng bất kỳ số nào cũng ra đúng. Cùng file còn có một danh
sách 15 mẫu số hiệu đầy đủ (Su-30, F-16, MiG-21, B-52…) nhưng danh sách đó không được
dùng ở đâu.

**Điểm yếu đã biết:** bước ghép cuối viết hoa toàn bộ tiền tố, xoá mất phần chữ thường mà
bảng tra đã ghi đúng. Xem mục 7.2.

### 5.8 Bộ đơn vị đo và bộ quý

Bộ đơn vị có 16 mục, khớp **toàn cụm sau khi bỏ hết khoảng trắng**: mi li mét→mm ·
xăng ti mét→cm · đề xi mét→dm · mét→m · ki lô mét→km · mi li gam→mg · gam→g ·
ki lô gam→kg · mi li lít→ml · lít→l · mét vuông→m² · mét khối→m³ · héc ta→ha · am pe→A ·
vôn→V · oát→W. Không khớp mục nào thì giữ nguyên dạng nói.

Bộ quý luôn lấy **từ thứ hai** của cụm và tra bảng 9 mục sang số La Mã
(`một`/`1`→I · `hai`/`2`→II · `ba`/`3`→III · `bốn`/`4`/`tư`→IV), rồi xuất theo khuôn
`Quý <số La Mã>`.

Ngoài ra còn một bộ tiền tệ với 4 bậc đơn vị và 7 ký hiệu tiền tệ (đồng→₫, đô→$, euro→€,
yên→¥, bảng anh→£) nhưng nó **không được gọi ở bất kỳ đâu** trong luồng hiện tại.

---

## 6. Luật quét toàn câu và bước dọn

Có những thứ không quyết định được khi chỉ nhìn một cụm, phải chờ ghép cả câu xong. Tổng
cộng khoảng **62 lượt thay thế bằng biểu thức chính quy chạy trên mỗi câu**, chia hai đợt.

### 6.1 Đợt một — dựng dạng viết

| Luật | Ví dụ |
|---|---|
| Dựng toạ độ từ độ – phút – giây – hướng | mười chín độ hai mươi phút bắc → 19°20'N |
| Dán số vào đơn vị đo (14 đơn vị) | 20 km → 20km |
| Ghép ký hiệu khí tài bị tách rời | A KA 47 → AK-47 |
| Hoàn tất các dạng phần trăm còn sót | 3 phẩy 5 phần trăm → 3,5% |
| Viết hoa chữ chỉ hướng sau ký hiệu độ | 19°20'n → 19°20'N |
| Xoá khoảng trắng thừa trước dấu câu và ký hiệu | 19 ° 20 ' N → 19°20'N |

### 6.2 Đợt hai — dọn lỗi mô hình hay mắc

Ba nhóm luật, chạy trên văn bản đã hoàn chỉnh.

**Dọn nhiễu hội thoại.** Tách tiếng đệm bị dính vào từ sau (`ờhọc` → `ờ học`, với 5 tiếng
đệm `à · ờ · ừ · ừm · ờm`). Khôi phục trường hợp cụm "em không" bị nhận nhầm thành mã hiệu
`M-0`. Hạ chuỗi tiếng cười bị viết hoa như chữ cái riêng. Hạ chữ cái đơn viết hoa đứng
trước dấu câu, có loại trừ để không đụng tới mã hiệu thật như `H-21` hay `B-52`.

**Gỡ dấu phẩy đặt sai.** 26 luật trên 15 cụm liên từ hay bị mô hình chèn phẩy vào giữa
hoặc chèn phẩy ngay sau: `tại vì · bởi vì · cho nên · thí dụ · ví dụ · nhưng mà · có nghĩa · tức là ·
đúng là · khi mà · để mà · tới khi · từ nãy · nói rằng · câu hỏi rằng`.

**Hạ viết hoa thừa.** 28 từ hay bị mô hình gắn nhãn tên riêng nhầm sẽ bị hạ về chữ thường
khi đứng sau dấu câu: `cái · câu · con · lớp · tôi · mình · bạn · ông · bà · thầy · cô ·
học · cầm · hậu · đúng · nhưng · mường · trích · há · may · đó · đại · trung · ở · thí ·
ra · từ · ngh`. Luật này chạy lặp ba vòng. Kèm theo là 7 cụm cố định bị ép về chữ thường
— đây là các bản vá theo dữ liệu quan sát được, không phải luật ngôn ngữ.

> Bốn luật phần trăm và hai luật ký hiệu khí tài **chạy hai lần**, một lần ở mỗi đợt.

---

## 7. Giới hạn đã kiểm chứng

Bảy mục dưới đây đo bằng cách chạy trực tiếp bộ hậu xử lý ngày 06/09/2026, không phải suy đoán.

### 7.1 Quân hàm bắt đầu bằng "trung" hoặc "đại" bị hạ chữ thường

```
gặp thiếu tá       →  Gặp Thiếu tá        ✔
gặp đại tá         →  Gặp đại tá          ✘
đồng chí trung úy  →  Đồng chí trung úy   ✘
```

Bộ viết hoa xử lý đúng, nhưng luật dọn ở đợt hai hạ chúng xuống vì "trung" và "đại" nằm
trong danh sách 28 từ hay bị viết hoa thừa. Ảnh hưởng phần lớn quân hàm tiếng Việt:
Trung úy, Trung tá, Trung tướng, Trung sĩ, Đại úy, Đại tá, Đại tướng. Đây là xung đột
giữa luật phân loại theo nhãn và luật quét toàn câu chạy sau nó.

### 7.2 Ký hiệu khí tài luôn ra chữ hoa toàn bộ

```
su ba mươi   →  SU-30    (chuẩn: Su-30)
míc hai mốt  →  MIG-21   (chuẩn: MiG-21)
```

Bảng tra tiền tố ghi đúng cách viết chuẩn `Su`, `MiG`, `Mi`, nhưng bước ghép cuối viết hoa
toàn bộ tiền tố nên phần chữ thường bị xoá. Công sức phân biệt hoa–thường trong bảng tra
hiện không có tác dụng.

### 7.3 Ngày bị mất khi cụm không chứa từ khoá chỉ ngày

```
ngày hai mươi tám tháng bốn  →  28/04   ✔
     hai mươi tám tháng bốn  →  04      ✘  mất số 28, không báo lỗi
```

Bộ ngày tháng nhận diện phần ngày bằng cách tìm từ khoá `ngày`/`mùng`/`mồng` **bên trong**
cụm, nhưng quy ước gắn nhãn lại để từ "ngày" nằm **ngoài** cụm.

Quét 60.000 câu huấn luyện: trong 22.419 cụm `DATE`, có 17.903 cụm (79,9%) không chứa từ
khoá, và **5.369 cụm (23,9%) có đủ cả ngày lẫn tháng nhưng thiếu từ khoá** — tức rơi đúng
vào trường hợp mất ngày. Ví dụ thật trong dữ liệu: "đến |hai mươi tám tháng bốn|",
"chiều |mười lăm tháng bốn|", "sáng |mười sáu tháng chín|".

Đây là lỗi lệch pha giữa quy ước gắn nhãn và giả định của bộ chuyển đổi, không phải lỗi
của mô hình. Mô hình dự đoán hoàn hảo thì vẫn sai gần một phần tư số ngày tháng.

### 7.4 Nhãn toạ độ không ảnh hưởng kết quả

```
gắn nhãn COORD  →  19°20'N
không gắn nhãn  →  19°20'N
```

Toạ độ được dựng hoàn toàn bằng luật quét toàn câu, chạy bất kể mô hình gắn nhãn gì.
Nhãn `COORD` không có nhánh xử lý riêng ở bước biến đổi nên rơi vào nhánh mặc định là giữ
nguyên dạng nói. Nhãn này chiếm 0,65% số từ — nhiều hơn cả `EQUIP` và `RANK` — và mang
trọng số huấn luyện 20,0, nhưng hiện không đóng góp gì vào kết quả.

### 7.5 Không viết hoa chữ đầu câu sau dấu chấm

```
Hôm nay trời đẹp. chúng tôi đi học.
```

Chỉ từ đầu tiên của cả đoạn được viết hoa. Đây là hệ quả của một thay đổi dữ liệu có chủ
đích: khi tạo tập v11, 260.006 nhãn viết hoa ở đầu câu đã bị gỡ bỏ, vì bộ sinh nhãn cũ gắn
nhãn tên riêng cho mọi từ viết hoa kể cả từ chỉ hoa do quy tắc chính tả, khiến mô hình học
viết hoa bừa. Tài liệu của bước lọc ghi rõ giả định: việc viết hoa đầu câu "sẽ do bước
hoa-theo-dấu-chấm đảm nhận". **Bước đó hiện chưa được viết.**

### 7.6 Ranh giới cụm phụ thuộc đầu ra dấu câu

Vì nhãn không có tiền tố `B-`/`I-`, hai thực thể cùng loại đứng liền nhau sẽ dính thành
một cụm. Luật vá hiện tại là cắt cụm khi gặp dấu câu — nghĩa là chất lượng gom cụm ITN
phụ thuộc vào độ chính xác của đầu ra dấu câu. Không thể tách hai nhánh này ra khi triển
khai.

### 7.7 Trọng số lớp đang tính theo phân bố dữ liệu cũ

Chú thích tần suất đặt cạnh bảng trọng số là số liệu của tập v10, trước bước lọc nhãn
viết hoa. Bộ trọng số hiện hành được chỉnh theo cột giữa:

| Nhãn | Ghi trong mã nguồn | Dữ liệu v11 thực tế | Chênh |
|---|---|---|---|
| `O` | 78,62 % | **83,07 %** | +4,45 |
| `CASE` | 11,27 % | **5,35 %** | −5,92 |
| `DATE` | 1,01 % | **2,89 %** | +1,88 |
| `NUM` | 4,76 % | 4,41 % | −0,35 |
| `COORD` | 0,51 % | 0,65 % | +0,14 |
| `TITLE` | 1,39 % | 1,22 % | −0,17 |
| `UNIT` | 1,09 % | 1,13 % | +0,04 |

---

## 8. Nơi tìm trong mã nguồn

| Việc | File |
|---|---|
| Khai báo nhãn, siêu tham số | [config.py](config.py) |
| Nạp dữ liệu, cắt subword, gióng nhãn, trọng số lấy mẫu | [loader.py](loader.py) |
| Kiến trúc mô hình và hàm mất mát | [model.py](model.py) |
| Vòng huấn luyện | [train.py](train.py) |
| Suy luận một câu | [inference.py](inference.py) |
| Gom cụm, điều phối theo nhãn, luật quét toàn câu | [post_process.py](post_process.py) |
| Chín bộ chuyển đổi theo lớp | [Expander/](Expander/) |
| Lọc nhãn viết hoa đầu câu (tạo tập v11) | [scripts/filter_case_sentence_start.py](scripts/filter_case_sentence_start.py) |

Toàn bộ tầng hậu xử lý — bộ điều phối và tám bộ chuyển đổi — được **nhân bản** sang
[runtime_phobert_asr_bilstm/](runtime_phobert_asr_bilstm/). Hai bản hiện giống hệt nhau
từng byte, nhưng là hai bản độc lập: sửa một bên không tự đồng bộ bên kia.

Trong module còn một số phần đã chết nhưng chưa gỡ: bộ tiền tệ không được gọi ở đâu; danh
sách 15 mẫu số hiệu khí tài khai báo nhưng không dùng; hai nhãn dấu chấm than và hai chấm
không có trong bộ nhãn; nhánh gộp subword theo ký hiệu cũ; nhánh mặt nạ streaming đã tắt.

---

## 9. Cách chạy

```bash
PY=/home/hoangbpm/Intekcom/.venv_nemo/bin/python

# huấn luyện
$PY train.py
SMOKE=1 $PY train.py          # chạy thử 30 batch
RESUME=1 $PY train.py         # tiếp từ checkpoints/last.ckpt

# suy luận một câu — sửa đường dẫn checkpoint trong phần __main__ trước khi chạy
$PY inference.py
```

Để thử riêng tầng chuyển đổi mà không cần mô hình, không cần GPU: khởi tạo bộ hậu xử lý
rồi gọi hàm decode với ba danh sách song song — danh sách từ (mỗi từ kèm hậu tố `</w>`),
danh sách nhãn nội dung, danh sách nhãn dấu câu. Cách này tách được lỗi của mô hình khỏi
lỗi của luật, và là cách toàn bộ mục 7 được kiểm chứng.

```bash
$PY -c "from post_process import ITNPostProcessor; p=ITNPostProcessor(); \
print(p.decode(['a</w>','ka</w>','bốn</w>','bảy</w>'], ['WEAPON_NAME']*4, ['O','O','O','PERIOD']))"
# -> AK-47.
```
