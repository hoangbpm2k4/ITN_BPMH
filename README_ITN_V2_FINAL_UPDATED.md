# ITN V2 — FINAL IMPLEMENTATION SPECIFICATION
**Domain:** Vietnamese Maritime + Military ASR Post-processing  
**Baseline backbone:** `vinai/phobert-base-v2`  
**Status:** FINAL V2 baseline spec for implementation  
**Target test catalog:** 33 ITN test groups currently defined

---

# 1. Objective

The module receives **spoken-form ASR text** and converts detected semantic spans into canonical written form.

Example:

```text
Input:
phát hiện xu ba lăm ở hướng không chín không tốc độ mười hai hải lý một giờ

Output:
Phát hiện Su-35 ở hướng 090°, tốc độ 12 kn.
```

Design priorities:

1. deterministic normalization;
2. exact preservation of critical numeric values;
3. explicit span boundaries;
4. domain-specific semantic types;
5. confidence + validation before emitting normalization;
6. safe fallback to raw spoken form;
7. complete debugging trace for every transformed span.

The neural model MUST NOT autoregressively generate final written text.

---

# 2. Final architecture

```text
Raw ASR transcript
        |
        v
Vietnamese word segmentation
+ reversible mapping to raw ASR tokens
        |
        v
PhoBERT tokenizer
        |
        v
PhoBERT-base-v2
(native tokenizer + native positional encoding)
        |
        v
Subword hidden states
        |
        v
Mean pooling -> one vector per model word
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
Boundary head             Type head           Punctuation head
Linear -> CRF             Linear -> CE         Linear -> Focal
B / I / E / S / O        46 semantic types    O/,/./?
        |                      |
        +----------+-----------+
                   |
                   v
          Typed semantic spans
                   |
                   v
          Typed normalizers
                   |
                   v
          Domain validators
                   |
                   v
        Confidence / validity gate
                   |
                   v
       Casing + punctuation rendering
                   |
                   v
            Final written text
```

Critical rule:

> **Punctuation MUST NEVER determine ITN span boundaries.**

---

# 3. PhoBERT input contract

Use `vinai/phobert-base-v2`.

PhoBERT expects Vietnamese word-segmented text. V2 therefore keeps two token layers:

```text
raw ASR tokens
        ↓
Vietnamese word segmentation
        ↓
model words
        ↓
PhoBERT tokenizer
```

Example:

```text
raw:
[hải] [lý]

segmented:
[hải_lý]

mapping:
hải_lý -> raw_token[5:7]
```

The mapping from model word back to raw ASR token span MUST be reversible.

Typed normalizers always receive the **raw ASR text span**, not the segmented form.

## 3.1 Positional encoding

Do not manually override PhoBERT position IDs.

Wrong:

```text
subword1 -> position 12
subword2 -> position 12
subword3 -> position 12
```

Correct:

```text
subword1 -> position 12
subword2 -> position 13
subword3 -> position 14
```

Use native PhoBERT positional embeddings.

## 3.2 Word representation

If one model word splits into multiple PhoBERT subwords:

```text
word_i
 -> subword_a
 -> subword_b
 -> subword_c
```

run all subwords through PhoBERT first, then:

```text
word_hidden_i =
mean(hidden_a, hidden_b, hidden_c)
```

Mean pooling is the required V2 baseline.

Do not classify only the first subword.

---

# 4. Final semantic taxonomy — 46 types

`O` means no ITN transformation and is not counted among the 46 semantic types.

## 4.1 Core numeric / generic ITN

```text
1.  CARDINAL
2.  ORDINAL
3.  DIGIT_SEQ
4.  DECIMAL
5.  FRACTION
6.  PERCENT
7.  RANGE
8.  RATIO
9.  MONEY
10. MEASURE
11. VERSION
```

Examples:

```text
hai mươi lăm
-> CARDINAL -> 25

thứ hai
-> ORDINAL -> thứ 2

không ba một hai
-> DIGIT_SEQ -> 0312

ba phẩy năm
-> DECIMAL -> 3,5

một phần hai
-> FRACTION -> 1/2

hai mươi lăm phần trăm
-> PERCENT -> 25%

một trên mười
-> RATIO -> 1:10

hai triệu đồng
-> MONEY -> 2.000.000 đồng

hai mươi bốn vôn
-> MEASURE -> 24 V

phiên bản một chấm hai chấm ba
-> VERSION -> 1.2.3
```

Negative number is NOT a separate semantic type.

`âm`, `trừ`, etc. are numeric modifiers handled inside the relevant parser.

---

## 4.2 Date / time

```text
12. DATE
13. TIME
14. TIMEZONE
15. DURATION
16. ETA
17. ETD
18. QUARTER
```

Examples:

```text
mười lăm tháng tám năm hai không hai sáu
-> DATE -> 15/08/2026

sáu giờ bốn mươi lăm
-> TIME -> 06:45

u tê xê cộng bảy
-> TIMEZONE -> UTC+7

ê tê a tám giờ mười lăm
-> ETA -> ETA 08:15

ê tê đê mười sáu giờ bốn mươi
-> ETD -> ETD 16:40

quý ba
-> QUARTER -> Quý III
```

The DATE parser MUST NOT require the token `ngày` to be inside the detected span.

---

## 4.3 Maritime / navigation

```text
19. COORD
20. HEADING
21. BEARING
22. SPEED
23. DISTANCE
24. DEPTH
25. DRAFT
26. FREQUENCY
27. CHANNEL
```

Examples:

```text
mười độ hai mươi lăm phút bắc
-> COORD -> 10°25'N

không chín không
-> HEADING -> 090°

mười hai hải lý một giờ
-> SPEED -> 12 kn

tám phẩy hai mét
-> DRAFT -> 8,2 m
```

`COORD` MUST have its own typed normalizer.

Specific maritime classes take precedence over generic `MEASURE` when context supports the specialized meaning.

---

## 4.4 Structured identifiers / electronic forms

```text
28. MMSI_ID
29. IMO_ID
30. CALLSIGN
31. VESSEL_ID
32. PORT_CODE
33. DOCUMENT_ID
34. LEGAL_DOC_ID
35. TELEPHONE
36. ELECTRONIC
37. VEHICLE_PLATE
38. ADDRESS
```

Examples:

```text
hai ba bốn năm sáu bảy tám chín không
-> MMSI_ID -> 234567890

số mười hai xẹt hai không hai tư
-> DOCUMENT_ID -> Số 12/2024

vê en hát pê hát
-> PORT_CODE -> VNHPH

không chín không tám một hai ba bốn năm sáu
-> TELEPHONE -> 0908123456
```

`ELECTRONIC` contains deterministic subtypes:

```text
EMAIL
URL
IPV4
IPV6
MAC
```

Do not create separate neural semantic classes for those subtypes in V2.

---

## 4.5 Entity / canonicalization

```text
39. EQUIPMENT_ID
40. EQUIPMENT_NAME
41. FOREIGN_NAME
42. PERSON_NAME
43. LOCATION_NAME
44. ACRONYM
45. MARITIME_TERM
46. RANK
```

Examples:

```text
xu ba lăm
-> EQUIPMENT_ID -> Su-35

míc hai chín
-> EQUIPMENT_ID -> MiG-29

pa tri ốt
-> EQUIPMENT_NAME -> Patriot

vê hát ép
-> ACRONYM -> VHF

nguyễn văn hùng
-> PERSON_NAME -> Nguyễn Văn Hùng

hải phòng
-> LOCATION_NAME -> Hải Phòng

đét xờ lâu a hét
-> MARITIME_TERM -> Dead slow ahead

đại tá
-> RANK -> Đại tá
```

`MARITIME_TERM` is for canonical maritime commands/signals/phrases such as:

```text
MAYDAY
PAN-PAN
SECURITE
Dead slow ahead
Full astern
Starboard
Port
safe speed
alter course
```

---

# 5. Mapping from current 33 test groups

The current test catalog maps to V2 as follows.

| Test group | Main V2 type / handling |
|---|---|
| 1. Đơn vị đo | `MEASURE`, or specialized `SPEED/DISTANCE/DEPTH/DRAFT/FREQUENCY` |
| 2. Phần trăm | `PERCENT` |
| 3. Tiền | `MONEY` |
| 4. Ngày | `DATE` |
| 5. Giờ | `TIME` |
| 6. Múi giờ | `TIMEZONE` |
| 7. ETA / ETD | `ETA`, `ETD` |
| 8. Tọa độ | `COORD` |
| 9. Nhận dạng tàu | `IMO_ID`, `MMSI_ID`, `CALLSIGN`, `VESSEL_ID` |
| 10. Từ viết tắt hàng hải | `ACRONYM` |
| 11. VHF / kênh | `ACRONYM` + `CHANNEL` |
| 12. Lệnh điều động / English maritime | `MARITIME_TERM` + domain numeric type if needed |
| 13. Tín hiệu khẩn cấp | `MARITIME_TERM`; acronym components remain `ACRONYM` |
| 14. Email | `ELECTRONIC:EMAIL` |
| 15. Website | `ELECTRONIC:URL` |
| 16. Địa chỉ mạng | `ELECTRONIC:IPV4/IPV6/MAC` |
| 17. Điện thoại | `TELEPHONE` |
| 18. Văn bản pháp lý | `DOCUMENT_ID`, `LEGAL_DOC_ID` |
| 19. Tên người | `PERSON_NAME` |
| 20. Tỉnh / thành phố | `LOCATION_NAME`, `ACRONYM` when appropriate |
| 21. Tên nước ngoài | `FOREIGN_NAME` |
| 22. Quý / Roman | `QUARTER` |
| 23. No. / số thứ tự | `ORDINAL` |
| 24. Phân số / tỷ lệ | `FRACTION`, `RATIO` |
| 25. Mã cảng / quốc gia | `PORT_CODE` |
| 26. Vũ khí / trang bị | `EQUIPMENT_ID`, `EQUIPMENT_NAME` |
| 27. Vũ khí hải quân / ngư lôi | `EQUIPMENT_ID`, `EQUIPMENT_NAME` |
| 28. Acronym cơ quan / hành chính | `ACRONYM` |
| 29. Acronym quân sự / CSB | `ACRONYM` |
| 30. Số âm | same target type + negative modifier; no new type |
| 31. Version | `VERSION` |
| 32. Địa chỉ | `ADDRESS` |
| 33. Biển số xe | `VEHICLE_PLATE` |

This taxonomy is frozen for the first V2 implementation.

Do not add a new neural class for each new surface format.

Prefer extending:

```text
parser grammar
validator
catalog
deterministic subtype
```

unless a new case has genuinely different contextual semantics or transformation behavior.

---

# 6. Factorized boundary + semantic type prediction

Do NOT use combined labels such as:

```text
B-COORD
I-COORD
E-COORD
B-EQUIPMENT_ID
...
```

inside the CRF.

V2 predicts two things independently:

```text
Boundary:
B / I / E / S / O

Semantic type:
46 types + O
```

Example:

```text
word       boundary     type
--------------------------------
su         B            EQUIPMENT_ID
ba         I            EQUIPMENT_ID
mươi       I            EQUIPMENT_ID
lăm        E            EQUIPMENT_ID

ở          O            O

hướng      O            O
không      B            HEADING
chín       I            HEADING
không      E            HEADING
```

## 6.1 Boundary CRF

Boundary labels:

```text
B
I
E
S
O
```

Valid structures:

```text
O
S
B -> I* -> E
```

Mask invalid transitions where practical.

## 6.2 Semantic span type

After boundary decoding returns a span:

```text
[start_word, end_word]
```

calculate:

```text
span_type_logits =
mean(type_logits[start_word:end_word+1], dim=word)
```

Then:

```text
span_type = argmax(span_type_logits)
```

Mean-logit aggregation is the V2 baseline.

---

# 7. Model heads

Input:

```text
word_hidden: [batch, words, 768]
```

Shared:

```text
Dropout(p=0.1)
```

## 7.1 Boundary head

```text
Linear(768, 5)
-> CRF
```

Expose:

```text
boundary_logits
decoded_boundaries
boundary_sequence_log_likelihood
boundary_marginals
```

CRF marginals are required for confidence estimation.

## 7.2 Semantic type head

```text
Linear(768, 47)
```

Outputs:

```text
46 semantic types + O
```

Training:

```text
CrossEntropy
```

Inference:

```text
decode boundary
-> build span
-> mean type logits over span
-> predict one type
```

## 7.3 Punctuation head

Labels:

```text
O
COMMA
PERIOD
QUESTION
```

Architecture:

```text
Linear(768, 4)
```

Punctuation MUST NOT:

- split ITN spans;
- merge ITN spans;
- determine semantic type;
- rewrite normalized span internals.

---

# 8. Loss

Final baseline loss:

```text
L_total =
    1.00 * L_BOUNDARY_CRF
  + 1.00 * L_TYPE
  + 0.30 * L_PUNCT
```

## 8.1 Boundary loss

```text
L_BOUNDARY_CRF =
negative CRF log likelihood
```

## 8.2 Type loss

```text
L_TYPE =
CrossEntropy(type_logits, type_targets)
```

Padding is ignored.

Words outside ITN spans use type target `O`.

Do not start with extreme 100x–150x class weights.

## 8.3 Punctuation loss

Use focal loss:

```text
gamma = 2.0
```

Initial weights:

```text
O        = 1.0
COMMA    = 2.0
PERIOD   = 2.0
QUESTION = 5.0
```

Do not combine extreme oversampling + extreme class weights + focal loss in the first baseline.

---

# 9. Confidence definition

Confidence must be explicit.

Do NOT use raw CRF sequence log-likelihood as span confidence.

## 9.1 Boundary confidence

Use CRF forward-backward marginals:

```text
P(boundary_tag_i | x)
```

For a predicted span:

```text
boundary_confidence =
min(
    P(predicted_boundary_i | x)
    for each word i in span
)
```

## 9.2 Type confidence

```text
span_type_logits =
mean(type_logits over span)

type_confidence =
softmax(span_type_logits)[predicted_type]
```

## 9.3 Combined confidence

Baseline:

```text
model_confidence =
min(boundary_confidence, type_confidence)
```

## 9.4 Per-class thresholds

Thresholds are tuned on `real_dev`.

Do not hard-code one global threshold.

Config:

```text
threshold[COORD]
threshold[HEADING]
threshold[MMSI_ID]
threshold[EQUIPMENT_ID]
...
```

Normalization is emitted only when:

```text
model_confidence >= threshold[predicted_type]
AND
parser succeeded
AND
validator passed
```

---

# 10. Typed normalizers

The model detects:

```text
boundary + semantic type
```

The model DOES NOT create normalized strings.

Mapping:

```text
CARDINAL        -> CardinalParser
ORDINAL         -> OrdinalParser
DIGIT_SEQ       -> DigitSequenceParser
DECIMAL         -> DecimalParser
FRACTION        -> FractionParser
PERCENT         -> PercentParser
RANGE           -> RangeParser
RATIO           -> RatioParser
MONEY           -> MoneyParser
MEASURE         -> MeasureParser
VERSION         -> VersionParser

DATE            -> DateParser
TIME            -> TimeParser
TIMEZONE        -> TimezoneParser
DURATION        -> DurationParser
ETA             -> ETAParser
ETD             -> ETDParser
QUARTER         -> QuarterParser

COORD           -> CoordinateParser
HEADING         -> HeadingParser
BEARING         -> BearingParser
SPEED           -> SpeedParser
DISTANCE        -> DistanceParser
DEPTH           -> DepthParser
DRAFT           -> DraftParser
FREQUENCY       -> FrequencyParser
CHANNEL         -> ChannelParser

MMSI_ID         -> MMSIParser
IMO_ID          -> IMOParser
CALLSIGN        -> CallsignParser
VESSEL_ID       -> VesselIDParser
PORT_CODE       -> PortCodeParser
DOCUMENT_ID     -> DocumentIDParser
LEGAL_DOC_ID    -> LegalDocumentParser
TELEPHONE       -> TelephoneParser
ELECTRONIC      -> ElectronicParser
VEHICLE_PLATE   -> VehiclePlateParser
ADDRESS         -> AddressParser

EQUIPMENT_ID    -> EquipmentIDParser
EQUIPMENT_NAME  -> EntityResolver
FOREIGN_NAME    -> EntityResolver
PERSON_NAME     -> PersonNameFormatter
LOCATION_NAME   -> LocationFormatter
ACRONYM         -> AcronymResolver
MARITIME_TERM   -> MaritimeTermResolver
RANK            -> RankFormatter
```

---

# 11. Generic measurement parser

`MEASURE` must support at least the current test catalog:

```text
km
m
mm
m²
m³
V
bar
°C
hPa
t
Hz
t/h
m³/h
NM
kn
dB
```

It must support:

```text
integer quantity
decimal quantity
negative quantity
compound unit
superscript unit
per-unit expression
```

Examples:

```text
bốn phẩy năm bar
-> 4,5 bar

âm mười độ xê
-> -10°C

tám mươi lăm tấn một giờ
-> 85 t/h

một trăm hai mươi mét khối một giờ
-> 120 m³/h
```

Specialized maritime classes override generic `MEASURE` when context supports them.

---

# 12. Electronic parser

`ELECTRONIC` deterministic subtypes:

```text
EMAIL
URL
IPV4
IPV6
MAC
```

Required examples:

```text
vts@portcontrol.vn
operations@coastguard.vn
master@oceanstar.com

www.vinamarine.gov.vn
www.marinetraffic.com
https://www.imo.org

192.168.10.25
2001:db8:100::25
00:1A:2B:3C:4D:5E
```

Preserve separators exactly according to subtype grammar.

Do not use lexical entity lookup for arbitrary electronic addresses.

---

# 13. Equipment ID parser

`EQUIPMENT_ID` must support general alphanumeric military and naval designations.

Do NOT implement only:

```text
PREFIX + NUMBER
```

Required grammar families:

```text
PREFIX + NUMBER
PREFIX + NUMBER + SUFFIX
PREFIX + NUMBER + LETTER + NUMBER
NUMBER + "-" + NUMBER + SUFFIX
MULTI_LETTER_PREFIX + NUMBER
PREFIX + optional separator + mixed alphanumeric suffix
```

Current required examples:

```text
AK-47
F-16
B-52
M16
M4A1
RPG-7
Mi-17
T-90
Su-30MK2
Ka-28

53-65KE
SET-65
MU90
MK 46
Kh-35
P-800
AK-630
```

Canonical prefix pronunciation mappings live in external catalog/config data.

Examples:

```text
su  -> Su
xu  -> Su
míc -> MiG
mi  -> Mi
ép  -> F
bê  -> B
tê  -> T
```

Preserve canonical mixed casing.

Correct:

```text
Su-35
MiG-29
Mi-17
Kh-35
Su-30MK2
```

Do not uppercase the entire prefix.

For ambiguous patterns, use grammar + catalog constraints and fall back when unresolved.

---

# 14. Entity/catalog resolver

Use catalog resolution for closed or semi-closed lexical entities:

```text
EQUIPMENT_NAME
FOREIGN_NAME
ACRONYM
MARITIME_TERM
```

Conceptual flow:

```text
raw spoken span
      ↓
pronunciation/text canonicalization
      ↓
candidate retrieval
      ↓
top-k
      ↓
ranking/similarity
      ↓
confidence threshold
      ↓
canonical value or raw fallback
```

Catalog entries support multiple spoken variants.

Example:

```text
canonical:
Patriot

spoken variants:
pa tri ốt
pat ri ốt
patriot
...
```

Catalog data must be external, not hard-coded into model source.

---

# 15. Validators

Every typed normalizer returns a structured result:

```python
{
    "raw_text": ...,
    "type": ...,
    "normalized": ...,
    "valid": ...,
    "reason": ...
}
```

## 15.1 Heading

```text
0 <= heading <= 359
```

Canonical 3-digit format:

```text
0   -> 000°
9   -> 009°
90  -> 090°
359 -> 359°
```

## 15.2 Coordinate

Latitude:

```text
0 <= degree <= 90
```

Longitude:

```text
0 <= degree <= 180
```

Minutes:

```text
0 <= minute < 60
```

Seconds:

```text
0 <= second < 60
```

## 15.3 MMSI

```text
exactly 9 digits
```

## 15.4 Other identifiers

Implement structural constraints when the domain standard defines them.

---

# 16. Safe fallback

Emit normalized text only if:

```text
model confidence passes threshold
AND
parser succeeds
AND
validator passes
```

Otherwise preserve the raw spoken form.

Example:

```text
raw:
ba bảy không độ

predicted type:
HEADING

parsed:
370°

validation:
invalid

final:
ba bảy không độ
```

Never emit an invalid high-risk value just because the classifier predicted a type.

---

# 17. Casing policy

Casing is applied after typed normalization.

Rules:

1. canonical entity spelling is protected;
2. generic cleanup cannot modify normalized span internals;
3. `RANK` keeps canonical casing;
4. sentence-start capitalization runs after punctuation restoration;
5. broad lowercase regex lists are not allowed to overwrite normalized entities.

Example:

```text
đại tá
-> RANK
-> Đại tá
```

No later cleanup may change it back.

---

# 18. Global regex policy

Global regex is allowed only for low-risk formatting cleanup:

```text
remove extra spaces before punctuation
collapse duplicate spaces
normalize safe punctuation spacing
```

Global regex MUST NOT:

```text
change semantic type
rewrite digits
rewrite equipment names
rewrite coordinates
change entity casing
split/merge ITN spans
```

Semantic normalization belongs in typed processors.

---

# 19. Long sequences

Do not hard-truncate long ASR text and discard the tail.

Use overlapping windows.

Initial policy:

```text
working useful window: ~224 PhoBERT subwords
overlap: 32-48 subwords
```

Exact values must respect PhoBERT special tokens and max length.

Stitch predictions using original word indices.

Prefer predictions from central/non-edge regions when overlap gives duplicate predictions.

---

# 20. Training data

Train with:

```text
A. clean spoken text
B. text-augmented spoken variants
C. synthetic audio -> current ASR -> noisy transcript
D. real annotated ASR data when available
```

Initial mixture:

```text
30% clean
20% text augmentation
50% ASR-generated noisy text
```

This is only a starting ratio.

Tune on `real_dev`.

ASR-error augmentation should increasingly reflect real confusion statistics from the production ASR.

---

# 21. Leakage prevention

All variants derived from one canonical source must remain in the same split.

Example:

```text
canonical:
Su-35

variants:
su ba mươi lăm
su ba lăm
xu ba mươi lăm
xu ba lăm
```

These cannot be divided between train and test.

Wrong:

```text
train: su ba mươi lăm
test:  su ba lăm
```

---

# 22. Dataset splits

Required:

```text
train
random_dev
entity_holdout_dev
template_holdout_dev
real_dev
real_test
```

Optional:

```text
asr_engine_holdout
```

## random_dev

Normal in-distribution dev data.

## entity_holdout_dev

Canonical entity/model completely absent from train.

Example:

```text
all Su-57 variants absent from train
```

Measures OOV/compositional generalization.

## template_holdout_dev

Sentence/template structures are unseen in train.

Example:

```text
train:
tàu đang ở hướng {HEADING}
chuyển hướng {HEADING}

template_holdout:
duy trì phương vị {HEADING} cho đến điểm chuyển hướng
```

This prevents synthetic-template leakage.

## real_dev

Used for:

```text
checkpoint selection
confidence threshold tuning
catalog threshold tuning
loss tuning
```

## real_test

Frozen production benchmark.

Never use for:

```text
training
augmentation
threshold tuning
checkpoint selection
```

---

# 23. Hard negatives

Training must contain contextual hard negatives.

Example:

```text
tốc độ mười hai hải lý một giờ
-> SPEED

chuyển sang kênh mười hai
-> CHANNEL

có mười hai tàu
-> CARDINAL
```

The model must learn meaning from context, not only recognize number words.

---

# 24. Training configuration

Initial baseline:

```text
Backbone:
PhoBERT-base-v2

Optimizer:
AdamW

Encoder LR:
2e-5

Boundary/type/punctuation heads LR:
1e-4

Weight decay:
0.01

Warmup:
5-10%

Scheduler:
linear decay

Dropout:
0.1

Gradient clipping:
1.0

Precision:
bf16 when supported

Effective batch size:
64

Epochs:
8-12
```

Use differential learning rates.

Early stopping uses real development metrics, not training loss.

---

# 25. Sampling

Start with:

```text
normal/random sampling
+ mild category balancing
```

Do not start with extreme weighted sampling.

Only increase balancing when per-class dev evidence requires it.

---

# 26. Metrics

Do not select models using overall token F1 only.

## 26.1 Boundary/type

```text
boundary strict span F1
semantic type strict span F1
BIOES invalid transition rate
boundary/type disagreement rate
```

## 26.2 Normalization

```text
exact normalized span accuracy
```

Report per semantic type.

Critical types include at least:

```text
COORD
HEADING
BEARING
MMSI_ID
IMO_ID
FREQUENCY
EQUIPMENT_ID
DATE
TIME
```

## 26.3 Safety

Mandatory metric:

```text
False Normalization Rate
```

Definition:

> percentage of content that should remain unchanged but is incorrectly transformed.

## 26.4 End-to-end

Evaluate final written output on `real_test`.

Also report:

```text
Critical Utterance Exact Match
```

If one critical coordinate/identifier digit is wrong, the complete critical utterance is counted as failed.

---

# 27. Checkpoint selection

Initial score:

```text
score =
    0.40 * ITN_exact_span_accuracy
  + 0.30 * critical_category_exact_accuracy
  + 0.10 * boundary_strict_F1
  + 0.10 * semantic_type_strict_F1
  + 0.10 * punctuation_F1
```

Use `real_dev` when it is sufficiently large.

A checkpoint can still be rejected if False Normalization Rate exceeds the acceptable threshold.

---

# 28. Required inference debug output

Example:

```json
{
  "raw_span": "xu ba lăm",
  "raw_token_start": 2,
  "raw_token_end": 5,
  "model_word_start": 2,
  "model_word_end": 4,
  "predicted_boundaries": ["B", "I", "E"],
  "predicted_type": "EQUIPMENT_ID",
  "boundary_confidence": 0.97,
  "type_confidence": 0.96,
  "model_confidence": 0.96,
  "threshold": 0.95,
  "normalized": "Su-35",
  "valid": true,
  "normalizer": "EquipmentIDParser"
}
```

This structured trace is mandatory for error attribution.

---

# 29. Oracle test

Before neural training, implement:

```text
gold raw spoken text
+
gold boundary
+
gold semantic type
      ↓
typed normalizer
      ↓
validator
      ↓
final written text
```

For deterministic categories, oracle normalization should be approximately 100%.

Known V1 failures that MUST disappear:

```text
DATE losing day when keyword is outside span
COORD label not controlling normalization
RANK casing overwritten by cleanup
equipment mixed casing overwritten
punctuation controlling ITN span boundary
tail silently lost by hard truncation
```

Do not consider neural training meaningful until oracle tests pass.

---

# 30. Required unit tests

Every typed processor must test:

```text
normal case
pronunciation variants
leading zero
negative value
decimal value
ASR substitution variant
invalid value
boundary value
unknown catalog entry
low-confidence catalog match
adjacent ITN spans
span near punctuation
long-sequence overlap
```

Minimum domain examples:

```text
su ba mươi lăm
su ba lăm
xu ba lăm

ép mười sáu
bê năm hai
míc hai chín
su ba mươi em ka hai

không chín không
một hai không

mười độ hai mươi lăm phút bắc
một trăm lẻ bảy độ tám phút đông

hai ba bốn năm sáu bảy tám chín không

đại tá
trung úy

bốn phẩy năm bar
âm mười độ xê

phiên bản một chấm hai chấm ba

vê en hát pê hát
```

All 33 current external test groups must also be converted into regression tests.

---

# 31. Suggested project layout

```text
itn_v2/
├── config.py
├── labels.py
├── segmentation_alignment.py
├── tokenizer_alignment.py
├── dataset.py
├── model.py
├── crf_constraints.py
├── confidence.py
├── losses.py
├── train.py
├── evaluate.py
├── inference.py
├── windowing.py
│
├── normalizers/
│   ├── base.py
│   ├── number.py
│   ├── ordinal.py
│   ├── fraction.py
│   ├── ratio.py
│   ├── money.py
│   ├── measure.py
│   ├── version.py
│   ├── date.py
│   ├── time.py
│   ├── timezone.py
│   ├── coordinate.py
│   ├── heading.py
│   ├── bearing.py
│   ├── maritime_measure.py
│   ├── frequency.py
│   ├── identifier.py
│   ├── telephone.py
│   ├── electronic.py
│   ├── address.py
│   ├── vehicle_plate.py
│   ├── equipment.py
│   └── entity.py
│
├── validators/
│   ├── coordinate.py
│   ├── heading.py
│   ├── mmsi.py
│   ├── identifiers.py
│   └── measurements.py
│
├── catalogs/
│   ├── equipment.json
│   ├── foreign_names.json
│   ├── acronyms.json
│   ├── maritime_terms.json
│   ├── port_codes.json
│   └── legal_docs.json
│
└── tests/
    ├── test_segmentation_alignment.py
    ├── test_tokenizer_alignment.py
    ├── test_boundary_crf.py
    ├── test_number.py
    ├── test_measure.py
    ├── test_date.py
    ├── test_coordinate.py
    ├── test_heading.py
    ├── test_identifier.py
    ├── test_electronic.py
    ├── test_equipment.py
    ├── test_entity.py
    ├── test_windowing.py
    ├── test_oracle_pipeline.py
    └── test_catalog_33_groups.py
```

---

# 32. Implementation order

Claude should implement in this order.

## Phase 1 — data contract / alignment

1. Define 46 semantic types + `O`.
2. Define boundary labels `B/I/E/S/O`.
3. Implement Vietnamese word segmentation.
4. Implement reversible model-word ↔ raw-token alignment.
5. Implement PhoBERT tokenizer/subword alignment.
6. Implement mean subword pooling.
7. Add alignment tests.

## Phase 2 — deterministic normalization

8. Implement typed normalizer interface.
9. Implement numeric core.
10. Implement `ORDINAL/FRACTION/RATIO`.
11. Implement `MONEY/MEASURE/VERSION`.
12. Implement date/time/timezone/ETA/ETD/quarter.
13. Implement coordinate.
14. Implement heading/bearing.
15. Implement specialized maritime measurement parsers.
16. Implement structured identifiers.
17. Implement telephone/electronic.
18. Implement address/vehicle plate.
19. Implement general alphanumeric equipment grammar.
20. Implement catalogs/entity resolvers.
21. Implement validators.
22. Build oracle pipeline.
23. Make oracle + 33-group regression tests pass.

## Phase 3 — neural model

24. Implement PhoBERT encoder.
25. Implement 5-state boundary CRF.
26. Implement 47-way type head.
27. Implement punctuation head.
28. Implement CRF marginals.
29. Implement confidence calculation.
30. Implement losses.
31. Implement training loop.

## Phase 4 — long input

32. Implement overlapping windows.
33. Implement raw/model word-index-preserving stitching.
34. Test overlap edge cases.

## Phase 5 — evaluation

35. Boundary strict F1.
36. Semantic type strict F1.
37. Per-type exact normalization.
38. False Normalization Rate.
39. Critical-category exact accuracy.
40. Template-holdout evaluation.
41. Checkpoint score.

## Phase 6 — production debugging

42. Structured span diagnostics.
43. Confidence/threshold logs.
44. Category failure logs.
45. Catalog miss logs.
46. Validator rejection logs.
47. ASR error-pattern breakdown.

---

# 33. Explicitly out of scope for V2 baseline

Do not add before V2 is stable:

```text
BiLSTM
T5
seq2seq normalization
LLM rewriting
alternative backbone
DAPT
audio-text punctuation fusion
large neural retrieval model
multi-stage neural reranker
```

First establish the controlled PhoBERT V2 baseline.

---

# 34. Future backbone benchmark

After V2 is stable, keep constant:

```text
dataset
46-type taxonomy
boundary CRF
typed normalizers
validators
loss
splits
metrics
```

Then replace only encoder/tokenizer.

Planned candidates:

```text
PhoBERT-base-v2
ViDeBERTa-base
XLM-R-base
mDeBERTa-v3-base
```

This allows clean backbone attribution.

---

# 35. Final coding checklist

V2 is complete only when:

```text
[ ] PhoBERT input is Vietnamese word-segmented.
[ ] Mapping back to raw ASR tokens is reversible.
[ ] PhoBERT uses native subword positions.
[ ] Word representation uses mean subword pooling.
[ ] Boundary prediction uses 5-state BIOES CRF.
[ ] Semantic type prediction is a separate 46-type + O head.
[ ] Punctuation is independent of ITN boundaries.
[ ] All 33 current test groups map to the taxonomy.
[ ] MONEY exists.
[ ] MEASURE exists.
[ ] TIMEZONE exists.
[ ] TELEPHONE exists.
[ ] ELECTRONIC exists.
[ ] VERSION exists.
[ ] ADDRESS exists.
[ ] VEHICLE_PLATE exists.
[ ] COORD has a real typed normalizer.
[ ] HEADING preserves leading zeroes.
[ ] MMSI preserves exact digit sequence.
[ ] EQUIPMENT_ID supports mixed alphanumeric patterns.
[ ] Canonical mixed casing such as Su/MiG/Kh is preserved.
[ ] Generic negative-number handling is a parser modifier, not a new class.
[ ] Electronic EMAIL/URL/IP/MAC subtypes work.
[ ] Entity lookup has confidence fallback.
[ ] CRF boundary marginals are available.
[ ] Span confidence is explicitly defined.
[ ] Per-class thresholds are configurable.
[ ] High-risk types have validators.
[ ] Failed normalization safely returns raw spoken text.
[ ] Global regex cannot overwrite normalized spans.
[ ] Long sequences use overlapping windows.
[ ] Derived variants do not leak across data splits.
[ ] entity_holdout_dev exists.
[ ] template_holdout_dev exists.
[ ] real_test remains frozen.
[ ] Oracle normalization tests pass.
[ ] All 33 test groups are regression-tested.
[ ] Per-category exact-match metrics exist.
[ ] False Normalization Rate is reported.
[ ] Inference exposes structured debug output.
```

---

# 36. Final design principle

> **PhoBERT predicts where an ITN span is and what semantic type it represents; deterministic typed processors decide how the original spoken span must be written; confidence gates and domain validators decide whether that transformation is safe to emit.**

This principle is frozen for the V2 baseline.
