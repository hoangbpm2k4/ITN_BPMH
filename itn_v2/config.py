"""Cấu hình V2: ngưỡng theo lớp, tham số mô hình, tham số cửa sổ trượt."""

from dataclasses import dataclass, field

from .labels import CRITICAL_TYPES, SEMANTIC_TYPES

PHOBERT_PATH = "/home/hoangbpm/inverse_text_norm/runtime_phobert_asr_bilstm/models/phobert_hf"

# Ngưỡng tin cậy theo lớp (spec §9.4). KHÔNG dùng một ngưỡng chung.
# Giá trị khởi đầu; phải tinh chỉnh lại trên real_dev.
DEFAULT_THRESHOLD = 0.60
CRITICAL_THRESHOLD = 0.90        # sai một chữ số là hỏng cả bản tin
CATALOG_THRESHOLD = 0.75         # lớp tra danh mục, đã có ngưỡng khớp riêng


def build_thresholds():
    thresholds = {t: DEFAULT_THRESHOLD for t in SEMANTIC_TYPES}
    for t in CRITICAL_TYPES:
        thresholds[t] = CRITICAL_THRESHOLD
    for t in ("TELEPHONE", "MMSI_ID", "IMO_ID", "VESSEL_ID", "CALLSIGN",
              "PORT_CODE", "VEHICLE_PLATE", "LEGAL_DOC_ID", "DOCUMENT_ID"):
        thresholds[t] = CRITICAL_THRESHOLD
    for t in ("EQUIPMENT_NAME", "FOREIGN_NAME", "ACRONYM", "MARITIME_TERM"):
        thresholds[t] = CATALOG_THRESHOLD
    return thresholds


@dataclass
class Config:
    phobert_path: str = PHOBERT_PATH
    # PhoBERT được huấn luyện trên văn bản đã tách từ ghép nên cần tầng tách từ.
    # Các backbone khác (XLM-R, mBERT) huấn luyện trên văn bản THÔ — ghép âm tiết
    # bằng dấu "_" rồi đưa vào sẽ tạo ra token lạ ngoài vốn từ của chúng.
    segment_words: bool = True
    max_len: int = 256                 # giới hạn cứng của PhoBERT
    window_useful: int = 224           # spec §19
    window_overlap: int = 40
    dropout: float = 0.1

    encoder_lr: float = 2e-5           # spec §24 — learning rate phân tầng
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    epochs: int = 10
    effective_batch_size: int = 64
    grad_clip: float = 1.0

    w_boundary: float = 1.00           # spec §8
    w_type: float = 1.00
    # Spec §8 đặt 0.30. ĐO ĐƯỢC 2026-09-08 trên 1.127 câu: đặt 1.00 thì COMMA
    # recall nhảy 0.18 -> 0.76 và KHÔNG chỉ số nào tụt (boundary 0.888->0.917,
    # type 0.747->0.772, utterance exact 0.082->0.129). Giữ nguyên giá trị spec
    # làm mặc định; dùng cờ --punct-weight để bật 1.00 khi đã quyết định đổi.
    w_punct: float = 0.30
    punct_focal_gamma: float = 2.0
    punct_class_weights: tuple = (1.0, 2.0, 2.0, 5.0)   # O, COMMA, PERIOD, QUESTION

    thresholds: dict = field(default_factory=build_thresholds)

    def threshold_for(self, type_name):
        return self.thresholds.get(type_name, DEFAULT_THRESHOLD)
