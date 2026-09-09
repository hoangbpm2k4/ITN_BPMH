"""Bộ kiểm tra miền giá trị (spec §15).

Validator chạy SAU normalizer và là cổng cuối cùng trước khi phát ra dạng viết.
Không đạt -> giữ nguyên dạng nói (spec §16). Nguyên tắc: thà giữ dạng nói còn
hơn phát ra một giá trị sai ở lớp quan trọng.
"""

import re

from .coordinate import validate_coordinate
from .heading import validate_heading
from .identifiers import (validate_imo, validate_mmsi, validate_port_code,
                          validate_telephone, validate_vehicle_plate)
from .measurements import validate_measure_like

# type -> hàm (normalized: str) -> (ok: bool, reason: str)
VALIDATORS = {
    "COORD": validate_coordinate,
    "HEADING": validate_heading,
    "BEARING": validate_heading,
    "MMSI_ID": validate_mmsi,
    "IMO_ID": validate_imo,
    "TELEPHONE": validate_telephone,
    "PORT_CODE": validate_port_code,
    "VEHICLE_PLATE": validate_vehicle_plate,
    "MEASURE": validate_measure_like,
    "SPEED": validate_measure_like,
    "DISTANCE": validate_measure_like,
    "DEPTH": validate_measure_like,
    "DRAFT": validate_measure_like,
    "FREQUENCY": validate_measure_like,
}


def validate(type_name, normalized):
    fn = VALIDATORS.get(type_name)
    if fn is None:
        return True, ""
    try:
        return fn(normalized)
    except Exception as exc:  # validator hỏng không được làm sập pipeline
        return False, f"validator lỗi: {type(exc).__name__}: {exc}"


__all__ = ["validate", "VALIDATORS"]
