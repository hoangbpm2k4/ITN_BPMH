"""Bảng điều phối kiểu ngữ nghĩa -> bộ chuẩn hoá (spec §10)."""

from .normalizers.coordinate import BearingParser, CoordinateParser, HeadingParser
from .normalizers.date import (DateParser, DurationParser, ETAParser, ETDParser,
                               QuarterParser, TimeParser, TimezoneParser)
from .normalizers.electronic import ElectronicParser
from .normalizers.entity import (AcronymResolver, EquipmentNameResolver,
                                 ForeignNameResolver, LocationFormatter,
                                 MaritimeTermResolver, PersonNameFormatter,
                                 RankFormatter)
from .normalizers.equipment import EquipmentIDParser
from .normalizers.identifier import (AddressParser, CallsignParser,
                                     DocumentIDParser, IMOParser,
                                     LegalDocumentParser, MMSIParser,
                                     PortCodeParser, TelephoneParser,
                                     VehiclePlateParser, VesselIDParser)
from .normalizers.measure import (ChannelParser, DepthParser, DistanceParser,
                                  DraftParser, FrequencyParser, MeasureParser,
                                  MoneyParser, SpeedParser)
from .normalizers.number import (CardinalParser, DecimalParser,
                                 DigitSequenceParser, FractionParser,
                                 OrdinalParser, PercentParser, RangeParser,
                                 RatioParser, VersionParser)

_REGISTRY = None


def _build():
    return {
        "CARDINAL": CardinalParser(),
        "ORDINAL": OrdinalParser(),
        "DIGIT_SEQ": DigitSequenceParser(),
        "DECIMAL": DecimalParser(),
        "FRACTION": FractionParser(),
        "PERCENT": PercentParser(),
        "RANGE": RangeParser(),
        "RATIO": RatioParser(),
        "MONEY": MoneyParser(),
        "MEASURE": MeasureParser(),
        "VERSION": VersionParser(),

        "DATE": DateParser(),
        "TIME": TimeParser(),
        "TIMEZONE": TimezoneParser(),
        "DURATION": DurationParser(),
        "ETA": ETAParser(),
        "ETD": ETDParser(),
        "QUARTER": QuarterParser(),

        "COORD": CoordinateParser(),
        "HEADING": HeadingParser(),
        "BEARING": BearingParser(),
        "SPEED": SpeedParser(),
        "DISTANCE": DistanceParser(),
        "DEPTH": DepthParser(),
        "DRAFT": DraftParser(),
        "FREQUENCY": FrequencyParser(),
        "CHANNEL": ChannelParser(),

        "MMSI_ID": MMSIParser(),
        "IMO_ID": IMOParser(),
        "CALLSIGN": CallsignParser(),
        "VESSEL_ID": VesselIDParser(),
        "PORT_CODE": PortCodeParser(),
        "DOCUMENT_ID": DocumentIDParser(),
        "LEGAL_DOC_ID": LegalDocumentParser(),
        "TELEPHONE": TelephoneParser(),
        "ELECTRONIC": ElectronicParser(),
        "VEHICLE_PLATE": VehiclePlateParser(),
        "ADDRESS": AddressParser(),

        "EQUIPMENT_ID": EquipmentIDParser(),
        "EQUIPMENT_NAME": EquipmentNameResolver(),
        "FOREIGN_NAME": ForeignNameResolver(),
        "PERSON_NAME": PersonNameFormatter(),
        "LOCATION_NAME": LocationFormatter(),
        "ACRONYM": AcronymResolver(),
        "MARITIME_TERM": MaritimeTermResolver(),
        "RANK": RankFormatter(),
    }


def get_normalizer(type_name):
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build()
    return _REGISTRY.get(type_name)


def all_types():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build()
    return sorted(_REGISTRY)
