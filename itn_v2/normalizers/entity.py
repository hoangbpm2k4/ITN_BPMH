"""Thực thể tra danh mục và định dạng tên (spec §14, §17)."""

from ..catalog import get_catalog
from .base import Normalizer, ParseError


class EntityResolver(Normalizer):
    """Tra một hoặc nhiều danh mục theo thứ tự; trượt thì báo lỗi để pipeline
    giữ nguyên dạng nói thay vì đoán bừa."""

    name = "EntityResolver"
    catalogs = ()

    def __init__(self, catalogs=None, threshold=None):
        if catalogs:
            self.catalogs = tuple(catalogs)
        self.threshold = threshold

    def parse(self, raw_text, context=None):
        best_score = 0.0
        for cat_name in self.catalogs:
            catalog = get_catalog(cat_name)
            canonical, score = catalog.lookup(raw_text)
            if canonical and (self.threshold is None or score >= self.threshold):
                return canonical
            best_score = max(best_score, score)
        raise ParseError(f"không có mục nào trong danh mục đạt ngưỡng (tốt nhất {best_score:.2f})")


class ForeignNameResolver(EntityResolver):
    name = "ForeignNameResolver"
    catalogs = ("foreign_names",)


class EquipmentNameResolver(EntityResolver):
    name = "EquipmentNameResolver"
    catalogs = ("foreign_names", "equipment")


class AcronymResolver(EntityResolver):
    name = "AcronymResolver"
    catalogs = ("acronyms",)


class MaritimeTermResolver(EntityResolver):
    name = "MaritimeTermResolver"
    catalogs = ("maritime_terms",)


def _title_words(text, all_words=True):
    out = []
    for i, w in enumerate(text.split()):
        w = w.replace("_", " ")
        if all_words or i == 0:
            out.append(" ".join(p[:1].upper() + p[1:].lower() if p else p for p in w.split()))
        else:
            out.append(w.lower())
    return " ".join(out)


class PersonNameFormatter(Normalizer):
    name = "PersonNameFormatter"

    def parse(self, raw_text, context=None):
        return _title_words(raw_text, all_words=True)


class LocationFormatter(Normalizer):
    name = "LocationFormatter"

    def parse(self, raw_text, context=None):
        return _title_words(raw_text, all_words=True)


class RankFormatter(Normalizer):
    """Quân hàm: hoa chữ đầu cụm. Dạng này được BẢO VỆ, không luật dọn nào
    được phép hạ xuống về sau (spec §17) — đúng lỗi V1 'Đại tá' -> 'đại tá'."""

    name = "RankFormatter"

    def parse(self, raw_text, context=None):
        return _title_words(raw_text, all_words=False)
