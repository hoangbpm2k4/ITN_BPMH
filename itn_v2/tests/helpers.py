"""Tiện ích cho test: chú thích span bằng CHỮ thay vì chỉ số, tránh đếm tay sai."""

from itn_v2.inference import normalize_with_gold_spans


def annotate(text, *pairs):
    """annotate("a b c d", ("b c", "TYPE")) -> [(1, 2, "TYPE")] theo token gốc."""
    words = text.split()
    anns, cursor = [], 0
    for span_text, type_name in pairs:
        sw = span_text.split()
        for i in range(cursor, len(words) - len(sw) + 1):
            if words[i:i + len(sw)] == sw:
                anns.append((i, i + len(sw) - 1, type_name))
                cursor = i + len(sw)
                break
        else:
            raise AssertionError(f"không tìm thấy cụm {span_text!r} trong {text!r}")
    return anns


def run(text, *pairs, punct=None, final_period=True):
    """Chạy oracle. ``punct`` là dict {từ_cuối_cụm: nhãn} theo chỉ số token gốc."""
    words = text.split()
    labels = ["O"] * len(words)
    if final_period:
        labels[-1] = "PERIOD"
    for idx, tag in (punct or {}).items():
        labels[idx] = tag
    out, spans, alignment, conflicts = normalize_with_gold_spans(
        text, annotate(text, *pairs), labels)
    return out, spans, alignment, conflicts
