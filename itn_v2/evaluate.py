"""Chỉ số đánh giá (spec §26, §27).

Cấm chọn mô hình chỉ bằng F1 token tổng. Bộ chỉ số ở đây tách riêng ranh giới,
kiểu, giá trị sau chuẩn hoá, và — quan trọng nhất — **False Normalization Rate**:
tỷ lệ nội dung lẽ ra phải giữ nguyên nhưng bị biến đổi.
"""

from collections import defaultdict

from .labels import CRITICAL_TYPES, decode_spans


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def boundary_strict_f1(gold_boundaries, pred_boundaries):
    """Span khớp khi trùng cả điểm đầu lẫn điểm cuối (spec §26.1)."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_boundaries, pred_boundaries):
        g = set(decode_spans(gold))
        p = set(decode_spans(pred))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return _prf(tp, fp, fn)


def type_strict_f1(gold_items, pred_items):
    """gold/pred: danh sách [(start, end, type)] cho từng câu."""
    tp = fp = fn = 0
    per_type = defaultdict(lambda: [0, 0, 0])
    for gold, pred in zip(gold_items, pred_items):
        g, p = set(gold), set(pred)
        for item in g & p:
            tp += 1
            per_type[item[2]][0] += 1
        for item in p - g:
            fp += 1
            per_type[item[2]][1] += 1
        for item in g - p:
            fn += 1
            per_type[item[2]][2] += 1
    overall = _prf(tp, fp, fn)
    overall["per_type"] = {t: _prf(*v) for t, v in sorted(per_type.items())}
    return overall


def invalid_transition_rate(pred_boundaries):
    """Tỷ lệ bước chuyển sai cấu trúc BIOES (spec §26.1)."""
    bad = total = 0
    for tags in pred_boundaries:
        prev = "O"
        for tag in tags:
            total += 1
            if prev in ("O", "S", "E") and tag in ("I", "E"):
                bad += 1
            elif prev in ("B", "I") and tag in ("O", "B", "S"):
                bad += 1
            prev = tag
        if prev in ("B", "I"):
            bad += 1
            total += 1
    return bad / total if total else 0.0


def exact_normalization_accuracy(records):
    """records: [{'type', 'gold', 'pred'}] — so khớp tuyệt đối giá trị sau chuẩn hoá."""
    per_type = defaultdict(lambda: [0, 0])
    for r in records:
        per_type[r["type"]][1] += 1
        if r["pred"] == r["gold"]:
            per_type[r["type"]][0] += 1
    out = {t: {"correct": c, "total": n, "accuracy": c / n if n else 0.0}
           for t, (c, n) in sorted(per_type.items())}
    total = sum(n for _, n in per_type.values())
    correct = sum(c for c, _ in per_type.values())
    return {"overall": correct / total if total else 0.0, "per_type": out}


def critical_category_accuracy(records):
    critical = [r for r in records if r["type"] in CRITICAL_TYPES]
    if not critical:
        return 0.0
    return sum(1 for r in critical if r["pred"] == r["gold"]) / len(critical)


def false_normalization_rate(records):
    """Tỷ lệ nội dung lẽ ra giữ nguyên nhưng bị biến đổi (spec §26.3).

    Một bản ghi tính là sai-chuẩn-hoá khi nhãn vàng là O (hoặc yêu cầu giữ
    nguyên) mà hệ thống vẫn phát ra dạng viết khác dạng nói.
    """
    should_keep = [r for r in records if r.get("gold_type", r.get("type")) == "O"
                   or r.get("must_keep")]
    if not should_keep:
        return 0.0
    changed = sum(1 for r in should_keep if r["pred"] != r["raw"])
    return changed / len(should_keep)


def critical_utterance_exact_match(utterances):
    """Sai một chữ số ở lớp quan trọng là hỏng cả phát ngôn (spec §26.4)."""
    if not utterances:
        return 0.0
    ok = 0
    for u in utterances:
        crit = [r for r in u["records"] if r["type"] in CRITICAL_TYPES]
        if crit and all(r["pred"] == r["gold"] for r in crit):
            ok += 1
        elif not crit:
            ok += 1
    return ok / len(utterances)


def checkpoint_score(itn_exact, critical_exact, boundary_f1, type_f1, punct_f1):
    """Điểm chọn checkpoint (spec §27)."""
    return (0.40 * itn_exact + 0.30 * critical_exact + 0.10 * boundary_f1
            + 0.10 * type_f1 + 0.10 * punct_f1)


def punctuation_f1(gold, pred, o_index=0):
    """Macro-F1 trên các lớp KHÁC O.

    Đừng báo cáo accuracy thô: 94% số từ mang nhãn "không có dấu câu", nên
    accuracy luôn đẹp kể cả khi mô hình bỏ sót gần hết dấu chấm. Đây đúng là
    cái bẫy mà V1 mắc phải.
    """
    labels = sorted({l for seq in gold for l in seq} | {l for seq in pred for l in seq})
    per_class, f1s = {}, []
    for label in labels:
        if label == o_index:
            continue
        tp = fp = fn = 0
        for g_seq, p_seq in zip(gold, pred):
            for g, p in zip(g_seq, p_seq):
                if g == label and p == label:
                    tp += 1
                elif p == label:
                    fp += 1
                elif g == label:
                    fn += 1
        stats = _prf(tp, fp, fn)
        per_class[label] = stats
        f1s.append(stats["f1"])
    return {"macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "per_class": per_class}
