"""Derlem genelinde kod -> kategori -> tema hiyerarşisi.

Varsayılan ücretsiz aşama: önerilen etiketleri deterministik olarak birleştirir.
LLM toplu hiyerarşi önerisi ayrı opsiyonel çağrıdır; doğrulama zorunludur.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict

FIELDS = ("code", "category", "theme")
MAX_LABEL = 200


def _label(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_LABEL:
        raise ValueError("Kod/kategori/tema etiketi boş veya çok uzun.")
    return " ".join(value.split())


def build_hierarchy(records, participant_ids=None):
    """Onaylanmış kayıtlardan benzersiz kod ilişkileri ve çoklu kodlama sayımları."""
    rows = list(records)
    universe = set(participant_ids) if participant_ids is not None else {r["participant_id"] for r in rows}
    nodes = {}
    themes = defaultdict(set)
    categories = defaultdict(set)
    for r in rows:
        pid = str(r["participant_id"]).strip()
        if not pid or pid not in universe:
            raise ValueError("Katılımcı listesinde bulunmayan kayıt.")
        code, category, theme = (_label(r[f]) for f in FIELDS)
        # Farklı üst ilişkilere atanan aynı kod: otomatik zorla birleştirme yapma.
        key = (code, category, theme)
        item = nodes.setdefault(key, {"code": code, "category": category,
                                      "theme": theme, "participants": set(), "evidence_count": 0})
        item["participants"].add(pid)
        item["evidence_count"] += 1
        categories[(category, theme)].add(code)
        themes[theme].add(category)
    n = len(universe)
    mapping = [
        {"code": x["code"], "category": x["category"], "theme": x["theme"],
         "n_participants": len(x["participants"]),
         "percentage": round(100 * len(x["participants"]) / n, 2) if n else 0,
         "n_evidence": x["evidence_count"]}
        for _, x in sorted(nodes.items())
    ]
    conflicts = sorted({k[0] for k in nodes if len({(q[1], q[2]) for q in nodes if q[0] == k[0]}) > 1})
    return {
        "mapping": mapping,
        "categories": [{"category": c, "theme": t, "codes": sorted(v)}
                       for (c, t), v in sorted(categories.items())],
        "themes": [{"theme": t, "categories": sorted(v)} for t, v in sorted(themes.items())],
        "conflicting_codes": conflicts,
        "denominator": n,
        "needs_researcher_approval": True,
    }


def prepare_hierarchy_prompt(records, *, max_codes=120):
    """Birleştirilmiş kod adlarıyla tek kompakt çağrı; ham görüşme metinlerini GÖNDERMEZ."""
    counts = Counter(_label(r["code"]) for r in records)
    if len(counts) > max_codes:
        raise ValueError("Tek çağrıda çok fazla kod: önce insan incelemesiyle sadeleştirin.")
    if not counts:
        raise ValueError("Önerilecek kod yok.")
    content = [{"code": code, "count": n} for code, n in sorted(counts.items())]
    return ("Türkçe nitel analizde aşağıdaki mevcut kodları tutarlı kategori ve temalarda "
            "grupla. Yeni kod üretme, bütün kodları tam bir kez döndür. Çok genel tema kurma. "
            "Yalnızca JSON: {\"mapping\":[{\"code\":\"...\",\"category\":\"...\",\"theme\":\"...\"}]}. "
            "Kodlar: " + json.dumps(content, ensure_ascii=False, separators=(",", ":")))


def validate_hierarchy_response(raw, existing_codes):
    """LLM'in yeni kod eklemesini, kod atlamasını ve yinelenen kod üretmesini reddet."""
    if not isinstance(raw, str) or len(raw) > 50000:
        raise ValueError("Geçersiz yanıt.")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("mapping"), list):
        raise ValueError("JSON mapping listesi gerekli.")
    expected = {_label(x) for x in existing_codes}
    mapping = {}
    for item in obj["mapping"]:
        if not isinstance(item, dict) or any(f not in item for f in FIELDS):
            raise ValueError("Eksik hiyerarşi alanı.")
        code, category, theme = (_label(item[f]) for f in FIELDS)
        if code not in expected or code in mapping:
            raise ValueError("Bilinmeyen veya yinelenen kod.")
        mapping[code] = (category, theme)
    if set(mapping) != expected:
        raise ValueError("Bazı kodlar eksik.")
    return mapping


def apply_hierarchy(records, mapping):
    """Kanıt metni ve katılımcı kimliği korunarak yalnızca üst etiketleri değiştir."""
    out = []
    for original in records:
        row = dict(original)
        key = _label(row["code"])
        if key not in mapping:
            raise ValueError("Kod için eşleme bulunamadı.")
        row["category"], row["theme"] = mapping[key]
        out.append(row)
    return out


def hierarchy_changes(records, mapping):
    """Araştırmacı onayı için mevcut → önerilen üst etiket farkları; hiçbir kaydı değiştirmez."""
    current = defaultdict(set)
    for r in records:
        current[_label(r["code"])].add((_label(r["category"]), _label(r["theme"])))
    if set(current) != set(mapping):
        raise ValueError("Öneri, onaylı kodlarla birebir eşleşmiyor.")
    rows = []
    for code in sorted(current):
        category, theme = mapping[code]
        for old_category, old_theme in sorted(current[code]):
            if (old_category, old_theme) != (category, theme):
                rows.append({"code": code, "current_category": old_category, "current_theme": old_theme,
                             "proposed_category": category, "proposed_theme": theme})
    return rows
