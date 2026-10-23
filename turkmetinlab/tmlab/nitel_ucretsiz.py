"""Ücretsiz, deterministik kod benzerliği önerileri.

Sadece öneri üretir; uzman onayı olmadan kod isimlerini değiştirmez.
Özel verileri modele veya dış API'ye göndermez.
"""
from __future__ import annotations
import re
import unicodedata
from difflib import SequenceMatcher

def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value.casefold())
    return set(re.findall(r"(?u)\b\w+\b", normalized))

def suggest_merges(codes, *, threshold=0.75, max_codes=500):
    """Yüksek benzerlikli kod ADI çiftleri; anlamsal eşdeğerlik garantisi vermez."""
    if not 0.5 <= threshold <= 1:
        raise ValueError("Benzerlik eşiği 0,50–1 aralığında olmalıdır.")
    names = sorted({x.strip() for x in codes if isinstance(x, str) and x.strip()})
    if len(names) > max_codes:
        raise ValueError("Kod sayısı güvenli üst sınırı aşıyor.")
    suggestions = []
    for i, left in enumerate(names):
        tok_a = _tokens(left)
        for right in names[i+1:]:
            tok_b = _tokens(right)
            jac = len(tok_a & tok_b) / len(tok_a | tok_b) if tok_a | tok_b else 0
            lexical = SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
            score = max(jac, lexical)
            if score >= threshold:
                suggestions.append({"code_a": left, "code_b": right,
                                    "lexical_similarity": round(score, 3),
                                    "review_required": True})
    return sorted(suggestions, key=lambda x: (-x["lexical_similarity"], x["code_a"], x["code_b"]))

def merge_approved_codes(records, replacements):
    """Yalnızca açıkça onaylanmış ad dönüşümlerini uygular, kanıtları korur."""
    if not isinstance(replacements, dict):
        raise ValueError("Onaylanan kod eşlemeleri sözlük olmalıdır.")
    if any(not isinstance(k,str) or not isinstance(v,str) or not k.strip() or not v.strip()
           for k,v in replacements.items()):
        raise ValueError("Kod adları boş olamaz.")
    result=[]
    for rec in records:
        row=dict(rec)
        row["code"]=replacements.get(row["code"],row["code"])
        result.append(row)
    return result
