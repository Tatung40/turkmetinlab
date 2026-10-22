"""TürkMetinLab 3.0: YZ destekli nitel kodlamanın doğrulanabilir çekirdeği.

Standart kütüphane; model/API bağlantısından bağımsız. Metinler diske yazılmaz.
Modelin ürettiği kanıt alıntıları özgün katılımcı metnine karşı doğrulanır.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Iterable

MAX_CHARS = 12000
MAX_CODES = 12


@dataclass(frozen=True)
class EvidenceCode:
    participant_id: str
    quote: str
    code: str
    category: str
    theme: str
    start: int
    end: int


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def document_fingerprint(text: str) -> str:
    """Cache anahtarı; metni veya katılımcı bilgisini içermez."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_prompt(text: str, codebook: Iterable[str] = ()) -> str:
    """Sınırlı, düşük token maliyetli Türkçe nitel analiz talimatı."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
        raise ValueError("Metin boş veya izin verilen sınırı aşıyor.")
    codes = [str(c).strip() for c in codebook if str(c).strip()]
    if len(codes) > 50:
        raise ValueError("Kod kitabı en çok 50 kod içerebilir.")
    instruction = (
        "Yalnızca JSON döndür; açıklama ekleme. Türkçe nitel veri için en çok 12 "
        "anlamlı kod öner. Her kodda quote, code, category, theme alanları olsun. "
        "quote, KAYNAK metinden HARFİYEN alınan kesintisiz bir alıntı olmalı. "
        "Metinde olmayan kanıt veya görüş uydurma. Belirsizse boş liste döndür. "
        'Şema: {"codes":[{"quote":"...","code":"...","category":"...","theme":"..."}]}.'
    )
    if codes:
        instruction += " Yalnızca verilen kod kitabından kod seç: " + json.dumps(codes, ensure_ascii=False)
    return instruction + "\nKAYNAK:\n" + text


def parse_response(raw: str) -> list[dict]:
    """JSON dışında kalan açıklama/markdown yanıtlarını reddet."""
    if not isinstance(raw, str) or len(raw) > 40000:
        raise ValueError("Model yanıtı biçimi veya uzunluğu geçersiz.")
    result = json.loads(raw)
    if not isinstance(result, dict) or not isinstance(result.get("codes"), list):
        raise ValueError("Beklenen JSON şeması yok.")
    if len(result["codes"]) > MAX_CODES:
        raise ValueError("Önerilen kod sayısı sınırı aşıyor.")
    return result["codes"]


def validate_codes(participant_id: str, source: str, candidates: list[dict],
                   allowed_codes: Iterable[str] | None = None) -> tuple[list[EvidenceCode], list[str]]:
    """Yanlış/uydurma kanıtları kesin biçimde ayır; insan onayı yine zorunlu."""
    if not isinstance(source, str) or not isinstance(participant_id, str):
        raise ValueError("Katılımcı ve metin zorunludur.")
    permitted = set(allowed_codes) if allowed_codes is not None else None
    valid: list[EvidenceCode] = []
    rejected: list[str] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            rejected.append(f"{index}: kod nesnesi değil")
            continue
        if not all(isinstance(item.get(k), str) and item[k].strip()
                   for k in ("quote", "code", "category", "theme")):
            rejected.append(f"{index}: eksik alan")
            continue
        if permitted is not None and item["code"].strip() not in permitted:
            rejected.append(f"{index}: kod kitabı dışında öneri")
            continue
        quote = item["quote"]
        start = source.find(quote)
        if start < 0:
            rejected.append(f"{index}: kanıt özgün metinde birebir bulunamadı")
            continue
        if any(len(item[k]) > 200 for k in ("code", "category", "theme")):
            rejected.append(f"{index}: etiket çok uzun")
            continue
        valid.append(EvidenceCode(
            participant_id=participant_id, quote=quote,
            code=item["code"].strip(), category=item["category"].strip(),
            theme=item["theme"].strip(), start=start, end=start + len(quote)))
    return valid, rejected


def analyze_model_response(participant_id: str, source: str, raw: str,
                           allowed_codes: Iterable[str] | None = None) -> dict:
    codes, rejected = validate_codes(participant_id, source, parse_response(raw), allowed_codes)
    return {"participant_id": participant_id, "codes": [asdict(c) for c in codes],
            "rejected": rejected, "requires_human_review": True}


def frequencies(records: Iterable[EvidenceCode], participant_ids: Iterable[str] | None = None) -> list[dict]:
    """Yüzdeyi benzersiz katılımcı sayısına göre hesapla."""
    rows = list(records)
    all_participants = set(participant_ids) if participant_ids is not None else {r.participant_id for r in rows}
    if any(r.participant_id not in all_participants for r in rows):
        raise ValueError("Kodlanmış katılımcı, katılımcı evreninde yok.")
    by_code: dict[str, set[str]] = {}
    for row in rows:
        by_code.setdefault(row.code, set()).add(row.participant_id)
    n = len(all_participants)
    return [{"code": code, "n_participants": len(ids),
             "percent": round(100 * len(ids) / n, 2) if n else 0.0}
            for code, ids in sorted(by_code.items())]
