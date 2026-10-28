"""Optional multi-word academic-term matching; NO changes to existing preprocessing or E5.

Explicit opt-in call required. Never treats candidate terminology as expert-validated.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DEFAULT_LEXICON = Path(__file__).resolve().parent / "veri" / "akademik_terimler_v1_1_deneysel.json"
_TOKEN = re.compile(r"(?u)\w+(?:['’]\w+)?")
_TR_MAP = str.maketrans({"I": "ı", "İ": "i"})

def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).translate(_TR_MAP).lower()

def load_lexicon(path=None) -> dict:
    with open(path or DEFAULT_LEXICON, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("varsayilan_etkin") is not False or data.get("uzman_onayi") is not False:
        raise ValueError("Experimental lexicon metadata unexpectedly changed")
    return data

def match_terms(text: str, lexicon=None) -> list[dict]:
    """Return longest non-overlapping whole-token spans; do not modify original text.

    Returns offsets in original text. Substring collisions (e.g. 'emek' inside
    'emekli') are excluded; longer phrases win. Optional normalization is only
    for lexical detection, not a claim of semantic equivalence.
    """
    data = lexicon if lexicon is not None else load_lexicon()
    raw_tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    normalized = [normalize(t) for t, _, _ in raw_tokens]
    entries = {}
    for entry in data["terimler"]:
        key = tuple(normalize(t) for t in _TOKEN.findall(entry["terim"]))
        if len(key) >= 2:
            entries[key] = entry
    results = []
    i = 0
    while i < len(raw_tokens):
        hits = [(len(key), entry) for key, entry in entries.items()
                if tuple(normalized[i:i+len(key)]) == key]
        if hits:
            width, entry = max(hits, key=lambda item: item[0])
            results.append({"terim": entry["terim"], "kanonik": entry["kanonik"],
                            "metin": text[raw_tokens[i][1]:raw_tokens[i+width-1][2]],
                            "baslangic": raw_tokens[i][1], "bitis": raw_tokens[i+width-1][2]})
            i += width
        else:
            i += 1
    return results
