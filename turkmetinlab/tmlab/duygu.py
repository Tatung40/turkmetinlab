"""SWNetTR++ Turkish lexicon-based sentiment signals (experimental).

Source: https://github.com/swnettr/SWNetTR
F. Sağlam, B. Genç, H. Sever (2019), Turkish J. Electrical Engineering & Computer Sciences 27, 1806–1820.
F. Sağlam, B. Genç, H. Sever (2016), IEEE/ACS AICCSA.
Only public lexicon is downloaded. User text never leaves the process.
"""
from __future__ import annotations
import csv
import io
import re
import unicodedata
import urllib.request
from functools import lru_cache

SOURCE = "https://raw.githubusercontent.com/swnettr/SWNetTR/69c303ba974aab1b862fd9cee97601477fe48453/SWNetTR%2B%2B.csv"
PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
TR = str.maketrans({"I": "ı", "İ": "i"})
MAX_BYTES = 3_000_000

def norm(value):
    return unicodedata.normalize("NFKC", str(value)).translate(TR).lower().strip()

def parse_lexicon(content: bytes) -> dict:
    if len(content) > MAX_BYTES:
        raise ValueError("Sözlük boyut sınırı aşıldı")
    txt = content.decode("utf-8-sig")
    lines = txt.splitlines()
    pos = next((i for i, line in enumerate(lines) if line.strip().upper() == "WORD;TONE;POLARITY"), None)
    if pos is None:
        raise ValueError("SWNetTR++ WORD;TONE;POLARITY başlığı bulunamadı")
    reader = csv.DictReader(lines[pos:], delimiter=";")
    result = {}
    for row in reader:
        term = norm(row.get("WORD") or "")
        if not term:
            continue
        try:
            tone = float((row["TONE"] or "").replace(",", "."))
            polarity = int(row["POLARITY"])
        except (ValueError, KeyError, TypeError):
            continue
        if not (-1 <= tone <= 1) or polarity not in (-1, 0, 1):
            continue
        key = tuple(PATTERN.findall(term))
        if key:
            result[key] = (tone, polarity)
    if not result:
        raise ValueError("Sözlükte geçerli veri bulunamadı")
    return result

@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    req = urllib.request.Request(SOURCE, headers={"User-Agent": "TurkMetinLab/2.5-public-lexicon"})
    with urllib.request.urlopen(req, timeout=25) as response:
        body = response.read(MAX_BYTES + 1)
    return parse_lexicon(body)

def analyze(text: str, lexicon: dict) -> dict:
    """Longest-match token spans. Simple lexical signals, NOT verified sentiment labels."""
    source = str(text)
    matches = list(PATTERN.finditer(source))
    toks = [norm(x.group()) for x in matches]
    maxlen = max(map(len, lexicon), default=1)
    i, pos, neg, total, entries = 0, 0, 0, 0.0, []
    while i < len(toks):
        selected = None
        for n in range(min(maxlen, len(toks)-i), 0, -1):
            v = lexicon.get(tuple(toks[i:i+n]))
            if v is not None:
                selected = (n, v)
                break
        if selected is None:
            i += 1
            continue
        n, (tone, polarity) = selected
        if polarity > 0:
            pos += 1
        elif polarity < 0:
            neg += 1
        total += tone
        entries.append({"Sözcük/ifade": source[matches[i].start():matches[i+n-1].end()],
                        "Sözlük tonu": tone, "Sözlük kutbu": polarity,
                        "Başlangıç": matches[i].start(), "Bitiş": matches[i+n-1].end()})
        i += n
    return {"Eşleşen ifade": len(entries), "Sözcük sayısı": len(toks),
            "Kapsama (%)": round(100 * sum(len(PATTERN.findall(x["Sözcük/ifade"])) for x in entries) / len(toks), 2) if toks else 0.0,
            "Olumlu eşleşme": pos, "Olumsuz eşleşme": neg,
            "Ton toplamı": round(total, 5),
            "Ton ortalaması": round(total / len(entries), 5) if entries else 0.0,
            "Eşleşmeler": entries}

def analyze_corpus(texts, names, lexicon):
    summaries, details = [], []
    for name, content in zip(names, texts):
        r = analyze(content, lexicon)
        summaries.append({"Belge": str(name), **{k:v for k,v in r.items() if k != "Eşleşmeler"}})
        for e in r["Eşleşmeler"]:
            details.append({"Belge": str(name), **e})
    return summaries, details
