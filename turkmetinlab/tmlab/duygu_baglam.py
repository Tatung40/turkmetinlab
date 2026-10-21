"""Dataset-independent review flags for separate sentiment methods.

Document frequency identifies common matched terms for human inspection ONLY.
It never strips words or claims semantic correctness.
"""
from collections import Counter
import re
import unicodedata
import pandas as pd

WORD = re.compile(r"[^\\W_]+", re.UNICODE)
TR = str.maketrans({"I": "ı", "İ": "i"})

def norm(value):
    value = unicodedata.normalize("NFKC", str(value)).translate(TR).lower()
    return unicodedata.normalize("NFD", value).replace("\u0307", "")

def frequent_evidence(texts, details, min_docs=3, min_ratio=0.20):
    columns = ["Sözcük", "Belge frekansı", "Belge oranı (%)", "Eşleşme sayısı"]
    n = len(texts)
    if n == 0:
        return pd.DataFrame(columns=columns)
    document_frequency = Counter()
    for text in texts:
        document_frequency.update(set(WORD.findall(norm(text))))
    match_frequency = Counter(norm(item.get("Sözcük/ifade", "")) for item in details)
    results = []
    for word, occurrences in match_frequency.items():
        if not word or len(WORD.findall(word)) != 1:
            continue
        doc_count = document_frequency[word]
        if doc_count >= min_docs and doc_count / n >= min_ratio:
            results.append({"Sözcük": word, "Belge frekansı": doc_count,
                            "Belge oranı (%)": round(100 * doc_count / n, 2),
                            "Eşleşme sayısı": occurrences})
    return pd.DataFrame(results, columns=columns).sort_values("Eşleşme sayısı", ascending=False)

def evaluate(names, sw=None, tes=None, electra=None):
    n = len(names)
    for data in (sw, tes, electra):
        if data is not None and len(data) != n:
            raise ValueError("Duygu sonuçları belge sayısıyla uyuşmuyor")
    results = []
    for i, name in enumerate(names):
        warnings = []
        model = electra[i] if electra is not None else None
        if model is None:
            warnings.append("Bağlamsal model çalıştırılmadı")
        else:
            if model.get("Kesildi mi"):
                warnings.append("64 token kırpması")
            score = model.get("En yüksek model skoru")
            if isinstance(score, (int, float)) and score < 0.65:
                warnings.append("Düşük model skoru: deneysel eşik")
        if sw is not None and (sw[i].get("Kapsama (%)") or 0) < 10:
            warnings.append("SWNetTR++ düşük kapsama")
        if tes is not None and (tes[i].get("Kapsama (%)") or 0) < 10:
            warnings.append("TES-Lex düşük kapsama")
        results.append({"Belge sırası": i + 1, "Belge": str(name),
                        "ELECTRA tahmini": model.get("Etiket") if model else "Çalıştırılmadı",
                        "İnceleme gerekli": bool(warnings),
                        "Gerekçeler": "; ".join(warnings) or "Otomatik uyarı yok"})
    return pd.DataFrame(results)
