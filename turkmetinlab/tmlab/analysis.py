"""Sözcük sıklığı, n-gram ve TF-IDF analizleri.

Tüm sonuçlar ön işlenmiş gerçek veriden deterministik olarak hesaplanır;
rastgelelik içeren hiçbir adım yoktur.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import IslenmisBelge

# ---------------------------------------------------------------------------
# Türkçe alfabetik sıralama
# ---------------------------------------------------------------------------

_ALFABE = "abcçdefgğhıijklmnoöprsştuüvyz"
_SIRA = {h: i for i, h in enumerate(_ALFABE)}


def turkce_sirala_anahtari(sozcuk: str):
    """Türk alfabesine göre sıralama anahtarı (ç, c'den sonra; ı, i'den önce gelir)."""
    from .preprocess import turkce_kucuk_harf

    return tuple((_SIRA.get(c, 100 + ord(c))) for c in turkce_kucuk_harf(sozcuk))


def _sirala(tablo: pd.DataFrame, sayi_sutunu: str, metin_sutunu: str) -> pd.DataFrame:
    """Sayıya göre azalan, eşitlikte Türk alfabesine göre artan sıralama."""
    if tablo.empty:
        return tablo
    anahtar = tablo[metin_sutunu].map(turkce_sirala_anahtari)
    sira = sorted(range(len(tablo)), key=lambda i: (-tablo[sayi_sutunu].iat[i], anahtar.iat[i]))
    return tablo.iloc[sira].reset_index(drop=True)


# ---------------------------------------------------------------------------
# N-gram üretimi
# ---------------------------------------------------------------------------

def belge_ngramlari(belge: IslenmisBelge, n: int) -> List[str]:
    """Bir belgedeki n-gramları üretir.

    N-gramlar bölüt (cümle/ifade) sınırlarını aşmaz ve çıkarılmış bir sözcüğü
    (None) içeren pencereler atlanır.
    """
    if n < 1:
        raise ValueError("n en az 1 olmalıdır.")
    if n == 1:
        return belge.sozcukler
    sonuc = []
    for bolut in belge.bolutler:
        for i in range(len(bolut) - n + 1):
            pencere = bolut[i:i + n]
            if None not in pencere:
                sonuc.append(" ".join(pencere))
    return sonuc


# ---------------------------------------------------------------------------
# Derlem özeti
# ---------------------------------------------------------------------------

def derlem_ozeti(belgeler: Sequence[IslenmisBelge]) -> Dict[str, float]:
    sayac = Counter(t for b in belgeler for t in b.sozcukler)
    analiz_edilen = sum(sayac.values())
    farkli = len(sayac)
    bos = sum(1 for b in belgeler if not b.sozcukler)
    dolu = len(belgeler) - bos
    return {
        "Belge sayısı": len(belgeler),
        "Ön işleme sonrası boş kalan belge": bos,
        "Toplam sözcük (ön işleme öncesi)": sum(b.ham_sozcuk_sayisi for b in belgeler),
        "Analiz edilen sözcük (ön işleme sonrası)": analiz_edilen,
        "Farklı sözcük sayısı": farkli,
        "Tür/Sözcük oranı (TTR)": round(farkli / analiz_edilen, 4) if analiz_edilen else 0.0,
        "Tek kez geçen sözcük (hapax)": sum(1 for v in sayac.values() if v == 1),
        "Belge başına ortalama sözcük": round(analiz_edilen / dolu, 2) if dolu else 0.0,
    }


# ---------------------------------------------------------------------------
# Sıklık analizleri
# ---------------------------------------------------------------------------

def _siklik_tablosu(belge_terimleri: List[List[str]], terim_basligi: str, min_siklik: int = 1) -> pd.DataFrame:
    sayac = Counter()
    belge_sayac = Counter()
    for terimler in belge_terimleri:
        sayac.update(terimler)
        belge_sayac.update(set(terimler))
    toplam = sum(sayac.values())
    n_belge = len(belge_terimleri)
    satirlar = [
        (t, f, belge_sayac[t]) for t, f in sayac.items() if f >= min_siklik
    ]
    tablo = pd.DataFrame(satirlar, columns=[terim_basligi, "Sıklık", "Belge sıklığı"])
    if tablo.empty:
        return pd.DataFrame(columns=["Sıra", terim_basligi, "Sıklık", "Göreli sıklık (%)",
                                     "Bin sözcükte", "Belge sıklığı", "Belge oranı (%)"])
    tablo["Sıklık"] = tablo["Sıklık"].astype(int)
    tablo["Belge sıklığı"] = tablo["Belge sıklığı"].astype(int)
    tablo["Göreli sıklık (%)"] = (tablo["Sıklık"] / toplam * 100).round(4)
    tablo["Bin sözcükte"] = (tablo["Sıklık"] / toplam * 1000).round(4)
    tablo["Belge oranı (%)"] = (tablo["Belge sıklığı"] / n_belge * 100).round(2)
    tablo = _sirala(tablo, "Sıklık", terim_basligi)
    tablo.insert(0, "Sıra", range(1, len(tablo) + 1))
    return tablo[["Sıra", terim_basligi, "Sıklık", "Göreli sıklık (%)", "Bin sözcükte",
                  "Belge sıklığı", "Belge oranı (%)"]]


def sozcuk_sikligi(belgeler: Sequence[IslenmisBelge], min_siklik: int = 1) -> pd.DataFrame:
    """Sözcük sıklık tablosu.

    Göreli sıklık ve 'bin sözcükte' değerleri, ön işleme sonrası toplam sözcük
    sayısına göre hesaplanır.
    """
    return _siklik_tablosu([b.sozcukler for b in belgeler], "Sözcük", min_siklik)


def ngram_sikligi(belgeler: Sequence[IslenmisBelge], n: int = 2, min_siklik: int = 1) -> pd.DataFrame:
    """N-gram sıklık tablosu. Göreli sıklık, toplam n-gram sayısına göredir."""
    return _siklik_tablosu([belge_ngramlari(b, n) for b in belgeler], "N-gram", min_siklik)


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

@dataclass
class TfidfSonucu:
    derlem_tablosu: pd.DataFrame
    belge_tablosu: pd.DataFrame
    matris: object  # scipy seyrek matris (belge x terim)
    terimler: np.ndarray
    belge_adlari: List[str]
    analiz_edilen_belge: int


class TfidfHatasi(ValueError):
    pass


def tfidf_hesapla(
    belgeler: Sequence[IslenmisBelge],
    belge_adlari: Optional[Sequence[str]] = None,
    ngram_ust: int = 1,
    smooth_idf: bool = True,
    sublinear_tf: bool = False,
    min_belge: int = 1,
    belge_basina_terim: int = 10,
) -> TfidfSonucu:
    """TF-IDF ağırlıklarını hesaplar (scikit-learn TfidfVectorizer).

    Formül (scikit-learn):
      tf(t, d)  = t teriminin d belgesindeki ham sayısı
                  (sublinear_tf=True ise 1 + ln(tf))
      idf(t)    = ln((1 + N) / (1 + df(t))) + 1      (smooth_idf=True)
                  ln(N / df(t)) + 1                  (smooth_idf=False)
      tfidf     = tf * idf, ardından her belge vektörü L2 normuyla
                  (uzunluğu 1 olacak biçimde) normalleştirilir.
    N: boş olmayan belge sayısı, df(t): t'yi içeren belge sayısı.
    """
    if belge_adlari is None:
        belge_adlari = [f"Belge {i + 1}" for i in range(len(belgeler))]
    if len(belge_adlari) != len(belgeler):
        raise ValueError("Belge adı sayısı belge sayısıyla eşleşmiyor.")

    terim_listeleri, adlar = [], []
    for b, ad in zip(belgeler, belge_adlari):
        terimler = []
        for n in range(1, ngram_ust + 1):
            terimler.extend(belge_ngramlari(b, n))
        if terimler:
            terim_listeleri.append(terimler)
            adlar.append(str(ad))

    if len(terim_listeleri) < 2:
        raise TfidfHatasi("TF-IDF analizi için ön işleme sonrası en az 2 dolu belge gerekir. "
                          "Tek belgeniz varsa TXT yüklerken 'Her paragraf bir belge' seçeneğini deneyin.")

    vektorlestirici = TfidfVectorizer(
        analyzer=lambda x: x,  # ön işleme bizim tarafımızdan yapıldı
        lowercase=False,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
        norm="l2",
        min_df=min_belge,
    )
    try:
        matris = vektorlestirici.fit_transform(terim_listeleri)
    except ValueError as e:
        raise TfidfHatasi("Seçilen ayarlarla TF-IDF için yeterli terim kalmadı. "
                          "'En az kaç belgede geçsin' değerini düşürmeyi deneyin.") from e

    terimler = vektorlestirici.get_feature_names_out()
    idf = vektorlestirici.idf_
    df = np.asarray((matris > 0).sum(axis=0)).ravel()
    ortalama = np.asarray(matris.mean(axis=0)).ravel()
    en_yuksek = matris.max(axis=0).toarray().ravel()
    toplam = np.asarray(matris.sum(axis=0)).ravel()

    derlem = pd.DataFrame({
        "Terim": terimler,
        "Ortalama TF-IDF": np.round(ortalama, 6),
        "En yüksek TF-IDF": np.round(en_yuksek, 6),
        "Toplam TF-IDF": np.round(toplam, 6),
        "Belge sıklığı": df.astype(int),
        "IDF": np.round(idf, 6),
    })
    derlem = _sirala(derlem, "Ortalama TF-IDF", "Terim")
    derlem.insert(0, "Sıra", range(1, len(derlem) + 1))

    # Belge bazında en ayırt edici terimler
    satirlar = []
    csr = matris.tocsr()
    for i, ad in enumerate(adlar):
        sayac = Counter(terim_listeleri[i])
        satir = csr.getrow(i)
        idx, deger = satir.indices, satir.data
        sira = sorted(range(len(idx)), key=lambda k: (-deger[k], turkce_sirala_anahtari(terimler[idx[k]])))
        for r, k in enumerate(sira[:belge_basina_terim], start=1):
            t = terimler[idx[k]]
            satirlar.append({"Belge": ad, "Sıra": r, "Terim": t, "TF-IDF": round(float(deger[k]), 6),
                             "Belgedeki sıklık": sayac[t], "Belge sıklığı": int(df[idx[k]])})
    belge_tablo = pd.DataFrame(satirlar, columns=["Belge", "Sıra", "Terim", "TF-IDF",
                                                  "Belgedeki sıklık", "Belge sıklığı"])
    return TfidfSonucu(derlem, belge_tablo, matris, terimler, adlar, len(adlar))

