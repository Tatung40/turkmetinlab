"""Sıklık, n-gram ve TF-IDF testleri.

TF-IDF sonuçları, kütüphaneden bağımsız olarak elle yazılmış formülle
karşılaştırılarak doğrulanır.
"""

import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from tmlab.analysis import (TfidfHatasi, belge_ngramlari, derlem_ozeti, ngram_sikligi, sozcuk_sikligi,
                            tfidf_hesapla, turkce_sirala_anahtari)
from tmlab.loader import dosya_oku
from tmlab.preprocess import IslenmisBelge, OnIslemeAyarlari, derlem_on_isle, turkce_kucuk_harf

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def belge(*bolutler):
    return IslenmisBelge([list(b) for b in bolutler])


# Elle hazırlanmış küçük derlem
D1 = belge(["elma", "armut", "elma"], ["elma", None, "kiraz"])
D2 = belge(["armut", "kiraz"])
D3 = belge(["çilek", "elma"])
DERLEM = [D1, D2, D3]


# --- Türkçe sıralama --------------------------------------------------------

def test_turkce_alfabe_sirasi():
    sozcukler = ["zeytin", "ılık", "çam", "cam", "ışık", "iğne", "öykü", "ozan", "şeker", "sabun", "ğ", "gül", "üzüm",
                 "uzun"]
    beklenen = ["cam", "çam", "gül", "ğ", "ılık", "ışık", "iğne", "ozan", "öykü", "sabun", "şeker", "uzun", "üzüm",
                "zeytin"]
    assert sorted(sozcukler, key=turkce_sirala_anahtari) == beklenen


# --- Sözcük sıklığı -------------------------------------------------------

def test_sozcuk_sikligi_degerleri():
    t = sozcuk_sikligi(DERLEM)
    satir = t.set_index("Sözcük")
    # elma: D1'de 3, D3'te 1 -> 4; toplam sözcük 3+2+2+2 = 9
    assert satir.loc["elma", "Sıklık"] == 4
    assert satir.loc["elma", "Belge sıklığı"] == 2
    assert satir.loc["elma", "Göreli sıklık (%)"] == pytest.approx(round(4 / 9 * 100, 4))
    assert satir.loc["elma", "Bin sözcükte"] == pytest.approx(round(4 / 9 * 1000, 4))
    assert satir.loc["elma", "Belge oranı (%)"] == pytest.approx(66.67)
    assert satir.loc["armut", "Sıklık"] == 2
    assert satir.loc["kiraz", "Sıklık"] == 2
    assert satir.loc["çilek", "Sıklık"] == 1
    assert t["Sıklık"].sum() == 9
    assert t["Göreli sıklık (%)"].sum() == pytest.approx(100, abs=0.01)


def test_sozcuk_sikligi_siralama():
    t = sozcuk_sikligi(DERLEM)
    # Sıklığa göre azalan; eşitlikte Türk alfabesi (armut < kiraz)
    assert t["Sözcük"].tolist() == ["elma", "armut", "kiraz", "çilek"]
    assert t["Sıra"].tolist() == [1, 2, 3, 4]


def test_sozcuk_sikligi_min_siklik_ve_bos():
    assert sozcuk_sikligi(DERLEM, min_siklik=2)["Sözcük"].tolist() == ["elma", "armut", "kiraz"]
    bos = sozcuk_sikligi([belge()])
    assert bos.empty and "Sözcük" in bos.columns


# --- N-gram -----------------------------------------------------------------

def test_ngram_bolut_ve_bosluk_saygisi():
    # D1: [elma armut elma] [elma None kiraz] -> "elma armut", "armut elma"; "elma kiraz" ÜRETİLMEMELİ
    assert belge_ngramlari(D1, 2) == ["elma armut", "armut elma"]
    assert belge_ngramlari(D1, 3) == ["elma armut elma"]
    assert belge_ngramlari(D1, 1) == ["elma", "armut", "elma", "elma", "kiraz"]
    with pytest.raises(ValueError):
        belge_ngramlari(D1, 0)


def test_ngram_sikligi():
    t = ngram_sikligi(DERLEM, 2)
    satir = t.set_index("N-gram")
    assert satir.loc["armut kiraz", "Sıklık"] == 1
    assert set(t["N-gram"]) == {"elma armut", "armut elma", "armut kiraz", "çilek elma"}
    assert "elma kiraz" not in set(t["N-gram"])
    assert t["Sıklık"].sum() == 4


def test_ngram_gercek_metin():
    belgeler = derlem_on_isle(["Uzaktan eğitim ve öğretim. Uzaktan eğitim süreci.", "Uzaktan eğitim"])
    t = ngram_sikligi(belgeler, 2, min_siklik=2)
    assert t["N-gram"].tolist() == ["uzaktan eğitim"]
    assert t.loc[0, "Sıklık"] == 3
    assert t.loc[0, "Belge sıklığı"] == 2
    # "eğitim öğretim" yapay ikilisi oluşmamalı (arada "ve" var)
    tum = ngram_sikligi(belgeler, 2)
    assert "eğitim öğretim" not in set(tum["N-gram"])
    # "eğitim uzaktan" cümle sınırını aşan ikili oluşmamalı
    assert "öğretim uzaktan" not in set(tum["N-gram"])


# --- Derlem özeti -----------------------------------------------------------

def test_derlem_ozeti():
    o = derlem_ozeti(DERLEM + [belge()])
    assert o["Belge sayısı"] == 4
    assert o["Ön işleme sonrası boş kalan belge"] == 1
    assert o["Toplam sözcük (ön işleme öncesi)"] == 10  # None dahil
    assert o["Analiz edilen sözcük (ön işleme sonrası)"] == 9
    assert o["Farklı sözcük sayısı"] == 4
    assert o["Tür/Sözcük oranı (TTR)"] == pytest.approx(round(4 / 9, 4))
    assert o["Tek kez geçen sözcük (hapax)"] == 1
    assert o["Belge başına ortalama sözcük"] == 3.0


# --- TF-IDF -----------------------------------------------------------------

def elle_tfidf(sayimlar, smooth=True, sublinear=False):
    """Kütüphaneden bağımsız TF-IDF (doğrulama için)."""
    N = len(sayimlar)
    terimler = {t for s in sayimlar for t in s}
    df = {t: sum(1 for s in sayimlar if t in s) for t in terimler}
    idf = {t: (math.log((1 + N) / (1 + df[t])) + 1) if smooth else (math.log(N / df[t]) + 1) for t in terimler}
    sonuc = []
    for s in sayimlar:
        v = {t: (1 + math.log(f) if sublinear else f) * idf[t] for t, f in s.items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        sonuc.append({t: x / norm for t, x in v.items()})
    return sonuc, idf


@pytest.mark.parametrize("smooth", [True, False])
@pytest.mark.parametrize("sublinear", [True, False])
def test_tfidf_elle_hesapla_eslesir(smooth, sublinear):
    s = tfidf_hesapla(DERLEM, ["A", "B", "C"], smooth_idf=smooth, sublinear_tf=sublinear, belge_basina_terim=50)
    beklenen, idf = elle_tfidf([Counter(b.sozcukler) for b in DERLEM], smooth, sublinear)
    for _, satir in s.belge_tablosu.iterrows():
        i = ["A", "B", "C"].index(satir["Belge"])
        assert satir["TF-IDF"] == pytest.approx(beklenen[i][satir["Terim"]], abs=1e-6)
    # IDF ve ortalama değerler
    d = s.derlem_tablosu.set_index("Terim")
    for t in idf:
        assert d.loc[t, "IDF"] == pytest.approx(idf[t], abs=1e-6)
        ort = sum(b.get(t, 0.0) for b in beklenen) / 3
        assert d.loc[t, "Ortalama TF-IDF"] == pytest.approx(ort, abs=1e-6)
        assert d.loc[t, "En yüksek TF-IDF"] == pytest.approx(max(b.get(t, 0.0) for b in beklenen), abs=1e-6)


def test_tfidf_l2_normu_bir():
    s = tfidf_hesapla(DERLEM)
    normlar = np.sqrt(np.asarray(s.matris.multiply(s.matris).sum(axis=1)).ravel())
    assert np.allclose(normlar, 1.0)


def test_tfidf_belge_sikligi_ve_sayim():
    s = tfidf_hesapla(DERLEM, ["A", "B", "C"])
    d = s.derlem_tablosu.set_index("Terim")
    assert d.loc["elma", "Belge sıklığı"] == 2
    a = s.belge_tablosu[s.belge_tablosu["Belge"] == "A"].set_index("Terim")
    assert a.loc["elma", "Belgedeki sıklık"] == 3


def test_tfidf_bos_belgeler_atlanir():
    s = tfidf_hesapla([D1, belge(), D2], ["A", "Boş", "B"])
    assert s.belge_adlari == ["A", "B"]
    assert s.analiz_edilen_belge == 2


def test_tfidf_tek_belge_hata():
    with pytest.raises(TfidfHatasi):
        tfidf_hesapla([D1])
    with pytest.raises(TfidfHatasi):
        tfidf_hesapla([D1, belge()])


def test_tfidf_min_belge_cok_yuksek():
    with pytest.raises(TfidfHatasi):
        tfidf_hesapla(DERLEM, min_belge=10)


def test_tfidf_min_belge_suzgeci():
    s = tfidf_hesapla(DERLEM, min_belge=2)
    assert set(s.terimler) == {"elma", "armut", "kiraz"}


def test_tfidf_bigram():
    s = tfidf_hesapla(DERLEM, ngram_ust=2)
    assert "elma armut" in set(s.terimler)
    assert "elma kiraz" not in set(s.terimler)


def test_tfidf_ad_uyusmazligi():
    with pytest.raises(ValueError):
        tfidf_hesapla(DERLEM, ["A"])


def test_tfidf_belirlenimci():
    a = tfidf_hesapla(DERLEM).derlem_tablosu
    b = tfidf_hesapla(DERLEM).derlem_tablosu
    pd.testing.assert_frame_equal(a, b)


# --- Örnek veri üzerinde bağımsız çapraz doğrulama -------------------------

def test_ornek_veride_sayimlar_bagimsiz_yontemle_eslesir():
    """Uygulamanın sıklıklarını, basit bir düzenli ifade sayımıyla karşılaştırır."""
    with open(os.path.join(KOK, "ornek_veri", "ornek_akademik_ozetler.csv"), "rb") as f:
        tablo = dosya_oku("ornek.csv", f.read()).tablo
    metinler = tablo["ozet"].tolist()
    t = sozcuk_sikligi(derlem_on_isle(metinler, OnIslemeAyarlari())).set_index("Sözcük")
    birlesik = turkce_kucuk_harf(" ".join(metinler))
    for sozcuk in ["öğrencilerin", "sağlık", "dijital", "yaşlı", "kuraklık", "uzaktan", "okuryazarlığı"]:
        beklenen = len(re.findall(rf"(?<![\w'-]){re.escape(sozcuk)}(?![\w-])", birlesik))
        assert beklenen > 0, sozcuk
        assert t.loc[sozcuk, "Sıklık"] == beklenen, sozcuk
