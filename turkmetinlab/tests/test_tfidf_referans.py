"""TF-IDF ve sözcük ayrımı: Türkçe referans verilerle karşılaştırmalı testler.

Referanslar:
1. UD Turkish-BOUN (altın standart sözcük ayrımı, 400 cümle, 5 tür: biyografi, deneme, bilgilendirici,
   haber, popüler). Belgeler: her türün ardışık 10'ar cümlesi.
2. Kütüphaneden (scikit-learn) bağımsız, saf Python ile yazılmış TF-IDF uygulaması.
3. Örnek veri seti için bağımsız uygulamayla üretilip dosyaya kaydedilmiş beklenen sonuçlar
   (tests/veri/ornek_tfidf_beklenen.json); ön işleme ya da TF-IDF davranışındaki istenmeyen
   değişiklikleri yakalar.
"""

import json
import math
import os
import random
from collections import Counter, defaultdict

import pytest

from tmlab.analysis import tfidf_hesapla
from tmlab.preprocess import IslenmisBelge, OnIslemeAyarlari, derlem_on_isle, on_isle, turkce_kucuk_harf

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEKLENEN = os.path.join(KOK, "tests", "veri", "ornek_tfidf_beklenen.json")

# Altın standartla karşılaştırma için ön işleme: yalnızca sözcük ayrımı (hiçbir sözcük çıkarılmaz)
AYRIM = OnIslemeAyarlari(durak_sozcuk_genel=False, min_uzunluk=1, sayilari_kaldir=False,
                         kesme_isareti="birlestir", tireli_birlesik_koru=False, url_eposta_kaldir=False,
                         bolutleme="cumle")


def bagimsiz_tfidf(belgeler, smooth=True, sublinear=False):
    """scikit-learn'den bağımsız TF-IDF: belge listesi (sözcük listeleri) -> [{terim: değer}]."""
    N = len(belgeler)
    sayimlar = [Counter(b) for b in belgeler]
    df = Counter(t for s in sayimlar for t in s)
    idf = {t: (math.log((1 + N) / (1 + d)) + 1) if smooth else (math.log(N / d) + 1) for t, d in df.items()}
    sonuc = []
    for s in sayimlar:
        v = {t: (1 + math.log(f) if sublinear else f) * idf[t] for t, f in s.items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        sonuc.append({t: x / norm for t, x in v.items()})
    return sonuc


def _ud_belgeleri(ud, boyut=10):
    tur_cumleleri = defaultdict(list)
    for c in ud:
        tur_cumleleri[c["tur"]].append(c)
    belgeler = []
    for tur in sorted(tur_cumleleri):
        cs = tur_cumleleri[tur]
        for i in range(0, len(cs) - boyut + 1, boyut):
            parca = cs[i:i + boyut]
            altin = [turkce_kucuk_harf(t["bicim"]).replace("'", "")
                     for c in parca for t in c["sozcukler"] if t["upos"] != "PUNCT"]
            belgeler.append({"ad": f"{tur}-{i // boyut + 1}", "metin": "\n\n".join(c["metin"] for c in parca),
                             "altin": altin})
    return belgeler


# --- 1. Sözcük ayrımı altın standarda karşı ---------------------------------

def test_sozcuk_ayrimi_ud_altin_standartla_uyumlu(ud):
    biz, altin = Counter(), Counter()
    for c in ud:
        biz.update(on_isle(c["metin"], AYRIM).sozcukler)
        altin.update(turkce_kucuk_harf(t["bicim"]).replace("'", "") for t in c["sozcukler"] if t["upos"] != "PUNCT")
    ortak = sum((biz & altin).values())
    kesinlik, duyarlilik = ortak / sum(biz.values()), ortak / sum(altin.values())
    f1 = 2 * kesinlik * duyarlilik / (kesinlik + duyarlilik)
    assert sum(altin.values()) > 4000
    assert f1 >= 0.99, (kesinlik, duyarlilik)  # ölçülen: 0,9974


# --- 2. TF-IDF: altın sözcüklerle, bağımsız uygulamaya karşı birebir --------

@pytest.mark.parametrize("smooth, sublinear", [(True, False), (False, False), (True, True), (False, True)])
def test_tfidf_ud_verisinde_bagimsiz_uygulamayla_ayni(ud, smooth, sublinear):
    belgeler = _ud_belgeleri(ud)
    assert len(belgeler) >= 35
    islenmis = [IslenmisBelge([b["altin"]]) for b in belgeler]
    s = tfidf_hesapla(islenmis, [b["ad"] for b in belgeler], smooth_idf=smooth, sublinear_tf=sublinear,
                      belge_basina_terim=10_000)
    beklenen = bagimsiz_tfidf([b["altin"] for b in belgeler], smooth, sublinear)
    assert len(s.terimler) > 1500
    karsilastirilan = 0
    for _, satir in s.belge_tablosu.iterrows():
        i = s.belge_adlari.index(satir["Belge"])
        assert satir["TF-IDF"] == pytest.approx(beklenen[i][satir["Terim"]], abs=1e-6)
        karsilastirilan += 1
    # Belge tablosu her belgenin tüm terimlerini içermeli (belge_basina_terim büyük verildi)
    assert karsilastirilan == sum(len(b) for b in beklenen)


def test_tfidf_ud_uctan_uca_ham_metinden(ud):
    """Ham metinden (kendi sözcük ayrımımızla) hesaplanan en ayırt edici terimler, altın sözcük
    ayrımıyla hesaplananlarla neredeyse aynı olmalı."""
    belgeler = _ud_belgeleri(ud)
    ham = tfidf_hesapla(derlem_on_isle([b["metin"] for b in belgeler], AYRIM), [b["ad"] for b in belgeler])
    altin = tfidf_hesapla([IslenmisBelge([b["altin"]]) for b in belgeler], [b["ad"] for b in belgeler])
    ortusme = []
    for ad in altin.belge_adlari:
        a = set(altin.belge_tablosu[altin.belge_tablosu["Belge"] == ad]["Terim"])
        h = set(ham.belge_tablosu[ham.belge_tablosu["Belge"] == ad]["Terim"])
        ortusme.append(len(a & h) / len(a))
    assert sum(ortusme) / len(ortusme) >= 0.95


# --- 3. Matematiksel özellikler gerçek Türkçe veride -------------------------

def test_belge_sirasi_sonucu_degistirmez(ud):
    belgeler = _ud_belgeleri(ud)
    islenmis = [IslenmisBelge([b["altin"]]) for b in belgeler]
    adlar = [b["ad"] for b in belgeler]
    a = tfidf_hesapla(islenmis, adlar, belge_basina_terim=5).belge_tablosu
    sira = list(range(len(belgeler)))
    random.Random(7).shuffle(sira)
    b = tfidf_hesapla([islenmis[i] for i in sira], [adlar[i] for i in sira], belge_basina_terim=5).belge_tablosu
    a = a.sort_values(["Belge", "Sıra"]).reset_index(drop=True)
    b = b.sort_values(["Belge", "Sıra"]).reset_index(drop=True)
    assert a.equals(b)


def test_belgeyi_iki_kez_yazmak_normalize_degeri_degistirmez(ud):
    """Ham TF ve L2 normunda belgeyi kendisiyle birleştirmek yönü değiştirmez (değerler aynı kalır)."""
    belgeler = _ud_belgeleri(ud)
    tek = [IslenmisBelge([b["altin"]]) for b in belgeler]
    cift = [IslenmisBelge([b["altin"] * 2]) for b in belgeler]
    adlar = [b["ad"] for b in belgeler]
    a = tfidf_hesapla(tek, adlar, belge_basina_terim=8).belge_tablosu
    b = tfidf_hesapla(cift, adlar, belge_basina_terim=8).belge_tablosu
    assert (a["Terim"] == b["Terim"]).all()
    assert ((a["TF-IDF"] - b["TF-IDF"]).abs() < 1e-9).all()
    assert (b["Belgedeki sıklık"] == 2 * a["Belgedeki sıklık"]).all()


# --- 4. Örnek veri seti: kayıtlı beklenen sonuçlar --------------------------

def _ornek_belgeleri():
    import csv

    with open(os.path.join(KOK, "ornek_veri", "ornek_akademik_ozetler.csv"), encoding="utf-8-sig") as f:
        satirlar = list(csv.DictReader(f))
    return [r["kimlik"] for r in satirlar], derlem_on_isle([r["ozet"] for r in satirlar], OnIslemeAyarlari())


def test_ornek_veri_kayitli_sonuclarla_ayni():
    with open(BEKLENEN, encoding="utf-8") as f:
        beklenen = json.load(f)
    adlar, belgeler = _ornek_belgeleri()
    s = tfidf_hesapla(belgeler, adlar, belge_basina_terim=5)
    for ad in adlar:
        bizim = s.belge_tablosu[s.belge_tablosu["Belge"] == ad]
        assert bizim["Terim"].tolist() == [t for t, _ in beklenen["belgeler"][ad]], ad
        for (_, deger), gercek in zip(beklenen["belgeler"][ad], bizim["TF-IDF"]):
            assert gercek == pytest.approx(deger, abs=1e-6)
    ust = s.derlem_tablosu.head(10)
    assert ust["Terim"].tolist() == [t for t, _ in beklenen["derlem_ilk10"]]
    for (_, deger), gercek in zip(beklenen["derlem_ilk10"], ust["Ortalama TF-IDF"]):
        assert gercek == pytest.approx(deger, abs=1e-6)


def beklenen_dosyasini_olustur():  # pragma: no cover - yalnızca bilinçli güncelleme için
    """Beklenen sonuçları scikit-learn KULLANMADAN, bağımsız uygulamayla üretir.

    Çalıştırma: python -c "from tests.test_tfidf_referans import beklenen_dosyasini_olustur as f; f()"
    """
    from tmlab.analysis import turkce_sirala_anahtari

    adlar, belgeler = _ornek_belgeleri()
    degerler = bagimsiz_tfidf([b.sozcukler for b in belgeler])
    sonuc = {"aciklama": "Örnek veri, varsayılan ön işleme, yumuşatılmış IDF, ham TF, L2. Bağımsız uygulamayla "
                         "üretildi (tests/test_tfidf_referans.py: bagimsiz_tfidf).", "belgeler": {}}
    for ad, v in zip(adlar, degerler):
        sirali = sorted(v.items(), key=lambda kv: (-round(kv[1], 12), turkce_sirala_anahtari(kv[0])))[:5]
        sonuc["belgeler"][ad] = [[t, round(x, 9)] for t, x in sirali]
    ort = Counter()
    for v in degerler:
        for t, x in v.items():
            ort[t] += x / len(degerler)
    sirali = sorted(ort.items(), key=lambda kv: (-round(kv[1], 12), turkce_sirala_anahtari(kv[0])))[:10]
    sonuc["derlem_ilk10"] = [[t, round(x, 9)] for t, x in sirali]
    with open(BEKLENEN, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=1)
