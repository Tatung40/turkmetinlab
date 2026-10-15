"""Modül 3 — Konu modellemesi testleri.

1. Elle hesaplanan NPMI, UMass, çeşitlilik ve Ortalama Jaccard örnekleri.
2. Bağımsız (saf Python) tutarlılık uygulamasıyla karşılaştırma.
3. Ekilmiş konular: konuları bilinen yapay bir derlemde LDA ve NMF doğru konuları geri buluyor mu?
"""

import itertools
import math

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from tmlab.konu import (KonuAyarlari, KonuHatasi, belge_konu_tablosu, cesitlilik, karsilastirma_grafigi,
                        konu_adlari, konu_eslestir, konu_grafigi, konu_modeli, konu_sayisi_karsilastir,
                        konu_sozcuk_tablosu, ortalama_jaccard, tipik_belgeler, tutarlilik, uygunluk_denetimi)
from tmlab.preprocess import IslenmisBelge


def B(*tokenler):
    return IslenmisBelge([list(tokenler)])


# --- Elle hesaplanabilir ölçütler -----------------------------------------

def _ikili(belgeler, sozcukler):
    import scipy.sparse as sp

    M = np.array([[1.0 if w in b else 0.0 for w in sozcukler] for b in belgeler])
    return sp.csc_matrix(M)


def test_elle_npmi_ve_umass():
    """4 belge: {a,b}, {a,b}, {a,c}, {c}. Konu sözcükleri sırasıyla [a, b, c].

    P(a)=3/4, P(b)=2/4, P(c)=2/4, P(a,b)=2/4, P(a,c)=1/4, P(b,c)=0
    NPMI(a,b) = ln(0,5 / (0,75·0,5)) / −ln 0,5 ; NPMI(a,c) = ln(0,25 / (0,75·0,5)) / −ln 0,25 ; NPMI(b,c) = −1
    UMass: ln((D(b,a)+1)/D(a)) = ln(3/3); ln((D(c,a)+1)/D(a)) = ln(2/3); ln((D(c,b)+1)/D(b)) = ln(1/2)
    """
    Bm = _ikili([{"a", "b"}, {"a", "b"}, {"a", "c"}, {"c"}], ["a", "b", "c"])
    t = tutarlilik([[0, 1, 2]], Bm)
    npmi = (math.log(0.5 / 0.375) / -math.log(0.5) + math.log(0.25 / 0.375) / -math.log(0.25) - 1) / 3
    umass = (math.log(1.0) + math.log(2 / 3) + math.log(1 / 2)) / 3
    assert t.loc[0, "NPMI tutarlılığı"] == pytest.approx(npmi, abs=1e-12)
    assert t.loc[0, "UMass tutarlılığı"] == pytest.approx(umass, abs=1e-12)


def test_tutarlilik_bagimsiz_uygulamayla_ayni():
    rng = np.random.default_rng(3)
    belgeler = [set(rng.choice(list("abcdefghij"), size=int(rng.integers(2, 6)), replace=False)) for _ in range(60)]
    sozcukler = list("abcdefghij")
    Bm = _ikili(belgeler, sozcukler)
    konular = [[0, 3, 5, 7], [1, 2, 9], [4, 6, 8, 0, 2]]
    t = tutarlilik(konular, Bm)
    D = len(belgeler)
    for k, liste in enumerate(konular):
        w = [sozcukler[i] for i in liste]
        npmi, umass = [], []
        for m in range(1, len(w)):
            for l in range(m):
                d_l = sum(w[l] in b for b in belgeler)
                d_m = sum(w[m] in b for b in belgeler)
                d_lm = sum(w[l] in b and w[m] in b for b in belgeler)
                npmi.append(-1.0 if d_lm == 0 else math.log((d_lm / D) / ((d_l / D) * (d_m / D))) / -math.log(d_lm / D))
                umass.append(math.log((d_lm + 1) / d_l))
        assert t.loc[k, "NPMI tutarlılığı"] == pytest.approx(np.mean(npmi), abs=1e-12)
        assert t.loc[k, "UMass tutarlılığı"] == pytest.approx(np.mean(umass), abs=1e-12)


def test_ortalama_jaccard_ve_eslestirme():
    # derinlik 1: {a}/{a}=1 ; 2: {a,b}/{a,c}=1/3 ; 3: {a,b,c}/{a,c,b}=1 → ortalama 7/9
    assert ortalama_jaccard(["a", "b", "c"], ["a", "c", "b"]) == pytest.approx(7 / 9)
    assert ortalama_jaccard(["a", "b"], ["c", "d"]) == 0.0
    ref = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
    puan, eslesen = konu_eslestir(ref, [ref[2], ref[0], ref[1]])
    assert puan.tolist() == [1.0, 1.0, 1.0] and eslesen.tolist() == [1, 2, 0]


def test_cesitlilik():
    H = np.array([[0.5, 0.4, 0.1, 0.0], [0.5, 0.0, 0.1, 0.4]])
    # ilk 2 sözcük: {0,1} ve {0,3} → 3 benzersiz / 4
    assert cesitlilik(H, 2) == pytest.approx(0.75)


# --- Ekilmiş konular ------------------------------------------------------

KONULAR = {
    "deprem": "deprem fay zemin sismik yapı hasar ivme kayma büyüklük artçı sarsıntı yer kabuk tsunami "
              "dalga levha kırık bina güçlendirme risk",
    "enerji": "enerji güneş rüzgâr elektrik panel türbin şebeke batarya depolama verim güç santral "
              "yenilenebilir fotovoltaik akım gerilim üretim tüketim yakıt karbon",
    "saglik": "hasta hemşire tedavi hastane ağrı ilaç klinik tanı bakım doktor kronik hastalık sağlık "
              "ameliyat semptom yoğun bakım terapi aşı virüs",
    "egitim": "öğrenci öğretmen okul ders sınıf müfredat öğrenme başarı ölçme sınav eğitim ödev "
              "kampüs akademik okuma yazma matematik veli rehberlik kütüphane",
}


ZIPF = (1 / np.arange(1, 21)) / (1 / np.arange(1, 21)).sum()


@pytest.fixture(scope="module")
def ekilmis():
    rng = np.random.default_rng(42)
    adlar = list(KONULAR)
    sozluk = {k: v.split() for k, v in KONULAR.items()}
    gurultu = "çalışma sonuç analiz yöntem veri model farklı önemli yüksek düşük".split()
    belgeler, etiket = [], []
    for i in range(240):
        k = i % 4
        ikinci = (k + 1 + int(rng.integers(0, 3))) % 4
        # Konu içi sözcük olasılıkları Zipf dağılımlı (gerçek konular gibi çarpık): ilk sözcükler belirgin
        t = list(rng.choice(sozluk[adlar[k]], 45, p=ZIPF)) + list(rng.choice(sozluk[adlar[ikinci]], 8, p=ZIPF)) + \
            list(rng.choice(gurultu, 10))
        rng.shuffle(t)
        belgeler.append(B(*t))
        etiket.append(k)
    return belgeler, np.array(etiket), sozluk, adlar


@pytest.mark.parametrize("yontem", ["lda", "nmf"])
def test_ekilmis_konular_geri_bulunur(ekilmis, yontem):
    belgeler, etiket, sozluk, adlar = ekilmis
    r = konu_modeli(belgeler, KonuAyarlari(yontem, 4, en_az_belge=1, en_fazla_belge_orani=1.0), kararlilik_tekrari=3)
    gercek = [sozluk[a] for a in adlar]
    puan, eslesen = konu_eslestir([g[:10] for g in gercek], r.ust_sozcukler(10))
    # Her gerçek konu için: eşleşen model konusunun ilk 10 sözcüğünün hepsi o konudan ve gerçek
    # ilk 10 sözcüğün (en olası) en az 8'i yeniden bulunuyor
    for g, j in zip(gercek, eslesen):
        assert set(r.ust_sozcukler(10)[j]) <= set(g), (yontem, r.ust_sozcukler(10)[j])
        assert len(set(r.ust_sozcukler(10)[j]) & set(g[:10])) >= 8, (yontem, r.ust_sozcukler(10)[j])
    ari = adjusted_rand_score(etiket[r.belge_indisleri], np.argmax(r.belge_konu, axis=1))
    assert ari > 0.9, ari
    assert r.olcutler["Ortalama kararlılık (AJ)"] > 0.8
    assert r.olcutler["Ortalama NPMI tutarlılığı"] > 0


@pytest.mark.parametrize("yontem", ["lda", "nmf"])
def test_belirlenimci(ekilmis, yontem):
    belgeler = ekilmis[0]
    a = konu_modeli(belgeler, KonuAyarlari(yontem, 3, en_az_belge=1))
    b = konu_modeli(belgeler, KonuAyarlari(yontem, 3, en_az_belge=1))
    assert np.array_equal(a.konu_sozcuk, b.konu_sozcuk) and np.array_equal(a.belge_konu, b.belge_konu)


def test_olasilik_ve_oran_toplamlari(ekilmis):
    lda = konu_modeli(ekilmis[0], KonuAyarlari("lda", 4, en_az_belge=1))
    assert np.allclose(lda.konu_sozcuk.sum(axis=1), 1) and np.allclose(lda.belge_konu.sum(axis=1), 1)
    nmf = konu_modeli(ekilmis[0], KonuAyarlari("nmf", 4, en_az_belge=1))
    assert np.allclose(nmf.belge_konu.sum(axis=1), 1) and (nmf.konu_sozcuk >= 0).all()
    assert "Şaşkınlık (perplexity)" in lda.olcutler and "Yeniden oluşturma hatası" in nmf.olcutler


# --- Tablolar ve adlandırma -------------------------------------------------

def test_tablolar_ve_arastirmaci_adlari(ekilmis):
    belgeler = ekilmis[0]
    r = konu_modeli(belgeler, KonuAyarlari("nmf", 4, en_az_belge=1, ust_sozcuk=7))
    adlar = konu_adlari(r, {0: "  Deprem riski ", 2: ""})
    assert adlar == ["Deprem riski", "Konu 2", "Konu 3", "Konu 4"]
    ks = konu_sozcuk_tablosu(r, adlar)
    assert len(ks) == 28 and ks["Konu adı"].iloc[0] == "Deprem riski" and "Ağırlık" in ks.columns
    bk = belge_konu_tablosu(r, [f"B{i}" for i in range(len(belgeler))], adlar)
    assert len(bk) == len(belgeler) and "1: Deprem riski" in bk.columns
    assert len(tipik_belgeler(r, 0, 3)) == 3
    assert len(konu_grafigi(r, adlar).data) == 4


def test_bos_belgeler_modele_girmez():
    belgeler = [B("a", "b", "c"), B(), B("a", "b", "d"), B("c", "d", "e"), B("e", "f", "a"), B("b", "f", "e")]
    r = konu_modeli(belgeler, KonuAyarlari("nmf", 2, en_az_belge=1))
    assert 1 not in r.belge_indisleri.tolist() and r.olcutler["Modeldışı kalan (boş) belge"] == 1


# --- Uygunluk denetimi ------------------------------------------------------

def test_uygunluk_uyarilari_ve_hatalar(ekilmis):
    with pytest.raises(KonuHatasi, match="en az 3"):
        uygunluk_denetimi([B("a"), B("b")], 2)
    with pytest.raises(KonuHatasi, match="küçük olmalıdır"):
        uygunluk_denetimi([B("a"), B("b"), B("c")], 3)
    u = uygunluk_denetimi([B("a", "b")] * 10, 4)
    assert any("çok küçük" in x for x in u) and any("kısa" in x for x in u) and any("yüksek" in x for x in u)
    with pytest.raises(KonuHatasi):
        konu_modeli([B("a", "b")] * 5, KonuAyarlari("nmf", 2, en_az_belge=100))


@pytest.mark.parametrize("kw", [dict(yontem="bert"), dict(konu_sayisi=1), dict(konu_sayisi=31),
                                dict(en_fazla_belge_orani=0), dict(ust_sozcuk=2)])
def test_gecersiz_ayarlar(kw):
    with pytest.raises(ValueError):
        KonuAyarlari(**kw)


def test_konu_sayisi_karsilastirma(ekilmis):
    t = konu_sayisi_karsilastir(ekilmis[0], KonuAyarlari("nmf", 2, en_az_belge=1), [2, 4, 6], kararlilik_tekrari=2)
    assert t["Konu sayısı"].tolist() == [2, 4, 6]
    assert "Ortalama kararlılık (AJ)" in t.columns and "Ortalama NPMI tutarlılığı" in t.columns
    # Gerçek konu sayısı 4: kararlılık 4'te 6'dakinden yüksek olmalı
    k = t.set_index("Konu sayısı")["Ortalama kararlılık (AJ)"]
    assert k[4] > k[6]
    assert len(karsilastirma_grafigi(t).data) == len([c for c in t.columns if c != "Konu sayısı"])


# --- Uygulama içinde -------------------------------------------------------

def _uygulama():
    import os

    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"), default_timeout=300)


def test_uygulamada_konu_modeli_ve_adlandirma():
    at = _uygulama().run()
    assert not at.exception and not at.error
    assert any("doğrulanmış nitel temalar değildir" in i.value for i in at.info)
    assert any("Derlem çok küçük" in w.value for w in at.warning)  # 16 belgelik örnek veri
    assert not any(m.label == "NPMI tutarlılığı" for m in at.metric)  # düğmeye basılmadan çalışmaz
    at.button(key="konu_calistir").click().run()
    assert not at.exception and not at.error
    assert any(m.label == "Kararlılık" for m in at.metric)
    ad_kutusu = [t for t in at.text_input if t.key and t.key.startswith("konu_ad_")][0]
    ad_kutusu.input("Sağlık hizmetleri").run()
    assert not at.exception
    assert any("Sağlık hizmetleri" in e.label for e in at.expander)  # araştırmacının verdiği ad kullanılıyor


def test_uygulamada_lda_ve_karsilastirma():
    at = _uygulama().run()
    at.radio(key="konu_yontem").set_value("lda").run()
    at.button(key="konu_calistir").click().run()
    assert not at.exception and not at.error
    assert any("kararsız" in w.value for w in at.warning)  # küçük derlemde LDA kararsızlığı bildirilir
    at.slider(key="konu_aralik").set_value((2, 4)).run()
    at.button(key="konu_karsilastir").click().run()
    assert not at.exception and not at.error


def test_uygulamada_lda_buyuk_derlem_siniri(monkeypatch):
    import tmlab.konu as km

    monkeypatch.setattr(km, "LDA_SOZCUK_SINIRI", 100)
    at = _uygulama().run()
    at.radio(key="konu_yontem").set_value("lda").run()
    assert any("çok uzun sürer" in e.value for e in at.error)
    assert "konu_calistir" not in [b.key for b in at.button]
