"""Türkçe ön işleme testleri."""

import math

import pytest

from tmlab.preprocess import (OnIslemeAyarlari, derlem_on_isle, on_isle, ozel_durak_sozcukleri_ayristir,
                              turkce_kucuk_harf, unicode_normallestir)
from tmlab.stopwords import GENEL_DURAK_SOZCUKLER, stopword_set

DURAKSIZ = OnIslemeAyarlari(durak_sozcuk_genel=False, min_uzunluk=1)


def kelimeler(metin, **kw):
    return on_isle(metin, OnIslemeAyarlari(**kw)).sozcukler


# --- Büyük/küçük harf ------------------------------------------------------

@pytest.mark.parametrize("girdi, beklenen", [
    ("IŞIK", "ışık"),
    ("İSTANBUL", "istanbul"),
    ("ILIK İĞNE", "ılık iğne"),
    ("Iğdır", "ığdır"),
    ("ÇĞÖŞÜ", "çğöşü"),
    ("KIRIKKALE", "kırıkkale"),
    ("i̇stanbul", "istanbul"),   # ayrık yazılmış i + birleşen nokta
    ("İSTANBUL", "istanbul"),   # ayrık yazılmış İ
])
def test_turkce_kucuk_harf(girdi, beklenen):
    assert turkce_kucuk_harf(girdi) == beklenen


def test_python_lower_hatasi_tekrarlanmiyor():
    # Python'un standart lower() işlevi "I" -> "i" yapar; bu hata bizde olmamalı.
    assert "I".lower() == "i"
    assert turkce_kucuk_harf("I") == "ı"
    assert kelimeler("ISI", min_uzunluk=1) == ["ısı"]


def test_turkce_karakterler_korunur():
    sonuc = kelimeler("Çağdaş öğrenciler ılık şehirde güzel günler geçirdi")
    assert sonuc == ["çağdaş", "öğrenciler", "ılık", "şehirde", "güzel", "günler", "geçirdi"]


def test_kucuk_harf_kapali():
    assert kelimeler("Eğitim eğitim", kucuk_harf=False) == ["Eğitim", "eğitim"]


def test_kucuk_harf_kapaliyken_durak_sozcuk_buyuk_harfe_duyarsiz():
    assert kelimeler("Bu VE şu Eğitim", kucuk_harf=False) == ["Eğitim"]


# --- Unicode normalleştirme -------------------------------------------------

def test_yumusak_tire_ve_bitisik_harf():
    assert unicode_normallestir("eği­tim ﬁkir") == "eğitim fikir"


def test_satir_sonu_tiresi_birlestirilir():
    assert kelimeler("öğren-\nciler") == ["öğrenciler"]


def test_tipografik_kesme_isareti():
    assert kelimeler("Türkiye’nin") == ["türkiye"]
    assert kelimeler("Ankaraʼda") == ["ankara"]


# --- Kesme işareti -------------------------------------------------------

@pytest.mark.parametrize("secenek, beklenen", [
    ("ek_at", ["türkiye", "ekonomisi"]),
    ("birlestir", ["türkiyenin", "ekonomisi"]),
    ("ayir", ["türkiye", "nin", "ekonomisi"]),
])
def test_kesme_secenekleri(secenek, beklenen):
    assert kelimeler("Türkiye'nin ekonomisi", kesme_isareti=secenek) == beklenen


def test_tirnak_olarak_kesme_isareti():
    assert kelimeler("'uzaktan eğitim' kavramı") == ["uzaktan", "eğitim", "kavramı"]


def test_sayi_ve_kesme():
    assert kelimeler("2023'te büyüme", sayilari_kaldir=False) == ["2023", "büyüme"]
    assert kelimeler("2023'te büyüme") == ["büyüme"]


# --- Sayılar, URL, e-posta ---------------------------------------------------

def test_sayilar():
    assert kelimeler("412 öğrenci %3,5 oranında 1.250 kişi") == ["öğrenci", "oranında", "kişi"]
    assert kelimeler("412 öğrenci 3,5 oran", sayilari_kaldir=False) == ["412", "öğrenci", "3,5", "oran"]


def test_ondalik_virgul_bolut_bolmez():
    b = on_isle("oran 3,5 arttı", OnIslemeAyarlari(sayilari_kaldir=False))
    assert b.bolutler == [["oran", "3,5", "arttı"]]


def test_karisik_harf_rakam_korunur():
    assert kelimeler("COVID-19 salgını") == ["covıd-19", "salgını"]


def test_url_eposta_doi():
    metin = "Bkz. https://ornek.edu.tr/makale ve www.site.com adresi; iletişim: ad.soyad@uni.edu.tr doi:10.1234/abc.5"
    sonuc = kelimeler(metin)
    assert "https" not in sonuc and "www" not in sonuc and "edu" not in sonuc and "uni" not in sonuc
    assert "abc" not in sonuc
    assert "adresi" in sonuc and "iletişim" in sonuc
    assert "bkz" not in sonuc  # 1.1: kısaltma durak sözcük listesinde


def test_url_kaldirma_kapali():
    assert "ornek" in kelimeler("https://ornek.com", url_eposta_kaldir=False)


# --- Tire, şapka, uzunluk ----------------------------------------------------

def test_tireli_sozcuk():
    assert kelimeler("sosyo-ekonomik düzey") == ["sosyo-ekonomik", "düzey"]
    assert kelimeler("sosyo-ekonomik düzey", tireli_birlesik_koru=False) == ["sosyo", "ekonomik", "düzey"]


def test_sapkali_harf():
    assert kelimeler("kâğıt Kâğıt") == ["kâğıt", "kâğıt"]
    assert kelimeler("kâğıt rüzgâr", sapkali_harf_normallestir=True) == ["kağıt", "rüzgar"]


def test_sapkali_durak_sozcuk_normallestirilince_de_cikar():
    # "hâlâ" listede; sadeleştirilince "hala" da çıkarılmalı
    assert kelimeler("hâlâ devam", sapkali_harf_normallestir=True) == ["devam"]


def test_min_uzunluk():
    assert kelimeler("ab abc abcd", min_uzunluk=4, durak_sozcuk_genel=False) == ["abcd"]


# --- Durak sözcükler -------------------------------------------------------

def test_durak_sozcukler_cikarilir():
    assert kelimeler("Bu çalışma ve araştırma için yapıldı") == ["çalışma", "araştırma", "yapıldı"]


def test_akademik_durak_sozcukler():
    sonuc = kelimeler("Bu çalışma ve araştırma eğitim için", durak_sozcuk_akademik=True)
    assert sonuc == ["eğitim"]


def test_durak_sozcuk_listesi_turkce_kucuk_harf():
    for s in GENEL_DURAK_SOZCUKLER:
        assert s == turkce_kucuk_harf(s), s
        assert "I" not in s


def test_ozel_durak_sozcukler():
    ozel = ozel_durak_sozcukleri_ayristir("Işık, İLGİ;  bulunmuştur\nkâğıt")
    assert ozel == tuple(sorted({"ışık", "ilgi", "bulunmuştur", "kâğıt"}))
    assert kelimeler("Işık ilgi bulunmuştur yeni", ozel_durak_sozcukler=ozel) == ["yeni"]


def test_ozel_durak_bos():
    assert ozel_durak_sozcukleri_ayristir("") == ()
    assert ozel_durak_sozcukleri_ayristir("  ,, ") == ()


def test_stopword_set_birlesim():
    assert "ve" in stopword_set()
    assert "ve" not in stopword_set(genel=False)
    assert "çalışma" in stopword_set(akademik=True)
    assert "xyz" in stopword_set(ozel=["xyz"])


# --- Bölütler ve n-gram güvenliği -----------------------------------------

def test_bolutler_noktalama_ile_ayrilir():
    b = on_isle("Eğitim politikası. Öğretim yöntemi, ölçme aracı", DURAKSIZ)
    assert b.bolutler == [["eğitim", "politikası"], ["öğretim", "yöntemi"], ["ölçme", "aracı"]]


def test_cikarilan_sozcuk_bosluk_birakir():
    b = on_isle("eğitim ve öğretim")
    assert b.bolutler == [["eğitim", None, "öğretim"]]
    assert b.sozcukler == ["eğitim", "öğretim"]
    assert b.ham_sozcuk_sayisi == 3


# --- Uç durumlar ------------------------------------------------------------

@pytest.mark.parametrize("girdi", [None, "", "   ", float("nan"), "...!!!", "ve ile bu"])
def test_bos_ve_gecersiz_girdiler(girdi):
    assert on_isle(girdi).sozcukler == []


def test_sayi_girdisi_metne_cevrilir():
    assert on_isle(12345, OnIslemeAyarlari(sayilari_kaldir=False)).sozcukler == ["12345"]


def test_gecersiz_ayarlar():
    with pytest.raises(ValueError):
        OnIslemeAyarlari(kesme_isareti="yok")
    with pytest.raises(ValueError):
        OnIslemeAyarlari(min_uzunluk=0)


def test_kisaltmalari_koru():
    assert kelimeler("COVID-19 ve TÜİK verisi", kisaltmalari_koru=True) == ["COVID-19", "TÜİK", "verisi"]
    assert kelimeler("COVID-19 ve TÜİK verisi") == ["covıd-19", "tüik", "verisi"]


def test_derlem_on_isle_sirayi_korur():
    sonuc = derlem_on_isle(["birinci metin", "ikinci metin", ""])
    assert [b.sozcukler for b in sonuc] == [["birinci", "metin"], ["ikinci", "metin"], []]


def test_ayarlar_sozluk():
    d = OnIslemeAyarlari(ozel_durak_sozcukler=("a", "b")).sozluk()
    assert d["ozel_durak_sozcukler"] == "a, b"
    assert d["kesme_isareti"] == "ek_at"


def test_ayarlar_hashlenebilir():
    # Streamlit önbelleği için ayarlar değişmez (frozen) olmalı
    assert hash(OnIslemeAyarlari()) == hash(OnIslemeAyarlari())
