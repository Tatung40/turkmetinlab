"""Dışa aktarım ve görselleştirme testleri."""

import io

import pandas as pd
import pytest
from PIL import Image

from tmlab.analysis import IslenmisBelge, sozcuk_sikligi, tfidf_hesapla
from tmlab.export import csv_bayt, excel_bayt, sozluk_tablosu
from tmlab.visualize import kelime_bulutu_png, tfidf_isi_haritasi, turkce_font_yolu, yatay_cubuk

TABLO = pd.DataFrame({"Sözcük": ["ışık", "İğne", "çağ"], "Sıklık": [3, 2, 1], "Oran": [0.5, 0.333, 0.167]})


# --- CSV --------------------------------------------------------------------

def test_csv_bom_ve_turkce():
    b = csv_bayt(TABLO)
    assert b.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM (Excel için)
    geri = pd.read_csv(io.BytesIO(b), encoding="utf-8-sig")
    assert geri["Sözcük"].tolist() == ["ışık", "İğne", "çağ"]
    assert geri["Oran"].tolist() == [0.5, 0.333, 0.167]


def test_csv_noktali_virgul_ondalik_virgul():
    b = csv_bayt(TABLO, ";")
    metin = b.decode("utf-8-sig")
    assert "ışık;3;0,5" in metin
    geri = pd.read_csv(io.BytesIO(b), sep=";", decimal=",", encoding="utf-8-sig")
    assert geri["Oran"].tolist() == [0.5, 0.333, 0.167]


# --- Excel ------------------------------------------------------------------

def test_excel_coklu_sayfa_ve_geri_okuma():
    b = excel_bayt({"Sözcük sıklığı": TABLO, "Ayarlar": sozluk_tablosu({"a": 1})})
    kitap = pd.read_excel(io.BytesIO(b), sheet_name=None)
    assert list(kitap) == ["Sözcük sıklığı", "Ayarlar"]
    pd.testing.assert_frame_equal(kitap["Sözcük sıklığı"], TABLO)
    assert kitap["Ayarlar"].iloc[0].tolist() == ["a", 1]


def test_excel_sayfa_adi_temizleme():
    uzun = "Bu çok uzun bir sayfa adı: [deneme]/test?*"
    b = excel_bayt({uzun: TABLO, uzun + " ": TABLO})
    adlar = list(pd.read_excel(io.BytesIO(b), sheet_name=None))
    assert len(adlar) == 2 and len(set(adlar)) == 2
    for ad in adlar:
        assert len(ad) <= 31
        assert not any(c in ad for c in "[]:*?/\\")


def test_excel_bos_tablo():
    b = excel_bayt({"Boş": pd.DataFrame(columns=["a", "b"])})
    assert list(pd.read_excel(io.BytesIO(b)).columns) == ["a", "b"]


def test_sozluk_tablosu():
    t = sozluk_tablosu({"x": True, "y": 2}, "Ayar", "Değer")
    assert t.values.tolist() == [["x", "True"], ["y", "2"]]


# --- Grafikler --------------------------------------------------------------

def test_font_turkce_karakter_iceriyor():
    from matplotlib.font_manager import FontProperties, get_font
    font = get_font(turkce_font_yolu())
    for harf in "çğıöşüÇĞİÖŞÜâ":
        assert font.get_char_index(ord(harf)) != 0, harf


def test_yatay_cubuk_veri_dogrulugu():
    fig = yatay_cubuk(TABLO, "Sözcük", "Sıklık", "Başlık", ilk_n=2)
    iz = fig.data[0]
    # En büyük en üstte görünsün diye ters sırada çizilir
    assert list(iz.y) == ["İğne", "ışık"]
    assert list(iz.x) == [2, 3]
    assert fig.layout.title.text == "Başlık"


def test_isi_haritasi():
    d = [IslenmisBelge([["elma", "armut"]]), IslenmisBelge([["kiraz", "elma"]])]
    s = tfidf_hesapla(d, ["A", "B"])
    fig = tfidf_isi_haritasi(s, 5)
    assert fig is not None
    assert list(fig.data[0].y) == ["A", "B"]
    assert tfidf_isi_haritasi(s, 5, en_fazla_belge=1) is None


def test_kelime_bulutu_png_ve_belirlenimcilik():
    agirlik = {"ışık": 10, "İstanbul": 7, "çağdaş": 5, "öğretim": 3, "ğüş": 1}
    a = kelime_bulutu_png(agirlik, genislik=400, yukseklik=200)
    b = kelime_bulutu_png(agirlik, genislik=400, yukseklik=200)
    assert a == b  # aynı girdi -> aynı görsel
    gorsel = Image.open(io.BytesIO(a))
    assert gorsel.size == (400, 200)


def test_kelime_bulutu_bos():
    with pytest.raises(ValueError):
        kelime_bulutu_png({})
    with pytest.raises(ValueError):
        kelime_bulutu_png({"a": 0})


def test_kelime_bulutu_gercek_siklikla():
    t = sozcuk_sikligi([IslenmisBelge([["eğitim", "eğitim", "öğrenci"]])])
    png = kelime_bulutu_png(dict(zip(t["Sözcük"], t["Sıklık"])), genislik=300, yukseklik=150)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
