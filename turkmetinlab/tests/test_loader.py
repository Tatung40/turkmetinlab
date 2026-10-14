"""Dosya okuma testleri."""

import io
import os

import pandas as pd
import pytest

from tmlab.loader import (VeriOkumaHatasi, baytlari_coz, csv_oku, dosya_oku, excel_oku, excel_sayfalari,
                          metin_sutunu_oner, mojibake_var_mi, txt_belgelere_ayir, txt_oku)

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TURKCE = "Işık, İstanbul'da çağdaş öğrenciler ğ ş ü"


# --- Kodlama ----------------------------------------------------------------

@pytest.mark.parametrize("kodlama", ["utf-8", "utf-8-sig", "cp1254", "iso-8859-9", "utf-16"])
def test_kodlamalar_turkce_karakterleri_korur(kodlama):
    metin, _ = baytlari_coz(TURKCE.encode(kodlama))
    assert metin.lstrip("﻿") == TURKCE


def test_utf8_oncelikli():
    _, kod = baytlari_coz("çğ".encode("utf-8"))
    assert kod.startswith("utf-8")


def test_bos_bayt_hata():
    with pytest.raises(VeriOkumaHatasi):
        baytlari_coz(b"")


def test_mojibake_tespiti():
    bozuk = "öğrenci".encode("utf-8").decode("latin-1")
    assert mojibake_var_mi(bozuk)
    assert not mojibake_var_mi("öğrenci ışık çağ")


# --- CSV --------------------------------------------------------------------

def test_csv_virgul():
    veri = 'id,metin\n1,"Eğitim, öğretim ve ölçme"\n2,"İkinci satır"\n'.encode("utf-8")
    s = csv_oku(veri)
    assert list(s.tablo.columns) == ["id", "metin"]
    assert s.tablo.loc[0, "metin"] == "Eğitim, öğretim ve ölçme"
    assert s.ayirici == ","


def test_csv_noktali_virgul_cp1254():
    veri = "kimlik;özet\n1;Işık ve ağaç\n2;Çiçek ve böcek\n".encode("cp1254")
    s = csv_oku(veri)
    assert s.ayirici == ";"
    assert s.kodlama == "cp1254"
    assert list(s.tablo.columns) == ["kimlik", "özet"]
    assert s.tablo["özet"].tolist() == ["Işık ve ağaç", "Çiçek ve böcek"]


def test_csv_sekme():
    s = csv_oku("a\tb\nx\tuzun metin\n".encode("utf-8"))
    assert s.ayirici == "\t"
    assert s.tablo.loc[0, "b"] == "uzun metin"


def test_csv_cok_satirli_hucre():
    veri = 'id,metin\n1,"birinci satır\nikinci satır"\n'.encode("utf-8")
    s = csv_oku(veri)
    assert s.tablo.loc[0, "metin"] == "birinci satır\nikinci satır"


def test_csv_sayilar_metin_olarak_okunur():
    s = csv_oku("id,metin\n007,a\n".encode("utf-8"))
    assert s.tablo.loc[0, "id"] == "007"


def test_csv_hatali_satir_uyarisi():
    veri = "a,b\n1,2\n3,4,5,6\n7,8\n".encode("utf-8")
    s = csv_oku(veri)
    # Veriler kaymamalı: hatalı satır atlanır, diğerleri olduğu gibi kalır
    assert s.tablo.values.tolist() == [["1", "2"], ["7", "8"]]
    assert any("1 satır" in u and "3" in u for u in s.uyarilar)


def test_csv_eksik_alanli_satir_doldurulur():
    s = csv_oku("a,b,c\n1,metin bir\n2,metin iki,x\n".encode("utf-8"))
    assert s.tablo["b"].tolist() == ["metin bir", "metin iki"]
    assert s.tablo["c"].isna().tolist() == [True, False]
    assert not s.uyarilar


def test_csv_bos_satir_ve_sutunlar_temizlenir():
    s = csv_oku("a,b,\n1,x,\n,,\n2,y,\n".encode("utf-8"))
    assert len(s.tablo) == 2
    assert list(s.tablo.columns) == ["a", "b"]


def test_csv_mojibake_uyarisi():
    bozuk = "id,metin\n1,öğrenci çalışması\n".encode("utf-8").decode("latin-1").encode("utf-8")
    s = csv_oku(bozuk)
    assert s.uyarilar


# --- Excel ------------------------------------------------------------------

def _excel(sayfalar):
    tampon = io.BytesIO()
    with pd.ExcelWriter(tampon, engine="openpyxl") as y:
        for ad, df in sayfalar.items():
            df.to_excel(y, sheet_name=ad, index=False)
    return tampon.getvalue()


def test_excel_okuma_ve_sayfalar():
    veri = _excel({"Özetler": pd.DataFrame({"no": [1, 2], "metin": ["Işık ve ağaç", "İğne"]}),
                   "Diğer": pd.DataFrame({"x": ["y"]})})
    assert excel_sayfalari(veri) == ["Özetler", "Diğer"]
    s = excel_oku(veri, "Özetler")
    assert s.tablo["metin"].tolist() == ["Işık ve ağaç", "İğne"]
    s2 = excel_oku(veri, "Diğer")
    assert s2.tablo["x"].tolist() == ["y"]


def test_excel_bozuk_dosya():
    with pytest.raises(VeriOkumaHatasi):
        excel_sayfalari(b"bu bir excel degil")
    with pytest.raises(VeriOkumaHatasi):
        excel_oku(b"bu bir excel degil")


def test_excel_bos_sayfa():
    with pytest.raises(VeriOkumaHatasi):
        excel_oku(_excel({"S": pd.DataFrame()}))


# --- TXT --------------------------------------------------------------------

def test_txt_modlari():
    metin = "Birinci paragraf.\nAynı paragraf.\n\nİkinci paragraf.\r\n\r\n\n Üçüncü."
    assert len(txt_belgelere_ayir(metin, "dosya")) == 1
    assert txt_belgelere_ayir(metin, "paragraf") == ["Birinci paragraf.\nAynı paragraf.", "İkinci paragraf.",
                                                    "Üçüncü."]
    assert len(txt_belgelere_ayir(metin, "satir")) == 4
    with pytest.raises(ValueError):
        txt_belgelere_ayir(metin, "yok")


def test_txt_coklu_dosya():
    s = txt_oku([("a.txt", "Işık".encode("utf-8")), ("b.txt", "Ağaç".encode("cp1254")),
                 ("bos.txt", b"")], "dosya")
    assert s.tablo["belge"].tolist() == ["a.txt", "b.txt"]
    assert s.tablo["metin"].tolist() == ["Işık", "Ağaç"]
    assert any("bos.txt" in u for u in s.uyarilar)


def test_txt_paragraf_etiketleri():
    s = txt_oku([("m.txt", "Bir.\n\nİki.".encode("utf-8"))], "paragraf")
    assert s.tablo["belge"].tolist() == ["m.txt — paragraf 1", "m.txt — paragraf 2"]


def test_txt_tamamen_bos():
    with pytest.raises(VeriOkumaHatasi):
        txt_oku([("bos.txt", b"   \n\n ")], "dosya")


# --- Yardımcılar ------------------------------------------------------------

def test_metin_sutunu_oner():
    df = pd.DataFrame({"id": ["1", "2"], "yil": ["2020", "2021"], "baslik": ["Kısa", "Kısa"],
                       "ozet": ["Çok daha uzun bir özet metni", "Bir başka uzun özet metni"]})
    assert metin_sutunu_oner(df) == "ozet"


def test_sayisal_uzun_sutun_onerilmez():
    df = pd.DataFrame({"buyuk_sayi": ["123456789012345", "123456789012345"], "m": ["kısa a", "kısa b"]})
    assert metin_sutunu_oner(df) == "m"


def test_yinelenen_sutun_adlari():
    s = csv_oku("metin,metin\na,b\n".encode("utf-8"))
    assert len(set(s.tablo.columns)) == 2


def test_dosya_oku_uzantilar():
    assert len(dosya_oku("x.CSV", "a,b\n1,2\n".encode()).tablo) == 1
    assert len(dosya_oku("x.txt", "metin".encode()).tablo) == 1
    with pytest.raises(VeriOkumaHatasi, match="xls"):
        dosya_oku("eski.xls", b"x")
    with pytest.raises(VeriOkumaHatasi, match="Desteklenmeyen"):
        dosya_oku("belge.pdf", b"x")


def test_ornek_veri_seti():
    with open(os.path.join(KOK, "ornek_veri", "ornek_akademik_ozetler.csv"), "rb") as f:
        s = dosya_oku("ornek.csv", f.read())
    assert len(s.tablo) == 16
    assert list(s.tablo.columns) == ["kimlik", "baslik", "alan", "yil", "ozet"]
    assert not s.uyarilar
    assert metin_sutunu_oner(s.tablo) == "ozet"
    assert s.tablo["kimlik"].is_unique


def test_ornek_txt():
    with open(os.path.join(KOK, "ornek_veri", "ornek_metin.txt"), "rb") as f:
        s = txt_oku([("ornek_metin.txt", f.read())], "paragraf")
    assert len(s.tablo) == 4
