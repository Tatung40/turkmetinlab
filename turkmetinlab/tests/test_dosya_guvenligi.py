"""Dosya yükleme güvenliği: içerik türü, sıkıştırma bombası, formüller, kapanmamış tırnak, başlık satırı."""

import io
import zipfile

import pandas as pd
import pytest

from tmlab.kalite import belgeleri_hazirla
from tmlab.loader import (VeriOkumaHatasi, csv_oku, dosya_oku, excel_dosyasi_denetle, excel_oku,
                          excel_sayfalari, metin_dosyasi_denetle, txt_oku)


def _excel(sayfalar, **kw):
    tampon = io.BytesIO()
    with pd.ExcelWriter(tampon, engine="openpyxl") as y:
        for ad, df in sayfalar.items():
            df.to_excel(y, sheet_name=ad, index=False, **kw)
    return tampon.getvalue()


# --- İçerik türü -----------------------------------------------------------

@pytest.mark.parametrize("bayt, ipucu", [
    (b"%PDF-1.7\n...", "PDF"),
    (b"PK\x03\x04" + b"\x00" * 40, "sıkıştırılmış"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 40, ".xls"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "PNG"),
])
def test_uzantisi_degistirilmis_dosyalar_reddedilir(bayt, ipucu):
    with pytest.raises(VeriOkumaHatasi, match=ipucu):
        dosya_oku("veri.csv", bayt)


def test_ikili_veri_reddedilir():
    with pytest.raises(VeriOkumaHatasi, match="ikili"):
        metin_dosyasi_denetle(b"metin\x00\x00\x01veri")
    with pytest.raises(VeriOkumaHatasi, match="denetim karakteri"):
        metin_dosyasi_denetle(bytes(range(1, 9)) * 200)


def test_utf16_metin_kabul_edilir():
    s = txt_oku([("a.txt", "Işık ve ağaç".encode("utf-16"))])
    assert s.tablo["metin"].tolist() == ["Işık ve ağaç"]


def test_txt_ikili_dosya_atlanir_digerleri_okunur():
    s = txt_oku([("resim.txt", b"\x89PNG\r\n\x1a\n\x00\x00"), ("metin.txt", "Ağaç".encode())])
    assert s.tablo["metin"].tolist() == ["Ağaç"]
    assert any("resim.txt" in u and "PNG" in u for u in s.uyarilar)


def test_word_belgesi_excel_diye_yuklenirse():
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("[Content_Types].xml", "<Types/>")
    with pytest.raises(VeriOkumaHatasi, match="Word"):
        excel_sayfalari(tampon.getvalue())


def test_eski_xls_ve_metin_excel_diye_yuklenirse():
    with pytest.raises(VeriOkumaHatasi, match=".xls"):
        excel_dosyasi_denetle(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)
    with pytest.raises(VeriOkumaHatasi, match="Excel çalışma kitabı değil"):
        dosya_oku("veri.xlsx", "a,b\n1,2".encode())


def test_sikistirma_bombasi_reddedilir():
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
        with z.open("xl/worksheets/sheet1.xml", "w") as g:  # 210 MB, ~0,2 MB'a sıkışır (parça parça yazılır)
            parca = b"0" * 2**20
            for _ in range(210):
                g.write(parca)
    veri = tampon.getvalue()
    assert len(veri) < 2 * 2**20
    with pytest.raises(VeriOkumaHatasi, match="çok büyük"):
        excel_dosyasi_denetle(veri)


def test_bozuk_zip():
    with pytest.raises(VeriOkumaHatasi, match="bozuk"):
        excel_dosyasi_denetle(b"PK\x03\x04bozuk-veri")


# --- Excel: formüller ve yer tutucular ---------------------------------------

def test_excel_formulleri_calistirilmaz_ve_bildirilir():
    from openpyxl import Workbook

    kitap = Workbook()
    ws = kitap.active
    ws.append(["no", "metin"])
    ws.append([1, "Normal metin"])
    ws.append([2, '=HYPERLINK("http://kotu.example","tıkla")'])
    ws.append([3, "=1+1"])
    tampon = io.BytesIO()
    kitap.save(tampon)
    s = excel_oku(tampon.getvalue())
    assert any("2 hücre formül" in u for u in s.uyarilar)
    # openpyxl ile kaydedilen dosyada hesaplanmış değer yoktur: formül hücreleri boş okunur, çalıştırılmaz
    assert s.tablo.loc[0, "metin"] == "Normal metin"
    assert "kotu.example" not in " ".join(str(v) for v in s.tablo["metin"].tolist())


def test_excel_NA_metni_sessizce_silinmez_raporlanir():
    veri = _excel({"S": pd.DataFrame({"metin": ["NA", "null", "Gerçek metin", "Yok"]})})
    s = excel_oku(veri)
    assert s.tablo["metin"].tolist() == ["NA", "null", "Gerçek metin", "Yok"]
    metinler, _, _, rapor = belgeleri_hazirla(s.tablo, "metin")
    assert metinler == ("Gerçek metin", "Yok")
    assert rapor.yer_tutucu_satirlar == [1, 2]


# --- Başlık satırı -------------------------------------------------------

def test_baslik_satiri_yoksa_ilk_satir_veri_sayilir():
    veri = "Birinci özet metni;1\nİkinci özet metni;2\n".encode("utf-8")
    s = csv_oku(veri, baslik_satiri=False)
    assert list(s.tablo.columns) == ["Sütun 1", "Sütun 2"]
    assert s.tablo["Sütun 1"].tolist() == ["Birinci özet metni", "İkinci özet metni"]
    assert len(csv_oku(veri).tablo) == 1  # başlık varsayılınca ilk satır kaybolur


def test_excel_baslik_satiri_yok():
    veri = _excel({"S": pd.DataFrame([["ilk metin", 1], ["ikinci metin", 2]])}, header=False)
    s = excel_oku(veri, baslik_satiri=False)
    assert s.tablo["Sütun 1"].tolist() == ["ilk metin", "ikinci metin"]


# --- Kapanmamış tırnak (satır kayması) ---------------------------------------

def test_kapanmamis_tirnak_uyarisi():
    satirlar = ["id;metin", '1;"Kapanmayan tırnak ile başlayan metin'] + [f"{i};satır {i} metni" for i in range(2, 30)]
    s = csv_oku("\n".join(satirlar).encode("utf-8"))
    assert len(s.tablo) == 1  # 28 satır tek hücreye yutuldu
    assert any("kapanmamış bir tırnak" in u and "Satır 2" in u for u in s.uyarilar)


def test_cok_satirli_normal_hucre_uyari_vermez():
    metin = "\\n".join(f"paragraf {i}" for i in range(15)).replace("\\n", "\n")
    s = csv_oku(('id,metin\n1,"' + metin + '"\n').encode("utf-8"))
    assert not any("tırnak" in u for u in s.uyarilar)


# --- Formül enjeksiyonu: içe aktarılan metin dışa aktarımda zararsız -------

def test_formul_metni_disa_aktarimda_calismaz():
    from tmlab.export import csv_bayt, excel_bayt

    t = pd.DataFrame({"Belge": ["=cmd|' /C calc'!A0", "@SUM(1+1)"], "Sıklık": [1, -2]})
    c = csv_bayt(t).decode("utf-8-sig")
    assert "'=cmd" in c and "'@SUM" in c and ",-2" in c  # sayılar değiştirilmez
    assert "'=cmd" not in csv_bayt(t, formul_korumasi=False).decode("utf-8-sig")
    with zipfile.ZipFile(io.BytesIO(excel_bayt({"S": t}))) as z:
        xml = z.read("xl/worksheets/sheet1.xml").decode()
    assert "<f>" not in xml
