"""Gizlilik mimarisi testleri.

Doğrulananlar:
* Kullanıcı verisinden türeyen hiçbir hesaplama st.cache_data (tüm kullanıcılarca paylaşılan) önbelleğe konmaz.
* Oturum belleği oturumlar arasında paylaşılmaz, sınırlıdır ve temizlenebilir.
* Uygulama ve modüller diske dosya yazmaz; Excel üretimi geçici dosya kullanmaz.
* Beklenmeyen hatalarda günlüğe kullanıcı metni yazılmaz.
"""

import ast
import os
import tempfile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tmlab import oturum
from tmlab.export import csv_bayt, excel_bayt
from tmlab.oturum import (ONEK, OturumBellegi, guvenli_hata_kaydi, oturum_verilerini_sil, ozet,
                          yuklenen_dosyalari_sil)

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(KOK, "app.py")
GIZLI = "GİZLİ_KATILIMCI_YANITI_4721"


# --- Paylaşılan önbellek kullanımı -----------------------------------------

def _kaynak_dosyalar():
    yollar = [APP]
    for ad in os.listdir(os.path.join(KOK, "tmlab")):
        if ad.endswith(".py"):
            yollar.append(os.path.join(KOK, "tmlab", ad))
    return yollar


def test_cache_data_yalnizca_ornek_veri_icin():
    """st.cache_data / st.cache_resource yalnızca izin verilen, kullanıcı verisi almayan işlevlerde."""
    # 2.5: _kodlayici yalnızca model klasörünü alır (model ağırlıkları; kullanıcı verisi değil)
    izinli = {"_ornek_oku": ["baslik_satiri"], "_kodlayici": ["kimlik", "klasor"]}
    for yol in _kaynak_dosyalar():
        agac = ast.parse(open(yol, encoding="utf-8").read())
        for dugum in ast.walk(agac):
            if isinstance(dugum, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in dugum.decorator_list:
                    metin = ast.unparse(d)
                    if "cache_data" in metin or "cache_resource" in metin:
                        assert dugum.name in izinli, f"{yol}: {dugum.name} paylaşılan önbellek kullanıyor"
                        # İzinli işlev bayt/dosya gibi kullanıcı verisi parametresi almamalı
                        parametreler = [a.arg for a in dugum.args.args]
                        assert parametreler == izinli[dugum.name], parametreler


# 2.5 istisnası: yalnızca model dosyalarını (ağırlık, belirteçleyici, künye) önbelleğe indiren işlev diske yazar.
MODEL_INDIREN = ("anlamsal.py", "model_hazirla")


def test_diske_yazan_kod_yok():
    """Uygulama kodunda yazma kipinde open(), tempfile veya to_csv/to_excel ile dosya yolu kullanımı yok."""
    for yol in _kaynak_dosyalar():
        kaynak = open(yol, encoding="utf-8").read()
        agac = ast.parse(kaynak)
        istisna = set()
        for f in ast.walk(agac):
            if isinstance(f, ast.FunctionDef) and (os.path.basename(yol), f.name) == MODEL_INDIREN:
                istisna = {id(d) for d in ast.walk(f)}
        for dugum in ast.walk(agac):
            if id(dugum) in istisna:
                continue
            if isinstance(dugum, ast.Call) and isinstance(dugum.func, ast.Name) and dugum.func.id == "open":
                kip = dugum.args[1].value if len(dugum.args) > 1 and isinstance(dugum.args[1], ast.Constant) else "r"
                for k in dugum.keywords:
                    if k.arg == "mode" and isinstance(k.value, ast.Constant):
                        kip = k.value.value
                assert not any(c in kip for c in "wax+"), f"{yol}: yazma kipinde open()"
            if isinstance(dugum, (ast.Import, ast.ImportFrom)):
                adlar = [a.name for a in dugum.names] + ([dugum.module] if isinstance(dugum, ast.ImportFrom) else [])
                assert "tempfile" not in adlar, f"{yol}: tempfile kullanılıyor"


def test_excel_ve_csv_gecici_dosya_kullanmaz(monkeypatch):
    def yasak(*a, **k):
        raise AssertionError("Diske geçici dosya yazılmaya çalışıldı")

    for ad in ("mkstemp", "mkdtemp", "TemporaryFile", "NamedTemporaryFile", "SpooledTemporaryFile"):
        monkeypatch.setattr(tempfile, ad, yasak)
    tablo = pd.DataFrame({"Özgün metin": [GIZLI] * 50, "n": range(50)})
    assert excel_bayt({"Ön işlenmiş metinler": tablo, "Diğer": tablo})[:2] == b"PK"
    assert GIZLI.encode("utf-8") in csv_bayt(tablo)


# --- Oturum belleği ----------------------------------------------------------

def test_oturumlar_birbirinin_verisini_goremez():
    oturum_a, oturum_b = {}, {}
    sayac = {"n": 0}

    def hesapla():
        sayac["n"] += 1
        return GIZLI

    anahtar = ozet("aynı dosya", b"ayni-bayt")
    assert OturumBellegi(oturum_a).al("onisleme", anahtar, hesapla) == GIZLI
    # Aynı içerik başka oturumda yüklense bile A'nın sonucu paylaşılmaz; B kendi hesaplamasını yapar
    assert OturumBellegi(oturum_b).al("onisleme", anahtar, hesapla) == GIZLI
    assert sayac["n"] == 2
    assert OturumBellegi(oturum_a).kayit_sayisi() == 1 and OturumBellegi(oturum_b).kayit_sayisi() == 1
    assert oturum_a[ONEK + "bellek"] is not oturum_b[ONEK + "bellek"]


def test_ayni_oturumda_yeniden_hesaplanmaz_ve_sinirli():
    depo, sayac = {}, {"n": 0}
    b = OturumBellegi(depo)

    def hesapla(deger):
        def f():
            sayac["n"] += 1
            return deger
        return f

    assert b.al("x", "k1", hesapla(1), en_fazla=2) == 1
    assert b.al("x", "k1", hesapla(99), en_fazla=2) == 1
    assert sayac["n"] == 1
    b.al("x", "k2", hesapla(2), en_fazla=2)
    b.al("x", "k3", hesapla(3), en_fazla=2)  # k1 en eski: silinir
    assert b.kayit_sayisi() == 2
    assert b.al("x", "k1", hesapla(10), en_fazla=2) == 10


def test_hata_olursa_bellege_bir_sey_yazilmaz():
    depo = {}
    b = OturumBellegi(depo)
    with pytest.raises(ValueError):
        b.al("x", "k", lambda: (_ for _ in ()).throw(ValueError("hata")))
    assert b.kayit_sayisi() == 0


def test_ozet_metni_icermez_ve_belirlenimci():
    a = ozet(GIZLI, b"\x00\x01", ("x", 1), None)
    assert a == ozet(GIZLI, b"\x00\x01", ("x", 1), None)
    assert GIZLI not in a and len(a) == 64
    assert ozet("ab", "c") != ozet("a", "bc")  # parça sınırları karışmaz
    assert ozet(("a",)) != ozet("a")


def test_oturum_verilerini_sil_yalnizca_uygulama_anahtarlari():
    depo = {ONEK + "bellek": {"x": GIZLI}, ONEK + "baska": 1, "kaynak": "Örnek veri setini kullan"}
    assert oturum_verilerini_sil(depo) == 2
    assert depo == {"kaynak": "Örnek veri setini kullan"}


def test_yuklenen_dosyalari_sil(monkeypatch):
    from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
    from streamlit.runtime.uploaded_file_manager import UploadedFileRec

    yonetici = MemoryUploadedFileManager("/_stcore/upload_file")
    yonetici.add_file("oturum-1", UploadedFileRec("d1", "veri.csv", "text/csv", GIZLI.encode()))
    yonetici.add_file("oturum-2", UploadedFileRec("d2", "baska.csv", "text/csv", b"x"))

    class Baglam:
        uploaded_file_mgr = yonetici
        session_id = "oturum-1"

    import streamlit.runtime.scriptrunner as sr
    monkeypatch.setattr(sr, "get_script_run_ctx", lambda: Baglam())
    assert yuklenen_dosyalari_sil() is True
    assert yonetici.get_files("oturum-1", ["d1"]) == []
    assert len(yonetici.get_files("oturum-2", ["d2"])) == 1  # başka oturumun dosyası etkilenmez


def test_yuklenen_dosyalari_sil_baglam_yoksa():
    assert yuklenen_dosyalari_sil() is False


# --- Hata günlüğü ----------------------------------------------------------

def test_hata_kaydi_metin_icermez(capsys):
    try:
        raise ValueError(f"Ayrıştırılamayan satır: {GIZLI}")
    except ValueError as e:
        kayit = guvenli_hata_kaydi(e)
    cikti = capsys.readouterr()
    assert GIZLI not in kayit and GIZLI not in cikti.err and GIZLI not in cikti.out
    assert "ValueError" in kayit and "test_gizlilik.py" in kayit


def test_uygulama_hatasi_kullaniciya_genel_mesajla_gosterilir(monkeypatch, capsys):
    import tmlab.analysis as analiz

    def bozuk(*a, **k):
        raise RuntimeError(f"iç hata: {GIZLI}")

    monkeypatch.setattr(analiz, "sozcuk_sikligi", bozuk)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    hatalar = " ".join(e.value for e in at.error)
    assert "Beklenmeyen bir hata" in hatalar
    tum_metin = " ".join(str(getattr(el, "value", "")) for el in at.main) + hatalar
    assert GIZLI not in tum_metin
    assert GIZLI not in capsys.readouterr().err


# --- Arayüz: temizleme düğmesi ve gizlilik metni -------------------------

def test_temizle_dugmesi_oturum_verisini_siler():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="kaynak").set_value("Kendi dosyamı yükle").run()
    at.session_state[ONEK + "bellek"] = {"onisleme": {"k": GIZLI}}
    at.button(key="temizle").click().run()
    assert not at.exception
    assert ONEK + "bellek" not in at.session_state
    assert any("silindi" in s.value for s in at.success)
    assert at.session_state["yukleme_surumu"] == 1


def test_gizlilik_metni_gercek_davranisi_anlatir():
    kaynak = open(APP, encoding="utf-8").read()
    for ifade in ["oturumunuza ait bellekte", "diske, veritabanına veya GitHub'a yazmaz",
                  "Verilerimi şimdi temizle", "2 dakika", "anonimleştirmenizi"]:
        assert ifade in kaynak
    # 1.0'daki "hiçbir yere kaydedilmez" iddiası artık gerekçesiyle birlikte
    assert "hiçbir yere kaydedilmez" not in kaynak
