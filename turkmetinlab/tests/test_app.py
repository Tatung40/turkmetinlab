"""Streamlit uygulamasının uçtan uca testleri (AppTest ile, tarayıcı olmadan)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

from tmlab.analysis import derlem_ozeti
from tmlab.loader import dosya_oku
from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(KOK, "app.py")


def calistir():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def hatasiz(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def metrikler(at):
    return {m.label: m.value for m in at.metric}


def beklenen_ozet(**ayar):
    with open(os.path.join(KOK, "ornek_veri", "ornek_akademik_ozetler.csv"), "rb") as f:
        tablo = dosya_oku("o.csv", f.read()).tablo
    return derlem_ozeti(derlem_on_isle(tablo["ozet"].tolist(), OnIslemeAyarlari(**ayar)))


@pytest.fixture(scope="module")
def at():
    return calistir()


def test_ornek_veriyle_acilir(at):
    hatasiz(at)
    assert at.title[0].value.startswith("📚 TürkMetinLab")
    assert len(at.tabs) >= 8


def test_metrikler_gercek_hesapla_eslesir(at):
    m = metrikler(at)
    o = beklenen_ozet()
    assert m["Belge"] == str(o["Belge sayısı"])
    assert m["Analiz edilen sözcük"] == f"{o['Analiz edilen sözcük (ön işleme sonrası)']:,}".replace(",", ".")
    assert m["Farklı sözcük"] == f"{o['Farklı sözcük sayısı']:,}".replace(",", ".")


def test_ornek_veri_kurgusal_uyarisi(at):
    assert any("kurgusal" in i.value for i in at.info)


def test_durak_sozcuk_kapatinca_sozcuk_sayisi_artar():
    at = calistir()
    once = metrikler(at)["Analiz edilen sözcük"]
    at.checkbox(key="durak_genel").uncheck().run()
    hatasiz(at)
    sonra = metrikler(at)["Analiz edilen sözcük"]
    o = beklenen_ozet(durak_sozcuk_genel=False)
    assert sonra == f"{o['Analiz edilen sözcük (ön işleme sonrası)']:,}".replace(",", ".")
    assert int(sonra.replace(".", "")) > int(once.replace(".", ""))


def test_akademik_ve_ozel_durak_sozcukler():
    at = calistir()
    at.checkbox(key="durak_akademik").check().run()
    at.text_area(key="ozel_durak").input("önerilmektedir, belirlenmiştir").run()
    hatasiz(at)
    o = beklenen_ozet(durak_sozcuk_akademik=True,
                      ozel_durak_sozcukler=("belirlenmiştir", "önerilmektedir"))
    assert metrikler(at)["Analiz edilen sözcük"] == f"{o['Analiz edilen sözcük (ön işleme sonrası)']:,}".replace(",", ".")


def test_ayar_kombinasyonlari_hatasiz():
    at = calistir()
    at.selectbox(key="kesme").set_value("ayir").run()
    at.selectbox(key="n").set_value(3).run()
    at.number_input(key="min_siklik").set_value(1).run()
    at.selectbox(key="tf_ngram").set_value(2).run()
    at.checkbox(key="tf_sublinear").check().run()
    at.checkbox(key="sapka").check().run()
    at.checkbox(key="kisaltma").check().run()
    at.checkbox(key="sayilar").uncheck().run()
    at.selectbox(key="bulut_kaynak").set_value("TF-IDF (toplam)").run()
    at.radio(key="csv_ayirici").set_value(",").run()
    hatasiz(at)


def test_tum_sozcukler_cikarilinca_anlasilir_uyari():
    at = calistir()
    at.number_input(key="min_uzunluk").set_value(10).run()
    at.number_input(key="tf_min_belge").set_value(50).run()
    assert not at.exception, [e.value for e in at.exception]


def test_ad_sutunu_degistirme():
    at = calistir()
    at.selectbox(key="ad_sutunu").set_value("kimlik").run()
    hatasiz(at)
    assert "OZ01" in at.selectbox(key="tfidf_belge").options


def test_yukleme_moduna_gecis():
    at = calistir()
    at.radio(key="kaynak").set_value("Kendi dosyamı yükle").run()
    hatasiz(at)
    assert any("dosya yükleyin" in i.value for i in at.info)


def test_yukleme_siniri_config_dosyasindan_bagimsiz():
    """README'deki 30 MB sınırı config.toml'a değil koda bağlı olmalı (depoda .streamlit yok)."""
    import inspect
    from pathlib import Path
    import streamlit as st
    assert "max_upload_size" in inspect.signature(st.file_uploader).parameters
    kaynak = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "YUKLEME_SINIRI_MB = 30" in kaynak
    assert kaynak.count("max_upload_size=YUKLEME_SINIRI_MB") == kaynak.count("st.file_uploader(")
