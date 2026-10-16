"""2.1 — Terim sözlüğünün arayüz testleri (AppTest; örnek veriyle)."""

import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tmlab import terimler as tm
from tmlab.analysis import derlem_ozeti
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


def ornek_metinler():
    return pd.read_csv(os.path.join(KOK, "ornek_veri", "ornek_akademik_ozetler.csv"))["ozet"].tolist()


def sozluk_ac(at, **kw):
    at.checkbox(key="terim_acik").check().run()
    for anahtar, deger in kw.items():
        at.radio(key=anahtar).set_value(deger).run() if anahtar == "terim_temsil" else None
    return at


def test_varsayilan_kapali_ve_2_0_ile_ayni():
    at = calistir()
    hatasiz(at)
    assert at.checkbox(key="terim_acik").value is False
    assert any("Sözlük **kapalı**" in i.value for i in at.info)
    beklenen = derlem_ozeti(derlem_on_isle(ornek_metinler(), OnIslemeAyarlari()))
    assert metrikler(at)["Analiz edilen sözcük"] == f"{beklenen['Analiz edilen sözcük (ön işleme sonrası)']:,}".replace(",", ".")


def test_sozluk_acilinca_terimler_ve_analizler():
    at = sozluk_ac(calistir())
    hatasiz(at)
    E = tm.TerimEslestirici(tm.cekirdek_sozluk())
    b = derlem_on_isle(ornek_metinler(), OnIslemeAyarlari(), terimci=E)
    m = metrikler(at)
    assert m["Bulunan terim geçişi"] == str(sum(len(x.terimler) for x in b))
    assert m["Farklı terim"] == str(len(tm.terim_ozeti(b)))
    beklenen = derlem_ozeti(b)
    assert m["Analiz edilen sözcük"] == f"{beklenen['Analiz edilen sözcük (ön işleme sonrası)']:,}".replace(",", ".")
    # Sözcük sıklığı tablosunda terim birimleri var
    tablolar = [d.value for d in at.dataframe]
    assert any("Sözcük" in t.columns and t["Sözcük"].astype(str).str.contains("_").any() for t in tablolar)
    # Karşılaştırma küçük derlemde kendiliğinden hesaplanır
    assert any("Sözlük kapalı" in t.columns and "Sözlük açık" in t.columns for t in tablolar)


def test_isaretle_temsili_analizleri_degistirmez():
    kapali = metrikler(calistir())
    at = calistir()
    at.checkbox(key="terim_acik").check().run()
    at.radio(key="terim_temsil").set_value("isaretle").run()
    hatasiz(at)
    acik = metrikler(at)
    for k in ("Belge", "Analiz edilen sözcük", "Farklı sözcük"):
        assert acik[k] == kapali[k]
    assert int(acik["Bulunan terim geçişi"]) > 0


def test_alan_secimi():
    at = sozluk_ac(calistir())
    tum = int(metrikler(at)["Bulunan terim geçişi"])
    at.multiselect(key="terim_alanlar").set_value(["Sağlık bilimleri"]).run()
    hatasiz(at)
    saglik = int(metrikler(at)["Bulunan terim geçişi"])
    E = tm.TerimEslestirici(tm.sozluk_olustur(["Sağlık bilimleri"]))
    assert saglik == sum(len(x.terimler) for x in derlem_on_isle(ornek_metinler(), terimci=E)) and 0 < saglik < tum
    at.multiselect(key="terim_alanlar").set_value([]).run()
    hatasiz(at)
    assert any("Etkin sözlük boş" in w.value for w in at.warning)


def test_kullanici_terimi_ekleme_silme_ve_oturum_yalitimi():
    at = sozluk_ac(calistir())
    once = int(metrikler(at)["Bulunan terim geçişi"])
    at.text_input(key="terim_yeni").set_value("çevrim içi anket")
    at.button(key="terim_ekle").click().run()
    hatasiz(at)
    assert any("eklendi" in s.value for s in at.success)
    assert int(metrikler(at)["Bulunan terim geçişi"]) == once + 2
    assert at.text_input(key="terim_yeni").value == ""
    # Aynı terim ikinci kez eklenemez
    at.text_input(key="terim_yeni").set_value("Çevrim İçi Anket")
    at.button(key="terim_ekle").click().run()
    assert any("zaten" in e.value for e in at.error)
    # Başka bir oturum bu terimi görmez
    diger = sozluk_ac(calistir())
    assert int(metrikler(diger)["Bulunan terim geçişi"]) == once
    assert not diger.session_state["_tml_kullanici_terimleri"] if "_tml_kullanici_terimleri" in diger.session_state \
        else True
    # Silme
    at.multiselect(key="terim_sil_sec").set_value(["K0001"])
    at.button(key="terim_sil").click().run()
    assert int(metrikler(at)["Bulunan terim geçişi"]) == once


def test_gecersiz_terim_hata_verir():
    at = sozluk_ac(calistir())
    at.text_input(key="terim_yeni").set_value("12345")
    at.button(key="terim_ekle").click().run()
    assert any("harf içermiyor" in e.value for e in at.error)
    assert not at.exception


def test_terimi_devre_disi_birakma_ve_geri_alma():
    at = sozluk_ac(calistir())
    once = int(metrikler(at)["Bulunan terim geçişi"])
    at.multiselect(key="terim_kapat_sec").set_value(["T0001"])  # uzaktan eğitim (örnek veride 3 kez)
    at.button(key="terim_kapat").click().run()
    hatasiz(at)
    assert int(metrikler(at)["Bulunan terim geçişi"]) == once - 3
    at.multiselect(key="terim_ac_sec").set_value(["T0001"])
    at.button(key="terim_ac").click().run()
    assert int(metrikler(at)["Bulunan terim geçişi"]) == once


def test_konu_karsilastirmasi_ve_lemma():
    at = sozluk_ac(calistir())
    at.checkbox(key="lemma").check().run()
    hatasiz(at)
    at.button(key="terim_konu_calistir").click().run()
    hatasiz(at)
    assert any("Sözlük açık" in md.value for md in at.markdown)


def test_verilerimi_temizle_kullanici_terimlerini_siler():
    at = sozluk_ac(calistir())
    at.text_input(key="terim_yeni").set_value("çevrim içi anket")
    at.button(key="terim_ekle").click().run()
    at.multiselect(key="terim_kapat_sec").set_value(["T0001"])
    at.button(key="terim_kapat").click().run()
    assert at.session_state["_tml_kullanici_terimleri"]
    at.radio(key="kaynak").set_value("Kendi dosyamı yükle").run()
    at.button(key="temizle").click().run()
    assert "_tml_kullanici_terimleri" not in at.session_state
    assert "_tml_kapali_terimler" not in at.session_state


def test_kisaltma_secenegi():
    at = sozluk_ac(calistir())
    at.checkbox(key="terim_kisaltma").check().run()
    hatasiz(at)
