"""Performans ve bellek testleri, büyük veri uyarıları.

Küçük ve orta (gerçekçi dağarcıklı) derlemler her test çalıştırmasında ölçülür. Büyük derlem
(~2,9 milyon sözcük) uzun sürdüğü için yalnızca TML_BUYUK_TEST=1 ortam değişkeniyle çalışır:

    TML_BUYUK_TEST=1 python -m pytest tests/test_performans.py
"""

import json
import os
import subprocess
import sys

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import tmlab.performans as performans
from tmlab.performans import boyut_degerlendir, sozcuk_tahmini

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(KOK, "app.py")
OLCUM = os.path.join(KOK, "degerlendirme", "performans_olcumu.py")


# --- Sınır mantığı ---------------------------------------------------------

def test_sozcuk_tahmini():
    assert sozcuk_tahmini(["bir iki  üç", "dört\nbeş", ""]) == 5


def test_boyut_duzeyleri():
    assert boyut_degerlendir(10, uyari=100, en_fazla=1000).duzey == "normal"
    u = boyut_degerlendir(500, uyari=100, en_fazla=1000)
    assert u.duzey == "uyari" and "500" in u.mesaj
    e = boyut_degerlendir(5000, uyari=100, en_fazla=1000)
    assert e.duzey == "engel" and "kendi bilgisayarınızda" in e.mesaj and "5.000" in e.mesaj
    assert "Lemmatizasyon" in boyut_degerlendir(500, True, uyari=100, en_fazla=1000).mesaj


def test_varsayilan_sinirlar_olcumle_uyumlu():
    # Ölçümde 2,9 milyon sözcük ~700-850 MB tepe bellek kullandı; ücretsiz planın alt sınırı ~690 MB.
    assert performans.UYARI_SOZCUK == 500_000
    assert performans.EN_FAZLA_SOZCUK == 1_500_000


def test_ortam_degiskeni_ile_sinir(monkeypatch):
    import importlib

    monkeypatch.setenv("TML_EN_FAZLA_SOZCUK", "3000000")
    monkeypatch.setenv("TML_UYARI_SOZCUK", "gecersiz")
    m = importlib.reload(performans)
    try:
        assert m.EN_FAZLA_SOZCUK == 3_000_000 and m.UYARI_SOZCUK == 500_000
    finally:
        monkeypatch.delenv("TML_EN_FAZLA_SOZCUK")
        monkeypatch.delenv("TML_UYARI_SOZCUK")
        importlib.reload(performans)


# --- Uygulamada uyarı ve engel ---------------------------------------------

def test_uygulama_buyuk_veride_uyarir(monkeypatch):
    monkeypatch.setattr(performans, "UYARI_SOZCUK", 500)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and not at.error
    assert any("Büyük veri" in w.value for w in at.warning)
    assert at.metric  # analiz yine yapılır


def test_uygulama_sinir_ustunde_analiz_yapmaz(monkeypatch):
    monkeypatch.setattr(performans, "EN_FAZLA_SOZCUK", 500)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("bellek sınırı" in e.value for e in at.error)
    assert not at.metric  # analiz yapılmadı
    assert any("Yüklenen veri" in h.value for h in at.subheader)  # veri yine de incelenebilir


# --- Ölçümler (ayrı süreçte; tepe bellek birbirini etkilemesin) ------------

def _olc(boyut: str, lemma: bool = False) -> dict:
    cikti = subprocess.run([sys.executable, OLCUM, "--tek", boyut, "1" if lemma else "0"],
                           capture_output=True, text=True, check=True, cwd=KOK, timeout=1800).stdout
    return json.loads(cikti.strip().splitlines()[-1])


def test_kucuk_veri_hizli():
    r = _olc("kucuk")
    analiz = r["toplam_sn"] - r["süreler_sn"]["Excel (sonuç tabloları)"]
    assert analiz < 15, r
    assert r["tepe RSS MB"] < 500, r


def test_orta_gercekci_veri_sinirlar_icinde():
    """~288 bin sözcük, ~60 bin farklı sözcük (Zipf dağılımı)."""
    r = _olc("orta_genis")
    assert r["sözcük (ön işleme öncesi)"] > 250_000 and r["farklı sözcük"] > 50_000
    analiz = r["toplam_sn"] - r["süreler_sn"]["Excel (sonuç tabloları)"]
    assert analiz < 60, r  # ölçüm makinesinde ~4 sn; yavaş CI için geniş pay
    assert r["tepe RSS MB"] < 690, r  # ücretsiz planın alt bellek sınırı


def test_orta_gercekci_veri_lemmatizasyonla():
    r = _olc("orta_genis", lemma=True)
    assert r["tepe RSS MB"] < 690, r


@pytest.mark.skipif(os.environ.get("TML_BUYUK_TEST") != "1", reason="Büyük ölçüm için TML_BUYUK_TEST=1")
def test_buyuk_gercekci_veri():
    r = _olc("buyuk_genis")
    assert r["sözcük (ön işleme öncesi)"] > 2_500_000
    assert r["tepe RSS MB"] < 1200, r
