"""Modül 4 — Grup karşılaştırması testleri."""

import math

import numpy as np
import pandas as pd
import pytest

from tmlab.grup import (GrupAyarlari, GrupHatasi, ayirt_edici_grafik, ayirt_edici_sozcukler, grup_ozeti,
                        grup_sutunu_oner, grup_verisi_hazirla, iki_grup_sacilim, siklik_karsilastirma,
                        yontemsel_uyarilar)
from tmlab.preprocess import IslenmisBelge, OnIslemeAyarlari, derlem_on_isle


def B(*t):
    return IslenmisBelge([list(t)])


def z_tablosu(belgeler, gruplar, **kw):
    kw.setdefault("min_siklik", 1)
    kw.setdefault("permutasyon", 0)
    veri, _ = grup_verisi_hazirla(belgeler, gruplar)
    t, _ = ayirt_edici_sozcukler(veri, GrupAyarlari(**kw))
    return t.set_index(["Grup", "Sözcük"])


# --- Elle hesaplanabilir örnekler -------------------------------------------

ORNEK = ([B("x", "x", "x", "y"), B("x", "y", "y", "y")], ["A", "B"])


def test_elle_zayif_onsel():
    """A: x×3, y×1 ; B: x×1, y×3 ; α = 0,01 → α0 = 0,02.
    δ_x = ln(3,01/1,01) − ln(1,01/3,01) ; σ² = 1/3,01 + 1/1,01"""
    t = z_tablosu(*ORNEK, onsel="zayif")
    delta = math.log(3.01 / 1.01) - math.log(1.01 / 3.01)
    z = delta / math.sqrt(1 / 3.01 + 1 / 1.01)
    assert t.loc[("A", "x"), "Log-odds farkı (δ)"] == pytest.approx(delta, abs=1e-4)
    assert t.loc[("A", "x"), "z (ağırlıklı log-odds)"] == pytest.approx(z, abs=1e-4)
    assert t.loc[("B", "x"), "z (ağırlıklı log-odds)"] == pytest.approx(-z, abs=1e-4)


def test_elle_bilgilendirici_onsel():
    """α_x = α_y = 4 (derlem sıklıkları), α0 = 8: δ_x = ln(7/5) − ln(5/7); σ² = 1/7 + 1/5"""
    t = z_tablosu(*ORNEK, onsel="bilgilendirici")
    z = (math.log(7 / 5) - math.log(5 / 7)) / math.sqrt(1 / 7 + 1 / 5)
    assert t.loc[("A", "x"), "z (ağırlıklı log-odds)"] == pytest.approx(z, abs=1e-4)
    # Bilgilendirici önsel daha tutucudur (|z| daha küçük)
    assert abs(z) < abs(z_tablosu(*ORNEK, onsel="zayif").loc[("A", "x"), "z (ağırlıklı log-odds)"])


def test_bin_sozcukte_belge_orani_ve_tek_belge_payi():
    belgeler = [B(*(["k"] * 9 + ["a"])), B("k", "a", "a", "a"), B("b", "b")]
    t = z_tablosu(belgeler, ["G", "G", "H"])
    r = t.loc[("G", "k")]
    assert r["Gruptaki sıklık"] == 10 and r["Diğer gruplardaki sıklık"] == 0
    assert r["Gruptaki bin sözcükte"] == pytest.approx(10 / 14 * 1000, abs=1e-3)
    assert r["Gruptaki belge oranı (%)"] == 100.0
    assert r["Tek belgeden gelen en büyük pay (%)"] == 90.0


# --- Bağımsız uygulama (FightingWords döngüsü) ile karşılaştırma ---------

def _fightin_words(c1, c2, prior):
    """jmhessel/FightingWords bayes_compare_language çekirdeğinin bağımsız kopyası."""
    a0 = float(np.sum(prior))
    n1, n2 = float(c1.sum()), float(c2.sum())
    z = np.empty(len(c1))
    for i in range(len(c1)):
        t1 = np.log((c1[i] + prior[i]) / (n1 + a0 - c1[i] - prior[i]))
        t2 = np.log((c2[i] + prior[i]) / (n2 + a0 - c2[i] - prior[i]))
        z[i] = (t1 - t2) / np.sqrt(1.0 / (c1[i] + prior[i]) + 1.0 / (c2[i] + prior[i]))
    return z


@pytest.fixture(scope="module")
def ud_grup(ud):
    secili = [c for c in ud if c["tur"] in ("news", "bio", "ess")]
    return derlem_on_isle([c["metin"] for c in secili], OnIslemeAyarlari()), [c["tur"] for c in secili]


@pytest.mark.parametrize("onsel", ["zayif", "bilgilendirici"])
def test_fightin_words_ile_ayni(ud_grup, onsel):
    belgeler, tur = ud_grup
    iki = [(b, g) for b, g in zip(belgeler, tur) if g in ("news", "bio")]
    veri, _ = grup_verisi_hazirla([b for b, _ in iki], [g for _, g in iki])
    t, _ = ayirt_edici_sozcukler(veri, GrupAyarlari(onsel=onsel, min_siklik=1, permutasyon=0))
    Y = np.vstack([np.asarray(veri.X[veri.etiket == k].sum(axis=0)).ravel() for k in range(2)]).astype(float)
    prior = Y.sum(axis=0) if onsel == "bilgilendirici" else np.full(Y.shape[1], 0.01)
    beklenen = dict(zip(veri.sozcukler, _fightin_words(Y[0], Y[1], prior)))
    bizim = t[t["Grup"] == veri.gruplar[0]].set_index("Sözcük")["z (ağırlıklı log-odds)"]
    assert len(bizim) > 500
    for w, z in bizim.items():
        assert z == pytest.approx(beklenen[w], abs=1e-4)


def test_cok_grup_her_grup_digerleriyle(ud_grup):
    """3 grupta 'news' sonucu, bio + ess birleştirilmiş iki grup karşılaştırmasıyla aynı olmalı."""
    belgeler, tur = ud_grup
    uc = z_tablosu(belgeler, tur, onsel="zayif").loc["news"]["z (ağırlıklı log-odds)"]
    iki = z_tablosu(belgeler, [g if g == "news" else "diğer" for g in tur], onsel="zayif").loc["news"][
        "z (ağırlıklı log-odds)"]
    assert np.allclose(uc.sort_index(), iki.sort_index())


# --- Permütasyon ------------------------------------------------------------

def test_permutasyon_belirlenimci_ve_rastgele_gruplamada_seyrek_asim(ud_grup):
    belgeler, _ = ud_grup
    rng = np.random.default_rng(5)
    rastgele = list(np.where(rng.permutation(len(belgeler)) % 2 == 0, "A", "B"))
    veri, _ = grup_verisi_hazirla(belgeler, rastgele)
    a1, p1 = ayirt_edici_sozcukler(veri, GrupAyarlari(min_siklik=2, permutasyon=100))
    a2, p2 = ayirt_edici_sozcukler(veri, GrupAyarlari(min_siklik=2, permutasyon=100))
    assert np.array_equal(p1["esik"], p2["esik"]) and a1.equals(a2)
    assert (a1["Permütasyon eşiğini aşıyor"] == "evet").sum() <= 2  # anlamsız gruplamada (neredeyse) hiç
    assert 0 <= p1["yanlis_pozitif_orani"] < 0.2


def test_gercek_tur_farki_permutasyonu_asar(ud_grup):
    belgeler, tur = ud_grup
    veri, _ = grup_verisi_hazirla(belgeler, tur)
    a, p = ayirt_edici_sozcukler(veri, GrupAyarlari(min_siklik=2, permutasyon=100))
    assert (a["Permütasyon eşiğini aşıyor"] == "evet").sum() >= 1
    assert p["esik"].shape == (3,)


# --- Veri hazırlama, uyarılar ve tablolar ---------------------------------

def test_bos_grup_ve_bos_metin():
    belgeler = [B("a", "b"), B("a"), B(), B("c", "d"), B("d")]
    veri, bos = grup_verisi_hazirla(belgeler, ["X", None, "X", " ", "Y"])
    assert bos == [1, 3] and veri.belge_indisleri.tolist() == [0, 4]


def test_grup_hatalari():
    with pytest.raises(GrupHatasi, match="en az 2"):
        grup_verisi_hazirla([B("a"), B("b")], ["X", "X"])
    with pytest.raises(GrupHatasi, match="en fazla 20"):
        grup_verisi_hazirla([B("a")] * 42, [str(i % 21) for i in range(42)])
    veri, _ = grup_verisi_hazirla([B("a"), B("b"), B("c")], ["1", "2", "3"])  # izin verilir, uyarılır
    assert any("kimlik" in u for u in yontemsel_uyarilar(grup_ozeti(veri)))
    veri, _ = grup_verisi_hazirla([B("a"), B("b"), B("a")], ["X", "Y", "X"])
    with pytest.raises(GrupHatasi, match="50 kez"):
        ayirt_edici_sozcukler(veri, GrupAyarlari(min_siklik=50, permutasyon=0))


@pytest.mark.parametrize("kw", [dict(onsel="x"), dict(min_siklik=0), dict(permutasyon=-1), dict(permutasyon=5000)])
def test_gecersiz_ayarlar(kw):
    with pytest.raises(ValueError):
        GrupAyarlari(**kw)


def test_ozet_uyarilar_ve_siklik_tablosu():
    belgeler = [B(*(["a"] * 600 + ["b"] * 600))] * 6 + [B("a", "c")]
    veri, _ = grup_verisi_hazirla(belgeler, ["Büyük"] * 6 + ["Küçük"])
    o = grup_ozeti(veri)
    assert o.set_index("Grup").loc["Büyük", "Sözcük"] == 7200
    u = " ".join(yontemsel_uyarilar(o))
    assert "küçük" in u and "dengesiz" in u
    s = siklik_karsilastirma(veri)
    assert s.columns.tolist()[:2] == ["Sözcük", "Toplam"] and "Küçük · bin sözcükte" in s.columns
    assert s.set_index("Sözcük").loc["c", "Küçük · bin sözcükte"] == 500.0


def test_grup_sutunu_onerisi():
    t = pd.DataFrame({"kimlik": list("abcdefgh"), "bolum": list("XXYYXXYY"), "yil": list("12341234"),
                      "metin": ["m"] * 8})
    assert grup_sutunu_oner(t, haric=("metin",)) == "bolum"
    assert grup_sutunu_oner(t[["kimlik", "metin"]], haric=("metin",)) is None


def test_grafikler(ud_grup):
    belgeler, tur = ud_grup
    iki = [(b, g) for b, g in zip(belgeler, tur) if g in ("news", "bio")]
    veri, _ = grup_verisi_hazirla([b for b, _ in iki], [g for _, g in iki])
    t, p = ayirt_edici_sozcukler(veri, GrupAyarlari(min_siklik=2, permutasyon=20))
    assert len(ayirt_edici_grafik(t, veri.gruplar, 10, p["esik"]).data) == 2
    assert len(iki_grup_sacilim(t, veri.gruplar).data) == 1


# --- Uygulama içinde -------------------------------------------------------

def test_uygulamada_grup_karsilastirmasi():
    import os

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"), default_timeout=300).run()
    assert not at.exception and not at.error
    assert at.selectbox(key="grup_sutunu").value == "alan"  # kimlik değil, kategorik sütun önerilir
    assert any("küçük" in w.value for w in at.warning)  # örnek veride gruplar 4'er belge
    at.selectbox(key="grup_onsel").set_value("zayif").run()
    at.selectbox(key="grup_secili").set_value("Sağlık").run()
    assert not at.exception and not at.error
    at.selectbox(key="grup_sutunu").set_value("kimlik").run()
    assert not at.exception and not at.error
    assert any("kimlik" in w.value for w in at.warning)
