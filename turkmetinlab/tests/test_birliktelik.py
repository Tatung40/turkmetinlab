"""Modül 2 — Sözcük birliktelik ağları testleri."""

import itertools
import math

import networkx as nx
import numpy as np
import pytest

from tmlab.birliktelik import (KENAR_DISA_AKTARIM_SINIRI, AgAyarlari, ag_figuru, ag_sec, birliktelik_hesapla,
                               dugum_tablosu, kenar_tablosu)
from tmlab.preprocess import IslenmisBelge, OnIslemeAyarlari, derlem_on_isle


def B(*bolutler):
    return IslenmisBelge([list(x) for x in bolutler])


def C(belgeler, **kw):
    kw.setdefault("aday_sozcuk", 50)
    kw.setdefault("min_birlikte", 1)
    r = birliktelik_hesapla(belgeler, AgAyarlari(**kw))
    ix = {w: i for i, w in enumerate(r.sozcukler)}
    return r, (lambda a, b: int(r.C[ix[a], ix[b]]))


# --- Elle hesaplanabilir örnekler -------------------------------------------

def test_pencere_genisligi():
    _, c = C([B(["a", "b", "c"])], pencere=1)
    assert c("a", "b") == 1 and c("b", "c") == 1 and c("a", "c") == 0
    _, c = C([B(["a", "b", "c"])], pencere=2)
    assert c("a", "c") == 1


def test_cikarilan_sozcuk_yer_kaplar():
    _, c = C([B(["a", None, "c"])], pencere=1)
    assert c("a", "c") == 0
    _, c = C([B(["a", None, "c"])], pencere=2)
    assert c("a", "c") == 1


def test_cumle_ve_belge_sinirlari():
    _, c = C([B(["a"], ["b"])], pencere=5, bolut_siniri=True)
    assert c("a", "b") == 0
    _, c = C([B(["a"], ["b"])], pencere=5, bolut_siniri=False)
    assert c("a", "b") == 1
    _, c = C([B(["a"]), B(["b"])], pencere=5, bolut_siniri=False)  # belgeler arası asla
    assert c("a", "b") == 0


def test_ayni_sozcuk_sayilmaz_ve_simetri():
    r, c = C([B(["a", "a", "b"])], pencere=1)
    assert c("a", "b") == 1 and c("a", "a") == 0
    assert np.array_equal(r.C, r.C.T) and np.all(np.diag(r.C) == 0)


def test_elle_pmi_ve_jaccard():
    """['a','b','a','c'], pencere 1: C(a,b)=2, C(a,c)=1, C(b,c)=0; satır toplamları a=3, b=2, c=1; T=6.

    PMI(a,b) = log2(2·6 / (3·2)) = 1 ; PMI(a,c) = log2(1·6 / (3·1)) = 1. Tek bölüt: Jaccard = 1.
    """
    r, _ = C([B(["a", "b", "a", "c"])], pencere=1)
    k = kenar_tablosu(r, olcut="ham").set_index(["Kaynak", "Hedef"])
    assert k.loc[("a", "b"), "Birlikte görülme"] == 2 and k.loc[("a", "b"), "PMI"] == pytest.approx(1.0)
    assert k.loc[("a", "c"), "PMI"] == pytest.approx(1.0)
    assert (k["Jaccard"] == 1.0).all()


def test_birim_baglamlari_ve_jaccard():
    belgeler = [B(["a", "b"], ["a", "c"]), B(["a", "b"]), B(["c"])]
    r, c = C(belgeler, baglam="bolut")
    assert c("a", "b") == 2 and c("a", "c") == 1 and c("b", "c") == 0
    k = kenar_tablosu(r).set_index(["Kaynak", "Hedef"])
    # Bölütler: {a,b}, {a,c}, {a,b}, {c} → a: 3, b: 2, c: 2 ; J(a,b) = 2 / (3+2-2) = 2/3
    assert k.loc[("a", "b"), "Jaccard"] == pytest.approx(2 / 3, abs=1e-6)
    # PMI (birim): log2(U(a,b)·U / (U(a)·U(b))) = log2(2·4 / (3·2))
    assert k.loc[("a", "b"), "PMI"] == pytest.approx(math.log2(8 / 6), abs=1e-6)
    r, c = C(belgeler, baglam="belge")
    assert c("a", "b") == 2 and c("a", "c") == 1 and c("b", "c") == 1


# --- Bağımsız kaba kuvvet uygulamasıyla karşılaştırma (gerçek Türkçe veri) --

def _kaba_kuvvet(belgeler, sozcukler, baglam, w, sinir):
    ix = {s: i for i, s in enumerate(sozcukler)}
    M = np.zeros((len(ix), len(ix)), dtype=np.int64)
    for b in belgeler:
        birlesik = [[t for s in b.bolutler for t in s]]
        if baglam == "pencere":
            for u in (b.bolutler if sinir else birlesik):
                for p, q in itertools.combinations(range(len(u)), 2):
                    if q - p <= w and u[p] in ix and u[q] in ix and u[p] != u[q]:
                        M[ix[u[p]], ix[u[q]]] += 1
                        M[ix[u[q]], ix[u[p]]] += 1
        else:
            for u in (b.bolutler if baglam == "bolut" else birlesik):
                for a, c in itertools.combinations(sorted({ix[t] for t in u if t in ix}), 2):
                    M[a, c] += 1
                    M[c, a] += 1
    return M


@pytest.fixture(scope="module")
def ud_belgeler(ud):
    return derlem_on_isle([" ".join(c["metin"] for c in ud[i:i + 5]) for i in range(0, len(ud), 5)],
                          OnIslemeAyarlari())


@pytest.mark.parametrize("baglam, w, sinir", [("pencere", 1, True), ("pencere", 5, True), ("pencere", 5, False),
                                              ("pencere", 12, True), ("bolut", 5, True), ("belge", 5, True)])
def test_sayim_kaba_kuvvetle_ayni(ud_belgeler, baglam, w, sinir):
    r = birliktelik_hesapla(ud_belgeler, AgAyarlari(baglam, w, sinir, 300, 1))
    assert np.array_equal(r.C, _kaba_kuvvet(ud_belgeler, r.sozcukler, baglam, w, sinir))
    assert r.C.sum() > 0


def test_ppmi_ve_jaccard_bagimsiz_hesapla_ayni(ud_belgeler):
    r = birliktelik_hesapla(ud_belgeler, AgAyarlari(aday_sozcuk=300, min_birlikte=1))
    k = kenar_tablosu(r)
    ix = {s: i for i, s in enumerate(r.sozcukler)}
    satir = r.C.sum(axis=1)
    T = satir.sum()
    kume = {s: set() for s in r.sozcukler}
    u = 0
    for b in ud_belgeler:
        for s in b.bolutler:
            ws = {t for t in s if t in ix}
            if ws:
                for t in ws:
                    kume[t].add(u)
                u += 1
    assert len(k) > 300
    for x in k.itertuples(index=False):
        a, c = ix[x.Kaynak], ix[x.Hedef]
        assert x.PPMI == pytest.approx(max(math.log2(r.C[a, c] * T / (satir[a] * satir[c])), 0), abs=1e-5)
        assert x.Jaccard == pytest.approx(len(kume[x.Kaynak] & kume[x.Hedef]) / len(kume[x.Kaynak] | kume[x.Hedef]),
                                          abs=1e-5)


# --- Ağ seçimi ve merkezîlik --------------------------------------------------

def test_derece_networkx_ile_ayni(ud_belgeler):
    r = birliktelik_hesapla(ud_belgeler, AgAyarlari(aday_sozcuk=300, min_birlikte=1))
    for olcut in ("jaccard", "ppmi", "ham"):
        sec = ag_sec(r, olcut, 60, 120)
        d = dugum_tablosu(sec, r)
        G = nx.Graph()
        G.add_weighted_edges_from(zip(sec["Kaynak"], sec["Hedef"], sec["Ağırlık"]))
        assert len(sec) <= 120 and set(d["Sözcük"]) <= set(r.sozcukler[:60])
        for w, der, ag, norm in zip(d["Sözcük"], d["Derece"], d["Ağırlıklı derece"], d["Normalleştirilmiş derece"]):
            assert G.degree(w) == der
            assert G.degree(w, weight="weight") == pytest.approx(ag, abs=1e-5)
            assert nx.degree_centrality(G)[w] == pytest.approx(norm, abs=1e-6)


def test_siralama_ve_sinir():
    r, _ = C([B(["a", "b", "c", "d", "a", "b"]), B(["c", "d"])], pencere=1)
    k = kenar_tablosu(r, olcut="ham")
    assert k["Birlikte görülme"].is_monotonic_decreasing
    assert len(kenar_tablosu(r, olcut="ham", en_fazla=2)) == 2
    assert KENAR_DISA_AKTARIM_SINIRI == 100_000
    sec = ag_sec(r, "ppmi", 2, 10)
    assert set(sec["Kaynak"]) | set(sec["Hedef"]) <= set(r.sozcukler[:2])
    assert (sec["PPMI"] > 0).all()


def test_esik_bos_sonuc_ve_sekil(ud_belgeler):
    r = birliktelik_hesapla(ud_belgeler, AgAyarlari(aday_sozcuk=100, min_birlikte=10_000))
    assert kenar_tablosu(r).empty and ag_sec(r).empty and dugum_tablosu(ag_sec(r), r).empty
    r = birliktelik_hesapla(ud_belgeler, AgAyarlari(aday_sozcuk=200, min_birlikte=1))
    sec = ag_sec(r, "jaccard", 30, 40)
    fig = ag_figuru(sec, dugum_tablosu(sec, r), "jaccard")
    assert len(fig.data) == len(sec) + 2


@pytest.mark.parametrize("kw", [dict(baglam="x"), dict(pencere=0), dict(pencere=51), dict(aday_sozcuk=1),
                                dict(aday_sozcuk=5000), dict(min_birlikte=0)])
def test_gecersiz_ayarlar(kw):
    with pytest.raises(ValueError):
        AgAyarlari(**kw)


def test_belirlenimci(ud_belgeler):
    a = kenar_tablosu(birliktelik_hesapla(ud_belgeler, AgAyarlari(aday_sozcuk=200)))
    b = kenar_tablosu(birliktelik_hesapla(ud_belgeler, AgAyarlari(aday_sozcuk=200)))
    assert a.equals(b)


# --- Uygulama içinde -------------------------------------------------------

def test_uygulamada_ag_sekmesi():
    import os

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"), default_timeout=180).run()
    assert not at.exception and not at.error
    assert any("nedensellik" in i.value for i in at.info)
    for anahtar, deger in [("ag_baglam", "belge"), ("ag_olcut", "ppmi"), ("ag_baglam", "bolut"), ("ag_olcut", "ham")]:
        at.selectbox(key=anahtar).set_value(deger).run()
        assert not at.exception and not at.error
    at.checkbox(key="lemma").check().run()  # lemmatizasyon ağa da yansır
    at.number_input(key="ag_min").set_value(500).run()
    assert not at.exception
    assert any("gösterilecek bağlantı yok" in w.value for w in at.warning)
