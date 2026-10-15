"""Modül 1 — Belge benzerliği testleri (elle hesaplanmış örnek + bağımsız uygulamalar)."""

import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity

import tmlab.benzerlik as bz
from tmlab.analysis import tfidf_hesapla
from tmlab.benzerlik import (belge_siralamasi, benzerlik_ozeti, ciftler_komsulardan, dendrogram_figuru,
                             en_benzer_ciftler, en_yakin_komsular, hiyerarsik_kumeleme, isi_haritasi,
                             kosinus_matrisi, kume_etiketleri, kume_tablosu, siluet_tablosu)
from tmlab.preprocess import IslenmisBelge, OnIslemeAyarlari, derlem_on_isle


def belgeler(*listeler):
    return [IslenmisBelge([list(x)]) for x in listeler]


# --- Elle hesaplanabilir örnek ---------------------------------------------

def test_elle_hesaplanan_kosinus():
    """A = {elma, armut}, B = {elma, kiraz}, C = {muz}; yumuşatılmış IDF, N = 3.

    idf(elma) = ln(4/3) + 1 ;  idf(armut) = idf(kiraz) = idf(muz) = ln(4/2) + 1
    cos(A, B) = idf(elma)² / (idf(elma)² + idf(armut)²) ;  cos(A, C) = 0
    """
    s = tfidf_hesapla(belgeler(["elma", "armut"], ["elma", "kiraz"], ["muz"]), ["A", "B", "C"])
    S = kosinus_matrisi(s.matris)
    e, a = math.log(4 / 3) + 1, math.log(2) + 1
    assert S[0, 1] == pytest.approx(e * e / (e * e + a * a), abs=1e-12)  # ≈ 0,3664
    assert S[0, 1] == pytest.approx(0.36644, abs=1e-4)
    assert S[0, 2] == 0 and S[1, 2] == 0
    assert np.allclose(np.diag(S), 1) and np.allclose(S, S.T)


def test_ayni_ve_ayrik_belgeler():
    s = tfidf_hesapla(belgeler(["a", "b"], ["a", "b"], ["c", "d"]))
    S = kosinus_matrisi(s.matris)
    assert S[0, 1] == pytest.approx(1.0) and S[0, 2] == 0.0


# --- Bağımsız uygulamalarla karşılaştırma ---------------------------------

@pytest.fixture(scope="module")
def ud_tfidf(ud):
    metinler = [" ".join(c["metin"] for c in ud[i:i + 5]) for i in range(0, len(ud), 5)]
    return tfidf_hesapla(derlem_on_isle(metinler, OnIslemeAyarlari()))


def test_sklearn_ve_saf_python_ile_ayni(ud_tfidf):
    X = ud_tfidf.matris
    S = kosinus_matrisi(X)
    assert np.abs(S - cosine_similarity(X)).max() < 1e-12
    satirlar = [dict(zip(X[i].indices, X[i].data)) for i in range(X.shape[0])]

    def cos(a, b):
        return sum(v * b.get(k, 0.0) for k, v in a.items()) / math.sqrt(sum(v * v for v in a.values())) \
            / math.sqrt(sum(v * v for v in b.values()))

    for i in range(0, X.shape[0], 7):
        for j in range(X.shape[0]):
            assert S[i, j] == pytest.approx(cos(satirlar[i], satirlar[j]), abs=1e-12)


def test_parcali_komsu_tam_matrisle_ayni_esitlik_agirlikli():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n, v = int(rng.integers(5, 90)), int(rng.integers(3, 25))
        A = sp.csr_matrix((rng.random((n, v)) < 0.15).astype(float))
        norm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel())
        norm[norm == 0] = 1
        A = sp.csr_matrix(sp.diags(1 / norm) @ A)
        S = kosinus_matrisi(A)
        k = int(rng.integers(1, n))
        I, V = en_yakin_komsular(A, k, parca=int(rng.integers(1, 15)))
        beklenen = [belge_siralamasi(S, list(range(n)), i, k)["Benzer belge"].tolist() for i in range(n)]
        assert I.tolist() == beklenen
        assert np.allclose(V, np.take_along_axis(S, I, 1))


def test_ciftler_komsulardan_tam_ciftlerle_ayni(ud_tfidf):
    S = kosinus_matrisi(ud_tfidf.matris)
    adlar = ud_tfidf.belge_adlari
    tam = en_benzer_ciftler(S, adlar, 40)
    komsu = ciftler_komsulardan(*en_yakin_komsular(ud_tfidf.matris, 40), adlar, 40)
    assert tam.equals(komsu)
    assert (tam["Kosinüs benzerliği"].diff().dropna() <= 1e-12).all()  # azalan sıra


def test_kumeleme_scipy_dogrudan_hesapla_ayni(ud_tfidf):
    X = ud_tfidf.matris
    S = kosinus_matrisi(X)
    assert np.abs(hiyerarsik_kumeleme(S, "average") - linkage(pdist(X.toarray(), "cosine"), "average")).max() < 1e-9
    assert np.abs(hiyerarsik_kumeleme(S, "ward") - linkage(X.toarray(), "ward")).max() < 1e-9


def test_bariz_iki_kume_ayrilir():
    s = tfidf_hesapla(belgeler(["deprem", "fay", "zemin"], ["deprem", "fay", "sismik"], ["fay", "zemin", "sismik"],
                               ["korozyon", "çelik", "asit"], ["korozyon", "çelik", "inhibitör"],
                               ["çelik", "asit", "inhibitör"]))
    S = kosinus_matrisi(s.matris)
    for yontem in ("average", "ward"):
        e = kume_etiketleri(hiyerarsik_kumeleme(S, yontem), 2)
        assert len(set(e[:3])) == 1 and len(set(e[3:])) == 1 and e[0] != e[3]
    t = siluet_tablosu(S, hiyerarsik_kumeleme(S, "average"), "average", 4)
    assert t.iloc[0]["Ortalama siluet"] > 0.5 and t["Ortalama siluet"].between(-1, 1).all()
    kt = kume_tablosu(kume_etiketleri(hiyerarsik_kumeleme(S, "average"), 2), list("ABCDEF"))
    assert kt["Kümedeki belge sayısı"].tolist() == [3] * 6


# --- Tablolar, özet ve grafikler -------------------------------------------

def test_tablolar_ve_ozet():
    s = tfidf_hesapla(belgeler(["a", "b"], ["a", "c"], ["d"]), ["X", "Y", "Z"])
    S = kosinus_matrisi(s.matris)
    c = en_benzer_ciftler(S, s.belge_adlari, 10)
    assert c.iloc[0][["Belge A", "Belge B"]].tolist() == ["X", "Y"] and len(c) == 3
    r = belge_siralamasi(S, s.belge_adlari, 0, 5)
    assert r["Benzer belge"].tolist() == ["Y", "Z"]
    o = benzerlik_ozeti(S)
    assert o["Belge çifti sayısı"] == 3 and o["Hiç ortak terimi olmayan çiftler (%)"] == pytest.approx(66.67)


def test_grafikler():
    s = tfidf_hesapla(belgeler(["a", "b"], ["a", "c"], ["c", "d"], ["e"]), list("ABCD"))
    S = kosinus_matrisi(s.matris)
    assert isi_haritasi(S, s.belge_adlari).data[0].z.shape == (4, 4)
    fig, yapraklar = dendrogram_figuru(hiyerarsik_kumeleme(S, "average"), s.belge_adlari, "average")
    assert sorted(yapraklar) == [0, 1, 2, 3] and len(fig.data) == 3


# --- Hatalı girdiler ve sınırlar ------------------------------------------

def test_normalize_olmayan_girdi_ve_sinir(monkeypatch):
    with pytest.raises(ValueError, match="L2"):
        kosinus_matrisi(sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 1.0]])))
    monkeypatch.setattr(bz, "YOGUN_MATRIS_SINIRI", 2)
    with pytest.raises(ValueError, match="en fazla"):
        kosinus_matrisi(sp.csr_matrix(np.eye(3)))


def test_kucuk_derlem_kumeleme_hatasi():
    with pytest.raises(ValueError, match="en az 3"):
        hiyerarsik_kumeleme(np.eye(2))
    with pytest.raises(ValueError, match="Geçersiz"):
        hiyerarsik_kumeleme(np.eye(3), "single")


def test_komsu_k_belge_sayisini_asamaz():
    I, V = en_yakin_komsular(sp.csr_matrix(np.eye(3)), k=10)
    assert I.shape == (3, 2)


# --- Uygulama içinde -------------------------------------------------------

def _uygulama():
    import os

    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"), default_timeout=180)


def test_uygulamada_benzerlik_ve_kumeleme():
    at = _uygulama().run()
    assert not at.exception and not at.error
    assert any("intihal" in i.value for i in at.info)  # yöntemsel uyarı görünür
    at.checkbox(key="benz_kume").check().run()
    at.number_input(key="benz_k").set_value(3).run()
    at.selectbox(key="benz_yontem").set_value("ward").run()
    assert not at.exception and not at.error


def test_uygulamada_buyuk_derlem_modu(monkeypatch):
    monkeypatch.setattr(bz, "YOGUN_MATRIS_SINIRI", 5)
    at = _uygulama().run()
    assert not at.exception
    assert any("tam benzerlik matrisi" in w.value for w in at.warning)
    at.button(key="benz_knn").click().run()
    assert not at.exception and not at.error
    assert "benz_kume" not in [c.key for c in at.checkbox]  # kümeleme bu modda sunulmaz
