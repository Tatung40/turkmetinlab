"""Frozen pre-refactor ConvergenceWarning oracle for NMF; small fixed Turkish corpus, no downloads."""
import warnings

import numpy as np
import pytest
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from tmlab.konu import KonuAyarlari, _model_kur, _vektorlestir
from tmlab.preprocess import IslenmisBelge

TEMALAR = [["öğretmen", "öğrenci", "sınıf", "ders", "ödev"], ["deprem", "çığ", "afet", "risk", "sığınak"],
           ["İstanbul", "ışık", "şehir", "trafik", "köprü"]]
BELGELER = [IslenmisBelge([TEMALAR[i % 3][j:] + TEMALAR[(i + 1) % 3][:1 + i % 2] + ["ortak"]])
            for i in range(12) for j in (0, 1)]


def old_converged(X, ayar):
    with warnings.catch_warnings(record=True) as uyarilar:
        warnings.simplefilter("always", ConvergenceWarning)
        m = NMF(n_components=ayar.konu_sayisi, init="nndsvda", random_state=ayar.tohum, max_iter=ayar.nmf_yineleme)
        W = m.fit_transform(X)
    durum = "hayır (yineleme sınırına ulaşıldı)" if any(issubclass(u.category, ConvergenceWarning)
                                                         for u in uyarilar) else "evet"
    return W, m.components_, durum


@pytest.mark.parametrize("yineleme", [1, 2, 5, 500])
def test_nmf_convergence_flag_matches_old_warning(yineleme):
    ayar = KonuAyarlari(yontem="nmf", konu_sayisi=3, nmf_yineleme=yineleme)
    X, _ = _vektorlestir(BELGELER, ayar, tfidf=True)
    W0, H0, durum0 = old_converged(X, ayar)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)  # the new path must not leak the warning
        W, H, ek = _model_kur(X, ayar, ayar.tohum)
    assert ek["NMF yakınsadı mı"] == durum0
    assert 1 <= ek["NMF yineleme sayısı"] <= yineleme
    np.testing.assert_allclose(W, W0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(H, H0, atol=1e-12, rtol=0)


def test_lda_perplexity_is_labelled_in_sample():
    ayar = KonuAyarlari(yontem="lda", konu_sayisi=3, lda_yineleme=5)
    X, _ = _vektorlestir(BELGELER, ayar, tfidf=False)
    _, H, ek = _model_kur(X, ayar, ayar.tohum)
    assert list(ek) == ["Eğitim şaşkınlığı (perplexity, örneklem içi)"] and np.isfinite(ek[list(ek)[0]])
    np.testing.assert_allclose(H.sum(axis=1), 1.0, atol=1e-12)
