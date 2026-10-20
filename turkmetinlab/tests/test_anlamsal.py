"""2.5 deneysel anlamsal modül: kodlayıcı, skorlar, model indirme (küçük sınama modeliyle; gerçek E5 değil)."""
import functools
import http.server
import os
import shutil
import threading
import time

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from anlamsal_model import kucuk_model_olustur, referans_vektor
from tmlab import anlamsal as an


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("e5"))
    return d, kucuk_model_olustur(d, azami_token=32)


@pytest.fixture(scope="module")
def kod(model):
    return an.E5Kodlayici(model[0])


def test_kodlayici_bagimsiz_hesapla_ayni_ve_sira_korunur(model, kod):
    d, E = model
    metin = ["kısa", "biraz daha uzun bir metin", "a", "x" * 80, "orta uzunlukta metin", "b" * 10] * 3
    V = kod.kodla(metin, an.SORGU_ONEKI)
    ref = np.array([referans_vektor(E, an.SORGU_ONEKI + m, 32, os.path.join(d, "tokenizer.json")) for m in metin])
    assert np.abs(V - ref).max() < 1e-5  # maskeli ortalama, kırpma, L2 ve özgün sıra
    assert np.allclose(np.linalg.norm(V, axis=1), 1, atol=1e-5)
    for b in (1, 3, 32):
        assert np.abs(kod.kodla(metin, an.SORGU_ONEKI, toplu_is=b) - V).max() < 1e-5


def test_belirtec_sayisi_kirpmasiz_ve_kodlama_kirpar(kod):
    uzun = "x" * 100
    assert kod.belirtec_sayilari([uzun], "")[0] == 102  # <s> + 100 + </s>; kırpılmadan sayılır
    assert np.allclose(kod.kodla([uzun], ""), kod.kodla(["x" * 200], ""), atol=1e-6)  # ikisi de 32'de kırpılır


def test_ilerleme_ve_bos_girdi(kod):
    adim = []
    kod.kodla(["a"] * 20, "", ilerleme=adim.append)
    assert adim[-1] == 1.0 and adim == sorted(adim) and len(adim) == 3
    assert kod.kodla([], "").shape[0] == 0


def test_es_zamanli_istekler_siraya_girer(kod, monkeypatch):
    etkin, en_cok, bekledi = [0], [0], []
    asil = kod._kodla_toplu

    def izle(m):
        etkin[0] += 1
        en_cok[0] = max(en_cok[0], etkin[0])
        time.sleep(0.02)
        try:
            return asil(m)
        finally:
            etkin[0] -= 1
    monkeypatch.setattr(kod, "_kodla_toplu", izle)
    iplik = [threading.Thread(target=kod.kodla, args=(["a b c"] * 24, ""), kwargs={"bekleme": lambda: bekledi.append(1)})
             for _ in range(3)]
    [t.start() for t in iplik]
    [t.join() for t in iplik]
    assert en_cok[0] == 1 and bekledi  # model aynı anda tek istek için çalışır


@pytest.mark.parametrize("smooth,sublinear", [(True, False), (False, True), (True, True)])
def test_tfidf_sorgu_vektoru_scikit_learn_ile_ayni(smooth, sublinear):
    belgeler = [["uzaktan", "eğitim", "öğrenci", "eğitim"], ["yapay", "zekâ", "eğitim"], ["öğrenci", "başarı"],
                ["yapay", "sinir", "ağı", "zekâ", "zekâ"]]
    v = TfidfVectorizer(analyzer=lambda x: x, lowercase=False, smooth_idf=smooth, sublinear_tf=sublinear, norm="l2")
    M = v.fit_transform(belgeler)
    sorgu = ["eğitim", "zekâ", "zekâ", "bilinmeyen"]
    beklenen = v.transform([sorgu]).toarray()[0]
    q = an.tfidf_sorgu_vektoru(M, v.get_feature_names_out(), sorgu, smooth, sublinear)
    assert np.allclose(q, beklenen)
    assert not an.tfidf_sorgu_vektoru(M, v.get_feature_names_out(), ["yok"], smooth, sublinear).any()


def test_hibrit_fizibilite_formulu():
    rng = np.random.default_rng(1)
    A, B = rng.random((6, 6)), rng.random((6, 6))
    np.fill_diagonal(A, -np.inf)
    np.fill_diagonal(B, -np.inf)
    H = an.hibrit(A, B)
    i = ~np.eye(6, dtype=bool)
    for S in (A, B):
        z = an.znorm(S)
        assert np.allclose([z[r][i[r]].mean() for r in range(6)], 0, atol=1e-9)  # kendisi hariç ortalama 0
    assert np.allclose(H[i], (an.znorm(A) + an.znorm(B))[i]) and (H[~i] < -1e8).all()
    assert an.ust_k_ortusme(A, A, 3) == 1.0
    assert an.en_yakinlar(np.array([0.1, 0.9, 0.5]), 5, haric=1) == [2, 0]


# --- Model indirme -----------------------------------------------------------

class _Sunucu:
    def __init__(self, kok):
        self.istek = []
        sahip = self

        class Isleyici(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                sahip.istek.append(self.path)
                return super().do_GET()
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Isleyici, directory=kok))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.adres = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def kapat(self):
        self.httpd.shutdown()


@pytest.fixture
def depo(model, tmp_path, monkeypatch):
    d = tmp_path / "sunucu" / "kullanici" / "e5" / "resolve" / "main"
    shutil.copytree(model[0], d)
    s = _Sunucu(str(tmp_path / "sunucu"))
    monkeypatch.setenv("TML_MODEL_ONBELLEK", str(tmp_path / "onbellek"))
    yield s, an.sha256_dosya(str(d / "model_fp32.onnx"))
    s.kapat()


def _kaynak(s, sha, **k):
    return an.ModelKaynagi(repo="kullanici/e5", sha256=sha, adres=s.adres, **k)


def test_indirme_dogrulama_ve_onbellek(depo):
    s, sha = depo
    olay = []
    klasor = an.model_hazirla(_kaynak(s, sha), lambda ad, o: olay.append(ad))
    assert sorted(os.listdir(klasor)) == sorted(an.DOSYALAR) and set(olay) == set(an.DOSYALAR)
    n = len(s.istek)
    assert an.model_hazirla(_kaynak(s, sha)) == klasor and len(s.istek) == n  # ikinci kez indirilmez
    assert an.E5Kodlayici(klasor).kodla(["a"], "").shape == (1, 16)


def test_ozet_uyusmazsa_reddedilir_ve_dosya_birakilmaz(depo):
    s, _ = depo
    with pytest.raises(an.ModelHatasi, match="özet"):
        an.model_hazirla(_kaynak(s, "0" * 64))
    klasor = an.onbellek_klasoru(_kaynak(s, "0" * 64))
    assert not [f for f in os.listdir(klasor) if f in an.DOSYALAR or f.endswith(".indiriliyor")]


def test_basarisiz_indirme_tek_deneme(depo):
    s, sha = depo
    s.kapat()
    with pytest.raises(an.ModelHatasi, match="indirilemedi"):
        an.model_hazirla(_kaynak(s, sha), zaman_asimi=3)
    s2 = _Sunucu(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        an.onbellek_klasoru(_kaynak(s, sha))))))))  # var olmayan depo: 404
    with pytest.raises(an.ModelHatasi, match="indirilemedi"):
        an.model_hazirla(an.ModelKaynagi(repo="yok/yok", sha256=sha, adres=s2.adres))
    assert len(s2.istek) == 1  # yeniden deneme yok
    s2.kapat()


def test_eksik_yapilandirma():
    assert "yapılandırılmamış" in an.ModelKaynagi().eksik()
    assert "yapılandırılmamış" in an.ModelKaynagi(repo="a/b", sha256="kisa").eksik()
    with pytest.raises(an.ModelHatasi):
        an.model_hazirla(an.ModelKaynagi(yerel="/olmayan/klasor"))


def test_500_belge_parcali_kodlama(kod):
    rng = np.random.default_rng(0)
    metin = ["".join(rng.choice(list("abcçdefgğ "), size=rng.integers(5, 300))) for _ in range(500)]
    adim = []
    t = time.perf_counter()
    V = kod.kodla(metin, an.BELGE_ONEKI, ilerleme=adim.append)
    assert V.shape == (500, 16) and len(adim) == 63 and time.perf_counter() - t < 30
