"""2.5 staging arayüzü (AppTest): bayrak kapalıyken 2.1 aynı kalır; deneysel sekmenin akışı, hata senaryoları,
oturum izolasyonu, 100/500 belge. Küçük sınama modeli kullanılır (gerçek E5 değil)."""
import io
import os

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from anlamsal_model import kucuk_model_olustur

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(KOK, "app.py")
STAGING = os.path.join(KOK, "staging", "streamlit_app.py")
SEKMELER_21 = ["🏠 Başlarken", "📄 Veri", "🧹 Ön İşleme", "📚 Terimler", "📊 Sözcük Sıklığı", "🔗 N-gram", "⚖️ TF-IDF",
               "🧭 Benzerlik", "🕸️ Sözcük Ağı", "🧩 Konular", "👥 Gruplar", "☁️ Kelime Bulutu", "⬇️ İndir"]
DENEYSEL = "🧪 Anlamsal (deneysel)"
ORTAM = ("TML_DENEYSEL_ANLAMSAL", "TML_E5_ONNX_REPO", "TML_E5_ONNX_SHA256", "TML_E5_ONNX_YEREL", "TML_E5_ONNX_URL",
         "TML_MODEL_ONBELLEK")


@pytest.fixture(autouse=True)
def temiz_ortam(monkeypatch):
    for a in ORTAM:  # setenv: test bitince özgün durum geri yüklenir (staging girişi bayrağı açar)
        monkeypatch.setenv(a, "")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("e5"))
    kucuk_model_olustur(d, azami_token=512)
    return d


def calistir(yol=STAGING):
    at = AppTest.from_file(yol, default_timeout=300)
    at.run()
    return at


def ana_sekmeler(at):  # at.tabs iç içe sekmeleri de içerir
    return [t.label for t in at.tabs if t.label in set(SEKMELER_21) | {DENEYSEL}]


def sekme(at, ad=DENEYSEL):
    return next(t for t in at.tabs if t.label == ad)


def hatasiz(at):
    assert not at.exception, [e.value for e in at.exception]


def metinler(t):
    return " ".join(str(e.value) for e in list(t.error) + list(t.warning) + list(t.info) + list(t.caption))


def dugme(at, anahtar):
    return next((b for b in at.button if b.key == anahtar), None)


def hesapla(at):
    if dugme(at, "e5_model_hazirla"):
        dugme(at, "e5_model_hazirla").click().run()
    hatasiz(at)
    dugme(at, "e5_hesapla").click().run()
    hatasiz(at)
    return at


def yukle(at, ad, bayt, tur="text/csv"):
    at.radio(key="kaynak").set_value("Kendi dosyamı yükle").run()
    at.file_uploader[0].set_value((ad, bayt, tur)).run()
    return at


def sentetik_csv(n, tohum=0):
    rng = np.random.default_rng(tohum)
    sozcuk = ("eğitim öğrenci yapay zekâ ağ model veri analiz yöntem sonuç başarı uzaktan çevrimiçi öğretmen "
              "sistem enerji yapı malzeme deney ölçüm hasta tedavi sağlık").split()
    ozet = [" ".join(rng.choice(sozcuk, size=rng.integers(20, 50))) + f" belge{i}" for i in range(n)]
    return pd.DataFrame({"baslik": [f"Makale {i}" for i in range(n)], "ozet": ozet}).to_csv(index=False).encode()


def test_bayrak_kapaliyken_2_1_sekmeleri_aynen_kalir():
    at = calistir(APP)
    hatasiz(at)
    assert ana_sekmeler(at) == SEKMELER_21 and DENEYSEL not in [t.label for t in at.tabs]


def test_yapilandirma_yoksa_anlasilir_hata_ve_diger_sekmeler_calisir():
    at = calistir()
    hatasiz(at)
    assert ana_sekmeler(at) == SEKMELER_21[:-1] + [DENEYSEL, SEKMELER_21[-1]]
    assert "yapılandırılmamış" in metinler(sekme(at))
    assert len(at.metric) >= 4  # 2.1 özet ölçütleri


def test_basarisiz_indirme_arayuzde_tek_deneme(monkeypatch, tmp_path):
    monkeypatch.setenv("TML_E5_ONNX_REPO", "kullanici/e5")
    monkeypatch.setenv("TML_E5_ONNX_SHA256", "a" * 64)
    monkeypatch.setenv("TML_E5_ONNX_URL", "http://127.0.0.1:9")  # kapalı bağlantı noktası
    monkeypatch.setenv("TML_MODEL_ONBELLEK", str(tmp_path))
    at = calistir()
    dugme(at, "e5_model_hazirla").click().run()
    hatasiz(at)
    assert "indirilemedi" in metinler(sekme(at))
    assert dugme(at, "e5_hesapla") is None and len(at.metric) >= 4


def test_yerel_model_akisi_arama_disa_aktarma_ve_oturum_izolasyonu(monkeypatch, model):
    monkeypatch.setenv("TML_E5_ONNX_YEREL", model)
    at = hesapla(calistir())
    t = sekme(at)
    assert len(t.dataframe) >= 1 and dugme(at, "e5_indir") is None  # indirme düğmesi download_button'dır
    assert any(d.key == "e5_indir" for d in at.get("download_button"))
    at.text_input(key="e5_sorgu").input("uzaktan eğitimde öğrenci başarısı").run()
    hatasiz(at)
    at.radio(key="e5_k").set_value(10).run()
    hatasiz(at)
    # Ayrı oturum: model paylaşılır ama vektörler paylaşılmaz
    at2 = calistir()
    hatasiz(at2)
    assert dugme(at2, "e5_model_hazirla") is None and dugme(at2, "e5_hesapla") is not None


def test_uzun_belge_secimi_sessiz_kirpma_yok(monkeypatch, tmp_path):
    kucuk_model_olustur(str(tmp_path), azami_token=64)  # örnek özetlerin hepsi 64 belirteçten uzun
    monkeypatch.setenv("TML_E5_ONNX_YEREL", str(tmp_path))
    at = calistir()
    if dugme(at, "e5_model_hazirla"):
        dugme(at, "e5_model_hazirla").click().run()
    assert "belirteç sınırını aşıyor" in metinler(sekme(at))
    at.radio(key="e5_uzun").set_value("disla").run()
    hatasiz(at)
    assert "en az 2 belge" in metinler(sekme(at))
    at.radio(key="e5_uzun").set_value("kirp").run()
    dugme(at, "e5_hesapla").click().run()
    hatasiz(at)
    assert len(sekme(at).dataframe) >= 1


@pytest.mark.parametrize("n", [100, 500])
def test_100_ve_500_belge(monkeypatch, model, n):
    monkeypatch.setenv("TML_E5_ONNX_YEREL", model)
    at = yukle(calistir(), "sentetik.csv", sentetik_csv(n))
    hatasiz(at)
    hesapla(at)
    assert len(sekme(at).dataframe) >= 1
    assert "e5_vektor" in at.session_state["_tml_bellek"]
    dugme(at, "temizle").click().run()  # "Verilerimi temizle" vektörleri de siler
    hatasiz(at)
    assert "_tml_bellek" not in at.session_state or "e5_vektor" not in at.session_state["_tml_bellek"]


def test_belge_siniri_asilinca_durur(monkeypatch, model):
    monkeypatch.setenv("TML_E5_ONNX_YEREL", model)
    at = yukle(calistir(), "buyuk.csv", sentetik_csv(501))
    hatasiz(at)
    assert "en fazla 500 belge" in metinler(sekme(at))


def test_hatali_dosya_deneysel_sekmeyi_bozmaz(monkeypatch, model):
    monkeypatch.setenv("TML_E5_ONNX_YEREL", model)
    at = yukle(calistir(), "bozuk.xlsx", b"PK\x03\x04bozuk", "application/vnd.ms-excel")
    hatasiz(at)
    assert at.error  # 2.1'in okuma hatası iletisi
    assert dugme(at, "e5_hesapla") is None
