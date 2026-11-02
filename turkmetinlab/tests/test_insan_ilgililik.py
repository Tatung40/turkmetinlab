"""degerlendirme/insan_ilgililik.py: yalnız sentetik veri; gerçek puanlayıcı verisi depoda yoktur."""
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "degerlendirme"))
import insan_ilgililik as ii  # noqa: E402

n = np.nan
# Krippendorff (2011) "Computing Krippendorff's alpha-reliability" örneği: 4 kodlayıcı, 12 birim, eksik değerli.
K2011 = np.array([[1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n], [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
                  [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n], [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n]])


@pytest.mark.parametrize("duzey,beklenen", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)])
def test_alpha_matches_published_reference(duzey, beklenen):
    assert ii.alfa_cakisma(K2011, duzey) == pytest.approx(beklenen, abs=5e-4)


@pytest.mark.parametrize("duzey", ["nominal", "ordinal", "interval"])
def test_alpha_matches_krippendorff_package(duzey):
    krippendorff = pytest.importorskip("krippendorff")
    assert ii.alfa_cakisma(K2011, duzey) == pytest.approx(krippendorff.alpha(K2011, level_of_measurement=duzey), abs=1e-10)


SORGULAR = {"S1": "Öğretmenlerin ölçme ve değerlendirme yeterlikleri ilkokulda incelenmiştir.",
            "S2": "Ortaokul matematik öğretim programı öğretmen görüşlerine göre değerlendirilmiştir."}
ADAYLAR = {"S1": ["İlkokul öğretmenlerinin ölçme değerlendirme yeterlikleri ve uygulamaları incelendi.",
                  "Ölçme ve değerlendirme sürecinde öğretmen yeterliği ele alındı.",
                  "Turizm önlisans programının ekonomik etkileri tartışıldı.",
                  "Okul öncesi çocukların oyun davranışları gözlendi."],
           "S2": ["Ortaokul matematik dersi öğretim programı öğretmen görüşleriyle değerlendirildi.",
                  "Matematik programının kazanımları ortaokul öğretmenlerine göre incelendi.",
                  "İngilizce öğretim programı bilim sanat merkezlerinde değerlendirildi.",
                  "Felsefe dersinde eleştirel düşünme etkinlikleri uygulandı."]}
PUAN = [2, 1, 0, 0]


def _excel(tmp_path, ad, puanlar, sira):
    satirlar = []
    for q, adaylar in ADAYLAR.items():
        for j, a in enumerate(adaylar):
            satirlar.append([None, q, SORGULAR[q], f"Başlık {q}-{j}", a, puanlar.get((q, j), PUAN[j]), None])
    satirlar = [satirlar[i] for i in sira]
    for i, s in enumerate(satirlar):
        s[0] = f"K{i + 1:03d}"  # kodlar puanlayıcıya göre farklı sıraya düşer
    t = pd.DataFrame(satirlar, columns=["Kayıt kodu", "Sorgu başlığı", "Sorgu özeti", "Aday tez başlığı", "Aday özeti",
                                        "Puan (0/1/2)", "Kısa gerekçe (isteğe bağlı)"])
    yol = tmp_path / f"{ad}.xlsx"
    t.to_excel(yol, index=False)
    return str(yol)


@pytest.fixture
def uc_dosya(tmp_path):
    return [_excel(tmp_path, "P1", {}, range(8)),
            _excel(tmp_path, "P2", {("S1", 1): None}, [7, 6, 5, 4, 3, 2, 1, 0]),  # ters sıra + eksik puan
            _excel(tmp_path, "P3", {("S2", 2): 1}, [3, 0, 6, 1, 5, 2, 7, 4])]


def test_rows_matched_by_text_not_order_and_missing_not_imputed(uc_dosya):
    tablolar = {f"P{i}": ii.puanlayici_oku(y) for i, y in enumerate(uc_dosya, start=1)}
    M, ciftler, butunluk = ii.birlestir(tablolar)
    assert M.shape == (3, 8) and len(set(ciftler)) == 8
    assert butunluk["Eksik puan"].tolist() == [0, 1, 0]
    assert np.isnan(M).sum() == 1
    s1 = ii.anahtar(SORGULAR["S1"])
    for j, a in enumerate(ADAYLAR["S1"]):
        k = ciftler.index((s1, ii.anahtar(a)))
        if j != 1:
            assert M[0, k] == M[1, k] == PUAN[j]


def test_mismatched_pair_sets_and_invalid_scores_are_rejected(tmp_path, uc_dosya):
    t = pd.read_excel(uc_dosya[0])
    t.loc[0, "Aday özeti"] = "Farklı bir aday özeti"
    t.to_excel(tmp_path / "farkli.xlsx", index=False)
    tablolar = {"P1": ii.puanlayici_oku(str(tmp_path / "farkli.xlsx")), "P2": ii.puanlayici_oku(uc_dosya[1])}
    with pytest.raises(ValueError, match="çift kümeleri farklı"):
        ii.birlestir(tablolar)
    t = pd.read_excel(uc_dosya[0])
    t.loc[0, "Puan (0/1/2)"] = 3
    t.to_excel(tmp_path / "gecersiz.xlsx", index=False)
    with pytest.raises(ValueError, match="0/1/2 dışında"):
        ii.puanlayici_oku(str(tmp_path / "gecersiz.xlsx"))


def test_perfect_ranking_metrics():
    m = ii.siralama_olcutleri(np.array([0.9, 0.5, 0.2, 0.1]), np.array([2.0, 1.0, 0.0, 0.0]),
                              np.array([True, True, False, False]))
    assert m["nDCG@5"] == pytest.approx(1.0) and m["AP"] == m["RR"] == m["P@5"] * 2 == 1.0
    assert m["Spearman"] == pytest.approx(np.corrcoef([4, 3, 2, 1], [4, 3, 1.5, 1.5])[0, 1])


def test_end_to_end_report_marks_e5_and_hybrid_not_run(uc_dosya, tmp_path, monkeypatch):
    for d in ("TML_E5_ONNX_REPO", "TML_E5_ONNX_SHA256", "TML_E5_ONNX_YEREL"):
        monkeypatch.delenv(d, raising=False)
    cikti = tmp_path / "rapor.xlsx"
    assert ii.main([*uc_dosya, "--cikti", str(cikti), "--bootstrap", "50"]) == 0
    wb = openpyxl.load_workbook(cikti)
    assert {"Uyum_Kappa", "Uyum_Alfa", "Uzlasma_Gerekli", "Model_Karsilastirma", "Model_E5", "Model_Hibrit"} <= set(wb.sheetnames)
    assert wb["Model_E5"]["A2"].value == ii.NOT_RUN and wb["Model_Hibrit"]["A2"].value == ii.NOT_RUN
    karsilastirma = pd.read_excel(cikti, sheet_name="Model_Karsilastirma")
    assert karsilastirma["Yöntem"].tolist() == ["TF-IDF"]
    uzlasma = pd.read_excel(cikti, sheet_name="Uzlasma_Gerekli")
    assert len(uzlasma) == 2  # P2'nin eksik puanı ve P3'ün farklı puanı; uyuşan çiftler listelenmez


def test_e5_and_hybrid_use_real_encoder_path_with_small_model(uc_dosya, tmp_path, monkeypatch):
    """Küçük sınama ONNX modeliyle gerçek E5 kod yolu (gerçek E5 değil): skorlar kodlayıcıyla aynı, hibrit sabit."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    sys.path.insert(0, str(Path(__file__).parent))
    from anlamsal_model import kucuk_model_olustur
    from tmlab import anlamsal as an

    klasor = str(tmp_path / "e5")
    kucuk_model_olustur(klasor, azami_token=32)
    monkeypatch.setenv("TML_E5_ONNX_YEREL", klasor)
    tablolar = {f"P{i}": ii.puanlayici_oku(y) for i, y in enumerate(uc_dosya, start=1)}
    sonuc = ii.degerlendir(tablolar, tekrar=20)
    assert sonuc["e5_gerekce"] is None
    assert sonuc["karsilastirma"]["Yöntem"].tolist() == ["TF-IDF", "E5", "Hibrit"]
    ilk = next(iter(tablolar.values()))
    metinler = {**dict(zip(ilk["sorgu"], ilk["sorgu_metni"])), **dict(zip(ilk["aday"], ilk["aday_metni"]))}
    V = an.E5Kodlayici(klasor).kodla([metinler[k] for k in sorted(metinler)], an.SORGU_ONEKI)
    S = V @ V.T
    pos = {k: i for i, k in enumerate(sorted(metinler))}
    q, c = ii.anahtar(SORGULAR["S1"]), ii.anahtar(ADAYLAR["S1"][0])
    satir = sonuc["ciftler"].query("Sorgu == 'S1'").iloc[0]
    assert satir["E5 skoru"] == pytest.approx(round(float(S[pos[q], pos[satir_aday(satir, metinler, pos)]]), 6), abs=1e-6)
    assert sonuc["sorgu_olcutleri"].groupby("Yöntem").size().to_dict() == {"TF-IDF": 2, "E5": 2, "Hibrit": 2}


def satir_aday(satir, metinler, pos):
    baslik_to_aday = {f"Başlık {q}-{j}": ii.anahtar(a) for q, adaylar in ADAYLAR.items() for j, a in enumerate(adaylar)}
    return baslik_to_aday[satir["Aday başlığı"]]
