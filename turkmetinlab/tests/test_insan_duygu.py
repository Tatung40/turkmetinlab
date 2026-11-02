"""degerlendirme/insan_duygu.py: yalnız sentetik veri; gerçek puanlayıcı verisi depoda yoktur, ağ kullanılmaz."""
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "degerlendirme"))
import insan_duygu as idu  # noqa: E402


def test_decision_rule_keeps_label_mapping():
    assert (idu.karar(2, 1), idu.karar(1, 2), idu.karar(1, 1), idu.karar(0, 0)) == (1, 2, 0, 0)
    assert idu.ETIKETLER == {0: "NÖTR", 1: "OLUMLU", 2: "OLUMSUZ"}


def test_fleiss_formula_matches_published_example():
    # Fleiss (1971) / yaygın ders örneği: 10 birim, 14 puanlayıcı, 5 kategori, kappa = 0,210.
    sayim = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    C = np.array([[k for k, s in enumerate(satir) for _ in range(s)] for satir in sayim]).T
    sayim3 = np.array([[np.sum(C[:, j] == k) for k in range(5)] for j in range(10)])
    m = C.shape[0]
    P = ((sayim3 ** 2).sum(1) - m) / (m * (m - 1))
    p = sayim3.sum(0) / sayim3.sum()
    assert (P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum()) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_wrapper_agrees_with_statsmodels_when_available():
    rng = np.random.default_rng(0)
    C = rng.integers(0, 3, size=(6, 40))
    deger, kaynak = idu.fleiss_kappa(C)
    sm = pytest.importorskip("statsmodels.stats.inter_rater")
    t, _ = sm.aggregate_raters(C.T, n_cat=3)
    assert deger == pytest.approx(sm.fleiss_kappa(t, method="fleiss"), abs=1e-12) and kaynak == "statsmodels"


def test_vote_table_separates_unanimity_majority_ties_and_missing():
    n = np.nan
    M = np.array([[1, 1, 0, 0, 1], [1, 1, 0, 1, 1], [1, 1, 0, 2, n], [1, 1, 1, 0, 1], [1, 2, 1, 0, 2], [1, 2, 1, 1, 0]], float)
    # Sütun 3: 3×0, 2×1, 1×2 → çoğunluk yok, eşitlik yok. Sütun 2: 3–3 eşitlik. Sütun 4: 5 oyun 3ü OLUMLU (eksik bir puan).
    t = idu.oy_tablosu(M, [f"P{i}" for i in range(1, 7)], [(f"U{j}", f"metin {j}") for j in range(5)])
    assert t["Durum"].tolist() == ["oy birliği", "mutlak çoğunluk", "eşitlik — uzlaşma gerekli",
                                   "çoğunluk yok — uzlaşma gerekli", "mutlak çoğunluk"]
    assert t["Oy sayısı"].tolist() == [6, 6, 6, 6, 5]
    assert t["Mutlak çoğunluk"].isna().tolist() == [False, False, True, True, False]
    assert pd.isna(t["Güçlü çoğunluk (oran ≥ 5/6)"].iat[1])  # 4/6 güçlü değil


def _dosyalar(tmp_path, etiketler):
    yollar = []
    yorumlar = ["Harika ürün, çok güzel", "Berbat, hiç beğenmedim", "Kargo geldi", "Güzel ama pahalı"]
    for i, e in enumerate(etiketler, start=1):
        satir = [[f"URUN-{j + 1:03d}", y, e[j]] for j, y in enumerate(yorumlar)]
        sira = satir[::-1] if i % 2 else satir  # satır sırasına güvenilmez
        yol = tmp_path / f"D{i}.xlsx"
        pd.DataFrame(sira, columns=["Yorum ID", "Ürün yorumu", "Puanlayıcı "]).to_excel(yol, index=False)
        yollar.append(str(yol))
    return yollar


def test_end_to_end_with_injected_methods_and_electra_not_run(tmp_path):
    yollar = _dosyalar(tmp_path, [[1, 2, 0, 1], [1, 2, 0, 2], [1, 2, 0, 0], [1, 2, None, 1]])
    tablolar = {f"P{i}": idu.puanlayici_oku(y) for i, y in enumerate(yollar, start=1)}
    M, kayitlar, butunluk, _ = idu.birlestir(tablolar)
    assert [k[0] for k in kayitlar] == ["URUN-001", "URUN-002", "URUN-003", "URUN-004"]
    assert butunluk["Eksik etiket"].tolist() == [0, 0, 0, 1] and np.isnan(M).sum() == 1
    sahte = {"sonuc": {"Sözlük": {"tahmin": np.array([1, 2, 1, 1]), "kapsam": np.array([True, True, False, True]),
                                  "sayim": [(1, 0), (0, 1), (0, 0), (1, 1)]}},
             "notlar": {"ELECTRA": idu.NOT_RUN + ": sınama"}}
    sonuc = idu.degerlendir(tablolar, yontemler=sahte)
    b = sonuc["basari"].set_index("Referans").loc["Mutlak çoğunluk"]
    assert (b["n"], b["Accuracy"]) == (3, round(2 / 3, 4))  # URUN-004 (1/2/0/1) çoğunluksuz → kapsam dışı
    cikti = tmp_path / "rapor.xlsx"
    idu.excel_yaz(sonuc, str(cikti))
    adlar = set(openpyxl.load_workbook(cikti).sheetnames)
    assert {"Veri_Butunlugu", "Puanlayici_Uyumu", "Sinif_Dagilimlari", "Uzlasma_Gerekli", "Model_Basari",
            "Karmaşıklık_Matrisleri", "Hata_Analizi", "Yontem_Sinirliliklari"} <= adlar
    notlar = pd.read_excel(cikti, sheet_name="Yontem_Sinirliliklari")
    assert notlar["Not"].str.startswith(idu.NOT_RUN).any()


def test_mismatched_comments_rejected(tmp_path):
    yollar = _dosyalar(tmp_path, [[1, 2, 0, 1], [1, 2, 0, 1]])
    t = pd.read_excel(yollar[1])
    t.loc[0, "Ürün yorumu"] = "Farklı metin"
    t.to_excel(yollar[1], index=False)
    with pytest.raises(ValueError, match="kümeleri farklı"):
        idu.birlestir({"P1": idu.puanlayici_oku(yollar[0]), "P2": idu.puanlayici_oku(yollar[1])})


def test_two_rater_format_mapping_invalid_cell_and_single_vote(tmp_path):
    """Veri B biçimi: farklı sütun adları, 1/2/3 kodları (3 = NÖTR yalnız analizde), metinsel hücre."""
    yorumlar = ["Harika", "Berbat", "Kargo geldi", "İdare eder", "Fena değil"]
    kodlar = [[1, 2, 3, "1 sonradan 2", 3], [1, 2, 3, 2, None]]
    yollar = []
    for i, k in enumerate(kodlar, start=1):
        satir = [[f"GRUP2-{j + 1:03d}", y, k[j]] for j, y in enumerate(yorumlar)] + [[None, None, None]] * 3
        yol = tmp_path / f"DT{i}.xlsx"
        pd.DataFrame(satir, columns=["Yorum ID", "Müşteri ürün yorumu", "Puanlayıcı kodu (1, 2, 3)"]).to_excel(yol, index=False)
        yollar.append(str(yol))
    ham = pd.read_excel(yollar[0])
    tablolar = {f"P{i}": idu.puanlayici_oku(y, {3.0: 0.0}) for i, y in enumerate(yollar, start=1)}
    assert pd.read_excel(yollar[0]).equals(ham)  # özgün dosya değişmez
    assert tablolar["P1"]["etiket"].tolist()[:3] == [1, 2, 0] and np.isnan(tablolar["P1"]["etiket"].iat[3])
    M, kayitlar, butunluk, gecersiz = idu.birlestir(tablolar)
    assert len(kayitlar) == 5 and butunluk["Geçersiz hücre (eksik sayıldı)"].tolist() == [1, 0]
    oy = idu.oy_tablosu(M, ["P1", "P2"], kayitlar, gecersiz)
    assert oy["Mutlak çoğunluk"].tolist()[:3] == [1, 2, 0]
    assert oy["Durum"].tolist()[3:] == ["yetersiz oy — uzlaşma gerekli"] * 2  # tek oy referans sayılmaz
    assert oy["Geçersiz hücre"].iat[3] == "P1: 1 sonradan 2"


def test_lexicon_and_electra_label_mappings(monkeypatch):
    from tmlab import electra_duygu as el

    sw = {("güzel",): (0.8, 1), ("berbat",): (-0.9, -1)}
    tes = pd.DataFrame({"Kelime (Word)": ["güzel", "berbat", "kargo"],
                        "Pozitif/Negatif/Nötr\n(Sentiment)": ["Pozitif", "Negatif", "Nötr"]})
    etiket = {"çok güzel": "Positive", "berbat ürün": "Negative", "kargo geldi": "Neutral"}
    monkeypatch.setattr(el, "load_model", lambda: ("tok", "sess", {0: "Negative", 1: "Neutral", 2: "Positive"}))
    monkeypatch.setattr(el, "infer", lambda m, durum: {"Etiket": etiket[m]})
    r = idu.yontem_tahminleri(list(etiket), sw_sozluk=sw, tes_cerceve=tes)
    for ad in ("SWNetTR++", "TES-Lex", "ELECTRA"):
        assert r["sonuc"][ad]["tahmin"].tolist() == [1, 2, 0], ad
    assert r["sonuc"]["SWNetTR++"]["kapsam"].tolist() == [True, True, False]
    monkeypatch.setattr(el, "load_model", lambda: ("tok", "sess", {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))
    r = idu.yontem_tahminleri(["çok güzel"], sw_sozluk=sw, tes_cerceve=tes)
    assert "ELECTRA" not in r["sonuc"] and r["notlar"]["ELECTRA"].startswith(idu.NOT_RUN)
