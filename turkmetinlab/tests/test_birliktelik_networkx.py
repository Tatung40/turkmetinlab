"""Frozen pre-refactor groupby oracle for dugum_tablosu; no corpus/model/network downloads."""
import numpy as np
import pandas as pd
import pytest

from tmlab.analysis import turkce_sirala_anahtari
from tmlab.birliktelik import AgAyarlari, ag_sec, birliktelik_hesapla, dugum_tablosu
from tmlab.preprocess import IslenmisBelge


def old_nodes(secili, sonuc):
    uzun = pd.concat([secili[["Kaynak", "Ağırlık", "Birlikte görülme"]].rename(columns={"Kaynak": "Sözcük"}),
                      secili[["Hedef", "Ağırlık", "Birlikte görülme"]].rename(columns={"Hedef": "Sözcük"})])
    g = uzun.groupby("Sözcük").agg(Derece=("Ağırlık", "size"), Agirlikli=("Ağırlık", "sum"),
                                     Toplam=("Birlikte görülme", "sum")).reset_index()
    siklik = dict(zip(sonuc.sozcukler, sonuc.siklik))
    t = pd.DataFrame({
        "Sözcük": g["Sözcük"],
        "Sıklık": g["Sözcük"].map(siklik).astype(int),
        "Derece": g["Derece"].astype(int),
        "Normalleştirilmiş derece": np.round(g["Derece"] / max(len(g) - 1, 1), 6),
        "Ağırlıklı derece": np.round(g["Agirlikli"], 6),
        "Toplam birlikte görülme": g["Toplam"].astype(int),
    })
    t["_k"] = t["Sözcük"].map(turkce_sirala_anahtari)
    t = t.sort_values(["Ağırlıklı derece", "Derece", "_k"], ascending=[False, False, True], kind="stable")
    return t.drop(columns="_k").reset_index(drop=True)


BELGELER = [
    IslenmisBelge([["öğretmen", "öğrenci", "ışık", "İstanbul"], ["çığ", "şeker", "öğretmen"]]),
    IslenmisBelge([["öğrenci", "öğretmen", None, "ışık"], ["İstanbul", "ılık", "çığ"]]),
    IslenmisBelge([["çelik", "korozyon"], ["çelik", "ılık", "korozyon", "öğretmen"]]),
    IslenmisBelge([["şeker", "çığ", "şeker", "öğrenci"]]),
    IslenmisBelge([["yalnız"]]),
]


@pytest.mark.parametrize("baglam", ["pencere", "bolut", "belge"])
@pytest.mark.parametrize("olcut", ["jaccard", "ppmi", "ham"])
@pytest.mark.parametrize("dugum,kenar", [(2, 1), (5, 4), (50, 150)])
def test_networkx_degree_matches_old_groupby(baglam, olcut, dugum, kenar):
    r = birliktelik_hesapla(BELGELER, AgAyarlari(aday_sozcuk=50, min_birlikte=1, baglam=baglam, pencere=2))
    sec = ag_sec(r, olcut, dugum, kenar)
    pd.testing.assert_frame_equal(dugum_tablosu(sec, r), old_nodes(sec, r), check_dtype=False,
                                  check_exact=False, atol=1e-12, rtol=0)


def test_empty_selection_keeps_schema():
    r = birliktelik_hesapla(BELGELER, AgAyarlari(aday_sozcuk=50, min_birlikte=10_000))
    sec = ag_sec(r)
    assert list(dugum_tablosu(sec, r).columns) == list(old_nodes(sec, r).columns) and dugum_tablosu(sec, r).empty


@pytest.mark.parametrize("kaynak,hedef", [(["a", "b"], ["b", "a"]), (["a"], ["a"])])
def test_duplicate_or_self_loop_edges_are_rejected(kaynak, hedef):
    r = birliktelik_hesapla(BELGELER, AgAyarlari(aday_sozcuk=50, min_birlikte=1))
    sec = pd.DataFrame({"Kaynak": kaynak, "Hedef": hedef, "Ağırlık": 1.0, "Birlikte görülme": 1})
    with pytest.raises(ValueError, match="öz-ilmek"):
        dugum_tablosu(sec, r)
