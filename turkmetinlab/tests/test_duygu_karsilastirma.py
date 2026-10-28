"""Sentiment method aggregation avoids duplicate-name cross-joins."""
import pytest
from tmlab.duygu_karsilastirma import birlestir, ozetsayim

def test_duplicate_names_are_kept_separate():
    sw=[{"Ton ortalaması":.1,"Kapsama (%)":20},{"Ton ortalaması":-.5,"Kapsama (%)":30}]
    te=[{"Neşe":1,"Kapsama (%)":10},{"Korku":2,"Kapsama (%)":20}]
    el=[{"Etiket":"Positive","En yüksek model skoru":.9,"Kesildi mi":False},
        {"Etiket":"Negative","En yüksek model skoru":.8,"Kesildi mi":True}]
    df=birlestir(["A","A"],sw,te,el)
    assert df["Belge sırası"].tolist()==[1,2]
    assert df["SWNetTR++ ton ort."].tolist()==[.1,-.5]
    assert __import__("pandas").isna(df.loc[0,"TES-Lex Korku"])
    assert df.loc[1,"TES-Lex Korku"]==2
    assert ozetsayim(df)["Frekans"].sum()==2

def test_partial_results_allowed():
    df=birlestir(["x"],electra=[{"Etiket":"Notr"}])
    assert "SWNetTR++ ton ort." not in df.columns
    assert df.iloc[0]["ELECTRA sınıfı"]=="Notr"

def test_invalid_lengths_rejected():
    with pytest.raises(ValueError):
        birlestir(["x"],sw=[])
