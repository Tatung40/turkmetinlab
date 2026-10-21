from tmlab.duygu_baglam import norm, frequent_evidence, evaluate
def test_unicode_and_frequency():
    assert norm("İKLİM") == "iklim"
    data = frequent_evidence(["İklim iyi", "iklim kötü", "iklim krizi"],
                              [{"Sözcük/ifade": "iklim"}])
    assert data.iloc[0]["Belge frekansı"] == 3
def test_warning_not_accuracy_claim():
    x = evaluate(["1"], electra=[{"Etiket": "Notr", "Kesildi mi": True,
                                   "En yüksek model skoru": 0.4}])
    assert x.loc[0, "İnceleme gerekli"]
    assert "64 token" in x.loc[0, "Gerekçeler"]
def test_partial_and_empty():
    assert evaluate(["x", "x"])["Belge sırası"].tolist() == [1, 2]
    assert frequent_evidence([], []).empty
