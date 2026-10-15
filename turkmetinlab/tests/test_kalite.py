"""Veri kalitesi denetimi: boş/yer tutucu değerler, yinelenen kimlik ve metinler."""

import pandas as pd
import pytest

from tmlab.kalite import belgeleri_hazirla, tekil_adlar, yer_tutucu_mu


@pytest.mark.parametrize("deger", ["NA", "n/a", "NaN", "null", "None", "-", "—", "?", "#YOK", "#N/A",
                                   "#DEĞER!", "#VALUE!", "#BAŞV!", "#SAYI/0!", "  na  "])
def test_yer_tutucular(deger):
    assert yer_tutucu_mu(deger)


@pytest.mark.parametrize("deger", ["Yok", "Hayır", "Na'im", "Nane", "a", "Bilmiyorum"])
def test_gercek_metinler_yer_tutucu_degil(deger):
    assert not yer_tutucu_mu(deger)


def _tablo():
    return pd.DataFrame({
        "kimlik": ["A1", "A2", "A1", "A3", "A4", "A5", None],
        "metin": ["Birinci metin", None, "İkinci metin", "#YOK", "   ", "birinci   METİN", "Üçüncü metin"],
    })


def test_rapor_ve_satir_numaralari():
    metinler, adlar, nolar, rapor = belgeleri_hazirla(_tablo(), "metin", "kimlik")
    assert metinler == ("Birinci metin", "İkinci metin", "birinci   METİN", "Üçüncü metin")
    assert nolar == [1, 3, 6, 7]
    assert rapor.bos_satirlar == [2, 5]
    assert rapor.yer_tutucu_satirlar == [4]
    assert rapor.yinelenen_metin_satirlari == [6]  # boşluk ve büyük/küçük harf farkı yok sayılır
    assert rapor.yinelenen_adlar == ["A1"]
    assert adlar == ["A1", "A1 (2)", "A5", "Adsız"]
    assert rapor.analiz_edilen == 4
    u = " ".join(rapor.uyarilar())
    assert "2 satırın metin hücresi boş" in u and "satır: 2, 5" in u
    assert "yer tutucu" in u and "satır: 4" in u
    assert "A1" in u and "aynı metni" in u


def test_yinelenen_metinleri_cikar():
    metinler, adlar, nolar, rapor = belgeleri_hazirla(_tablo(), "metin", None, yinelenen_metinleri_cikar=True)
    assert nolar == [1, 3, 7]
    assert adlar == ["Satır 1", "Satır 3", "Satır 7"]
    assert rapor.analiz_edilen == 3
    assert "tercihiniz üzerine" in " ".join(rapor.uyarilar())


def test_temiz_veride_uyari_yok():
    t = pd.DataFrame({"m": ["a b", "c d"]})
    _, _, _, rapor = belgeleri_hazirla(t, "m")
    assert rapor.uyarilar() == []


def test_tekil_adlar_cakisma():
    adlar, tekrar = tekil_adlar(["X", "X", "X (2)", ""])
    assert len(set(adlar)) == 4
    assert adlar[0] == "X" and adlar[2] == "X (2)" and adlar[3] == "Adsız"
    assert tekrar == ["X"]


def test_sayisal_kimlik_ve_metin():
    t = pd.DataFrame({"no": [1, 2], "m": [123, "metin"]})
    metinler, adlar, _, _ = belgeleri_hazirla(t, "m", "no")
    assert metinler == ("123", "metin") and adlar == ["1", "2"]
