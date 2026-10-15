"""N-gram bölütleme kuralları ve dilbilimsel gerekçesi."""

import pytest

from tmlab.analysis import belge_ngramlari, ngram_sikligi
from tmlab.preprocess import OnIslemeAyarlari, bolutle, derlem_on_isle, on_isle


def bolutler(metin, **kw):
    kw.setdefault("durak_sozcuk_genel", False)
    kw.setdefault("min_uzunluk", 1)
    return on_isle(metin, OnIslemeAyarlari(**kw)).bolutler


METIN = "Eğitim politikası, okul yönetimi; öğretmen yetiştirme: kurum (köy) okulları - yerel yönetim. Yeni konu"


def test_kati_virgul_ve_noktali_virgul_sinirdir():
    assert bolutler(METIN, bolutleme="kati") == [
        ["eğitim", "politikası"], ["okul", "yönetimi"], ["öğretmen", "yetiştirme"], ["kurum"], ["köy"],
        ["okulları"], ["yerel", "yönetim"], ["yeni", "konu"]]


def test_orta_virgul_sinir_degildir():
    assert bolutler(METIN, bolutleme="orta")[0] == ["eğitim", "politikası", "okul", "yönetimi"]
    assert ["öğretmen", "yetiştirme"] in bolutler(METIN, bolutleme="orta")


def test_cumle_yalnizca_cumle_sonu():
    b = bolutler(METIN, bolutleme="cumle")
    assert len(b) == 2
    assert b[1] == ["yeni", "konu"]


def test_virgulu_asan_ngram_secime_gore():
    belge = derlem_on_isle(["ekonomi, sağlık"], OnIslemeAyarlari(bolutleme="kati"))[0]
    assert belge_ngramlari(belge, 2) == []
    belge = derlem_on_isle(["ekonomi, sağlık"], OnIslemeAyarlari(bolutleme="orta"))[0]
    assert belge_ngramlari(belge, 2) == ["ekonomi sağlık"]


@pytest.mark.parametrize("metin, beklenen", [
    ("19. yüzyıl Osmanlı eğitimi", [["19", "yüzyıl", "osmanlı", "eğitimi"]]),
    ("Prof. Dr. Ayşe Yılmaz konuştu", [["prof", "dr", "ayşe", "yılmaz", "konuştu"]]),
    ("Sınav 2023. Bu yıl", [["sınav", "2023"], ["bu", "yıl"]]),          # büyük harf: cümle sonu
    ("oran 3,5 ve 1.250 kişi", [["oran", "3,5", "ve", "1.250", "kişi"]]),  # ondalık ayırıcılar bölmez
    ("saat 10:30 başladı", [["saat", "10:30", "başladı"]]),
])
def test_sira_sayisi_kisaltma_ve_sayilar(metin, beklenen):
    assert bolutler(metin, sayilari_kaldir=False) == beklenen


def test_bitisik_tire_sozcuk_bosluklu_tire_sinir():
    assert bolutler("sosyo-ekonomik düzey") == [["sosyo-ekonomik", "düzey"]]
    assert bolutler("ekonomik düzey - yerel yönetim") == [["ekonomik", "düzey"], ["yerel", "yönetim"]]


def test_satir_sonu_secenegi():
    metin = "uzaktan\neğitim süreci\n\nyeni paragraf"
    assert bolutler(metin) == [["uzaktan", "eğitim", "süreci"], ["yeni", "paragraf"]]
    assert bolutler(metin, satir_sonu_sinir=True) == [["uzaktan"], ["eğitim", "süreci"], ["yeni", "paragraf"]]


def test_paragraf_her_zaman_sinir():
    for mod in ("kati", "orta", "cumle"):
        assert len(bolutler("birinci kısım\n\nikinci kısım", bolutleme=mod)) == 2


def test_url_kaldirildiginda_sinir_kalir():
    # Web adresi çıkarılınca iki yanındaki sözcükler yapay ikili oluşturmamalı
    assert bolutler("kaynak https://ornek.org/x makale") == [["kaynak"], ["makale"]]


def test_crlf_satir_sonlari():
    assert bolutler("bir\r\n\r\niki") == [["bir"], ["iki"]]


def test_gecersiz_bolutleme():
    with pytest.raises(ValueError):
        OnIslemeAyarlari(bolutleme="yok")


def test_bolutle_dogrudan():
    assert bolutle("a, b. c", "kati") == ["a", " b", " c"]
    assert bolutle("a, b. c", "cumle") == ["a, b", " c"]


def test_ngram_sayimlari_bolutlemeye_gore():
    metinler = ["Ekonomi, sağlık ve eğitim. Ekonomi, sağlık politikası."]
    kati = ngram_sikligi(derlem_on_isle(metinler, OnIslemeAyarlari(bolutleme="kati")), 2)
    orta = ngram_sikligi(derlem_on_isle(metinler, OnIslemeAyarlari(bolutleme="orta")), 2)
    assert "ekonomi sağlık" not in set(kati["N-gram"])
    assert orta.set_index("N-gram").loc["ekonomi sağlık", "Sıklık"] == 2


# --- Dilbilimsel gerekçe: UD Turkish-BOUN ağaç bankasında ölçüm -------------

def _bitisik_iliski_oranlari(ud):
    """Aralarında hiç / tek noktalama bulunan ardışık sözcük çiftlerinin öbek içi bağlılık oranı."""
    from collections import Counter

    sayim = {"yok": Counter(), ",": Counter(), ";": Counter()}
    for c in ud:
        s = c["sozcukler"]
        i = 0
        while i < len(s) - 1:
            if s[i]["upos"] == "PUNCT":
                i += 1
                continue
            j, arada = i + 1, []
            while j < len(s) and s[j]["upos"] == "PUNCT":
                arada.append(s[j]["bicim"]); j += 1
            if j >= len(s):
                break
            tur = "yok" if not arada else (arada[0] if len(arada) == 1 and arada[0] in (",", ";") else None)
            if tur:
                a, b = s[i], s[j]
                rel = a["iliski"] if a["bas"] == b["no"] else (b["iliski"] if b["bas"] == a["no"] else None)
                ic = rel is not None and rel.split(":")[0] not in ("conj", "parataxis", "cc", "root")
                sayim[tur]["öbek içi" if ic else "değil"] += 1
            i = j
    return {k: v["öbek içi"] / max(1, sum(v.values())) for k, v in sayim.items()}, sayim


def test_virgul_siniri_ud_verisiyle_gerekcelendirilir(ud):
    oran, sayim = _bitisik_iliski_oranlari(ud)
    assert sum(sayim["yok"].values()) > 2000 and sum(sayim[","].values()) > 200
    # Noktalama yokken yan yana sözcüklerin büyük kısmı aynı öbekte...
    assert oran["yok"] > 0.5
    # ...virgülün iki yanındakilerin ise çok azı. Virgülü aşan n-gramlar çoğunlukla yapaydır.
    assert oran[","] < 0.10
    assert oran["yok"] / max(oran[","], 1e-9) > 5
