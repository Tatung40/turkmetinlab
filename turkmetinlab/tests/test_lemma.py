"""İsteğe bağlı lemmatizasyon testleri (altın standart: UD Turkish-BOUN örneği)."""

from collections import Counter

import pytest

from tmlab.analysis import ngram_sikligi, sozcuk_sikligi, tfidf_hesapla
from tmlab.lemma import LEMMA_DURAK_SOZCUKLERI, OLCULEN_DOGRULUK, Lemmatizer, kullanilabilir_mi
from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle, on_isle, turkce_kucuk_harf

LEMMA = OnIslemeAyarlari(lemmatizasyon=True)


def test_kutuphane_kurulu():
    assert kullanilabilir_mi()


@pytest.mark.parametrize("sozcuk, lemma", [
    ("öğrencilerin", "öğrenci"), ("kitapları", "kitap"), ("eğitimde", "eğitim"), ("gelmişti", "gel"),
    ("düzeyleri", "düzey"),
])
def test_temel_lemmalar(sozcuk, lemma):
    assert Lemmatizer()(sozcuk) == lemma


def test_olumsuzluk_eki_korunur():
    # simplemma "anlamsız" için "anlam" döndürür; anlam tersine döneceği için koruma kuralı devreye girer
    import simplemma
    assert simplemma.lemmatize("anlamsız", lang="tr") == "anlam"
    assert Lemmatizer()("anlamsız") == "anlamsız"
    assert Lemmatizer()("anlamsızlık") in ("anlamsız", "anlamsızlık")


def test_belirlenimci_ve_onbellekli():
    l = Lemmatizer()
    assert l("çalışmalarında") == l("çalışmalarında") == Lemmatizer()("çalışmalarında")


def test_sayisal_ve_bos_girdi():
    l = Lemmatizer()
    assert l("2023") == "2023" and l("") == ""


def test_varsayilan_kapali():
    assert OnIslemeAyarlari().lemmatizasyon is False
    assert on_isle("öğrencilerin kitapları").sozcukler == ["öğrencilerin", "kitapları"]


def test_on_islemede_lemma_ve_kayit():
    kayit = Counter()
    b = derlem_on_isle(["Öğrencilerin kitapları. Öğrenci kitap okudu."], LEMMA, kayit)[0]
    assert b.sozcukler == ["öğrenci", "kitap", "öğrenci", "kitap", "oku"]
    assert kayit[("öğrencilerin", "öğrenci")] == 1 and kayit[("kitapları", "kitap")] == 1
    assert ("öğrenci", "öğrenci") not in kayit  # yalnızca değişenler kaydedilir


def test_lemmasi_durak_sozcuk_olan_cikarilir():
    assert "ol" in LEMMA_DURAK_SOZCUKLERI
    b = on_isle("Sonuçlar anlamlı olmuştur", LEMMA)
    assert "ol" not in b.sozcukler
    assert b.bolutler[0][-1] is None  # n-gram güvenliği için boşluk bırakılır


def test_korunan_kisaltma_lemmatize_edilmez():
    b = on_isle("COVID-19 salgınları", OnIslemeAyarlari(lemmatizasyon=True, kisaltmalari_koru=True))
    assert b.sozcukler[0] == "COVID-19"


def test_lemma_sikliklari_birlestirir():
    metinler = ["Uzaktan eğitimde sorunlar.", "Uzaktan eğitim sorunu.", "Uzaktan eğitimin sorunları."]
    f = sozcuk_sikligi(derlem_on_isle(metinler, LEMMA)).set_index("Sözcük")
    assert f.loc["eğitim", "Sıklık"] == 3 and f.loc["sorun", "Sıklık"] == 3
    n = ngram_sikligi(derlem_on_isle(metinler, LEMMA), 2).set_index("N-gram")
    # Bilinen ve arayüzde belirtilen sınırlılık: belirteç "uzaktan" da köke iner (uzak),
    # bu yüzden kalıp terim "uzaktan eğitim" lemma düzeyinde "uzak eğitim" olarak sayılır.
    assert n.loc["uzak eğitim", "Sıklık"] == 3
    assert tfidf_hesapla(derlem_on_isle(metinler, LEMMA)).derlem_tablosu["Terim"].tolist()


# --- Altın standarda karşı doğruluk ---------------------------------------

def _dogruluk(ud, tahmin):
    def norm(s):
        s = turkce_kucuk_harf(s).translate(str.maketrans("âîû", "aiu"))
        return s[:-3] if len(s) > 4 and s.endswith(("mek", "mak")) else s

    dogru = toplam = 0
    for c in ud:
        for t in c["sozcukler"]:
            if t["upos"] not in {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}:
                continue
            toplam += 1
            dogru += norm(tahmin(turkce_kucuk_harf(t["bicim"]))) == norm(t["lemma"])
    return dogru / toplam, toplam


def test_ud_boun_ornek_dogrulugu(ud):
    lem = Lemmatizer()
    dogruluk, n = _dogruluk(ud, lem)
    temel, _ = _dogruluk(ud, lambda w: w)
    assert n > 3000
    # Tam test kümesinde ölçülen değer %94,5; örnekte de %90'ın üzerinde kalmalı.
    assert dogruluk >= 0.90, dogruluk
    assert dogruluk - temel > 0.4  # lemmatizasyon anlamlı bir fark yaratıyor
    assert min(OLCULEN_DOGRULUK.values()) >= 0.94
