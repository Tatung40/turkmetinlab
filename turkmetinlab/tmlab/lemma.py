"""İsteğe bağlı Türkçe lemmatizasyon (simplemma, sözlük tabanlı).

Yöntem seçimi ölçüme dayanır (ayrıntı: docs/lemmatizasyon_degerlendirmesi.md).
UD Turkish-BOUN ve UD Turkish-IMST test kümelerinde (altın standart lemmalar)
ölçülen doğruluklar:

    Yöntem                          BOUN     IMST    Hız (ms/sözcük)
    İşlem yok (sözcüğün kendisi)    %45,8    %43,0   —
    Snowball kök bulucu (kural)     %57,0    %54,3   0,05
    zeyrek (bağlamsız)              %83,6    —       1,1
    Zemberek-python (bağlamlı)      %92,3    %93,3   4,3
    simplemma (küçük harf girdi)    %94,5    %94,4   0,08

Bu modül simplemma'yı kullanır. Sınırlılıklar:
* Bağlamdan bağımsızdır: eş yazımlı sözcüklerde (ör. "yüz") tek bir lemma seçer.
* UD kuralına uyar; eylem adları fiil köküne iner: çalışma → çalış,
  araştırma → araştır, öğrenme → öğren.
* Bazı türetim eklerini fazla soyar (politikacı → politika).
* Olumsuzluk eki -sız/-siz/-suz/-süz atılırsa anlam tersine döner
  (anlamsız → anlam). Bunu önlemek için dar kapsamlı bir koruma kuralı
  uygulanır; kural altın veride doğruluğu düşürmemiş, BOUN'da bu sınıftaki
  hataların çoğunu düzeltmiştir.
"""

from __future__ import annotations

from typing import Dict

YONTEM = "simplemma 2.0 (sözlük tabanlı, bağlamdan bağımsız)"
OLCULEN_DOGRULUK = {"UD Turkish-BOUN test": 0.9447, "UD Turkish-IMST test": 0.9440}

_OLUMSUZLUK_EKLERI = ("sız", "siz", "suz", "süz")

# Lemmatizasyondan sonra ortaya çıkan, tek başına anlam taşımayan yardımcı fiil
# kökleri (yüzey biçimleri zaten durak sözcük listesindedir: olan, olarak, etmek…)
LEMMA_DURAK_SOZCUKLERI = frozenset({"ol", "et", "i", "de", "yap"})


def kullanilabilir_mi() -> bool:
    try:
        import simplemma  # noqa: F401
        return True
    except ImportError:  # pragma: no cover
        return False


class Lemmatizer:
    """Sözcük türü (type) düzeyinde önbellekli lemmatizer.

    Aynı sözcük her zaman aynı lemmayı alır (belirlenimci). Girdi, Türkçe
    kurallarıyla küçük harfe çevrilmiş sözcük olmalıdır.
    """

    def __init__(self):
        import simplemma

        self._lemmatize = simplemma.lemmatize
        self._onbellek: Dict[str, str] = {}

    def __call__(self, sozcuk: str) -> str:
        sonuc = self._onbellek.get(sozcuk)
        if sonuc is None:
            sonuc = self._hesapla(sozcuk)
            self._onbellek[sozcuk] = sonuc
        return sonuc

    def _hesapla(self, sozcuk: str) -> str:
        if not sozcuk or not any(c.isalpha() for c in sozcuk):
            return sozcuk
        try:
            lemma = self._lemmatize(sozcuk, lang="tr")
        except Exception:  # sözlükte beklenmeyen bir durum: sözcüğü olduğu gibi bırak
            return sozcuk
        if not lemma:
            return sozcuk
        from .preprocess import turkce_kucuk_harf  # döngüsel içe aktarmayı önlemek için burada

        lemma = turkce_kucuk_harf(lemma)
        # Olumsuzluk eki koruması: anlamsız → anlam yerine anlamsız
        if lemma != sozcuk:
            for ek in _OLUMSUZLUK_EKLERI:
                if sozcuk.startswith(lemma + ek):
                    return lemma + ek
        return lemma
