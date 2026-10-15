"""Analiz öncesi veri kalitesi denetimi.

Boş hücreler, yer tutucu değerler (NA, null, #YOK…), yinelenen belge
kimlikleri ve yinelenen metinler tespit edilir ve kullanıcıya satır numarasıyla
bildirilir. Hiçbir satır sessizce atılmaz.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from .preprocess import turkce_kucuk_harf

# Metin değil, "veri yok" anlamına gelen değerler ve Excel hata değerleri (Türkçe ve İngilizce)
YER_TUTUCULAR = frozenset({
    "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "—", "–", "?", ".",
    "#n/a", "#yok", "#değer!", "#value!", "#başv!", "#ref!", "#ad?", "#name?", "#sayı/0!", "#div/0!",
    "#sayı!", "#num!", "#boş!", "#null!", "#spill!", "#taşma!", "#calc!", "#hesap!",
})


# Türkçede de sözcük olan kısa biçimler (on, an, as, in) bilerek dışarıda bırakıldı
_INGILIZCE_ISARET = frozenset("the of and to is for this that with are was were be by from".split())
_SOZCUK_RE = re.compile(r"[^\W\d_]+")


def ingilizce_oran(metin: str) -> float:
    """Metindeki sözcüklerin ne kadarının çok sık İngilizce işlev sözcüğü olduğu (Türkçe metinde ~0)."""
    sozcukler = [turkce_kucuk_harf(w) for w in _SOZCUK_RE.findall(str(metin))]
    return sum(w in _INGILIZCE_ISARET for w in sozcukler) / len(sozcukler) if sozcukler else 0.0


def ingilizce_iceren(metinler, esik: float = 0.08) -> List[int]:
    """Belirgin miktarda İngilizce metin içeren belgelerin sıra indisleri.

    Eşik: Türkçe metinlerde bu 15 İngilizce işlev sözcüğünün oranı neredeyse 0'dır; İngilizce
    metinlerde ~%20-25'tir. %8, yarısı İngilizce olan bir belgeyi yakalar.
    """
    return [i for i, m in enumerate(metinler) if ingilizce_oran(m) >= esik]


def yer_tutucu_mu(deger) -> bool:
    return turkce_kucuk_harf(str(deger).strip()) in YER_TUTUCULAR


def _ornekle(nolar: List[int], en_fazla: int = 10) -> str:
    return ", ".join(map(str, nolar[:en_fazla])) + (" …" if len(nolar) > en_fazla else "")


@dataclass
class DenetimRaporu:
    toplam_satir: int = 0
    bos_satirlar: List[int] = field(default_factory=list)
    yer_tutucu_satirlar: List[int] = field(default_factory=list)
    yinelenen_adlar: List[str] = field(default_factory=list)
    yinelenen_metin_satirlari: List[int] = field(default_factory=list)
    yinelenen_metinler_cikarildi: bool = False

    @property
    def analiz_edilen(self) -> int:
        n = self.toplam_satir - len(self.bos_satirlar) - len(self.yer_tutucu_satirlar)
        if self.yinelenen_metinler_cikarildi:
            n -= len(self.yinelenen_metin_satirlari)
        return n

    def uyarilar(self) -> List[str]:
        u = []
        if self.bos_satirlar:
            u.append(f"{len(self.bos_satirlar)} satırın metin hücresi boş olduğu için analize alınmadı "
                     f"(satır: {_ornekle(self.bos_satirlar)}).")
        if self.yer_tutucu_satirlar:
            u.append(f"{len(self.yer_tutucu_satirlar)} satırda metin yerine 'NA', 'null', '#YOK' gibi bir yer tutucu "
                     f"ya da Excel hata değeri vardı; analize alınmadı (satır: {_ornekle(self.yer_tutucu_satirlar)}).")
        if self.yinelenen_adlar:
            u.append(f"Belge adı sütununda {len(self.yinelenen_adlar)} değer birden fazla kez geçiyor "
                     f"(ör. {', '.join(self.yinelenen_adlar[:5])}). Ayırt edilebilmeleri için adlarına (2), (3)… eklendi.")
        if self.yinelenen_metin_satirlari:
            n = len(self.yinelenen_metin_satirlari)
            if self.yinelenen_metinler_cikarildi:
                u.append(f"Daha önceki bir satırla aynı metni içeren {n} satır, tercihiniz üzerine analize alınmadı "
                         f"(satır: {_ornekle(self.yinelenen_metin_satirlari)}).")
            else:
                u.append(f"{n} satır, daha önceki bir satırla aynı metni içeriyor "
                         f"(satır: {_ornekle(self.yinelenen_metin_satirlari)}). Bu metinler sıklıkları artırır; "
                         "istemiyorsanız sol menüden 'Yinelenen metinleri bir kez say' seçeneğini açın.")
        return u


def tekil_adlar(adlar) -> Tuple[List[str], List[str]]:
    """Aynı ada sahip belgeleri ayırt etmek için (2), (3)… ekler. (adlar, yinelenen_degerler)"""
    sayac, sonuc = {}, []
    adlar = [str(a).strip() or "Adsız" for a in adlar]
    tekrarlar = [a for a, n in Counter(adlar).items() if n > 1]
    kullanilan = set(adlar)
    for a in adlar:
        if a in sayac:
            sayac[a] += 1
            aday = f"{a} ({sayac[a]})"
            while aday in kullanilan:  # "X (2)" zaten veri içinde varsa
                sayac[a] += 1
                aday = f"{a} ({sayac[a]})"
            kullanilan.add(aday)
            sonuc.append(aday)
        else:
            sayac[a] = 1
            sonuc.append(a)
    return sonuc, tekrarlar


def _metin_anahtari(m: str) -> str:
    return re.sub(r"\s+", " ", turkce_kucuk_harf(m)).strip()


def belgeleri_hazirla(tablo: pd.DataFrame, metin_sutunu: str, ad_sutunu: Optional[str] = None,
                      yinelenen_metinleri_cikar: bool = False):
    """Tablodan analiz edilecek metinleri, belge adlarını ve denetim raporunu üretir.

    Satır numaraları, arayüzdeki 'Veri' sekmesinde gösterilen numaralarla aynıdır (1'den başlar).
    Döndürür: (metinler: tuple, adlar: list, satir_nolari: list, rapor: DenetimRaporu)
    """
    rapor = DenetimRaporu(toplam_satir=len(tablo), yinelenen_metinler_cikarildi=yinelenen_metinleri_cikar)
    metinler, ham_adlar, nolar = [], [], []
    gorulen = set()
    sutun = tablo[metin_sutunu]
    adlar_sutun = tablo[ad_sutunu] if ad_sutunu else None
    for konum, (indeks, deger) in enumerate(sutun.items()):
        no = int(indeks) + 1
        if deger is None or (isinstance(deger, float) and deger != deger) or not str(deger).strip():
            rapor.bos_satirlar.append(no)
            continue
        metin = str(deger)
        if yer_tutucu_mu(metin):
            rapor.yer_tutucu_satirlar.append(no)
            continue
        anahtar = _metin_anahtari(metin)
        if anahtar in gorulen:
            rapor.yinelenen_metin_satirlari.append(no)
            if yinelenen_metinleri_cikar:
                continue
        gorulen.add(anahtar)
        metinler.append(metin)
        nolar.append(no)
        if adlar_sutun is not None:
            a = adlar_sutun.iat[konum]
            ham_adlar.append("" if a is None or (isinstance(a, float) and a != a) else str(a))
    if adlar_sutun is not None:
        adlar, rapor.yinelenen_adlar = tekil_adlar(ham_adlar)
    else:
        adlar = [f"Satır {n}" for n in nolar]
    return tuple(metinler), adlar, nolar, rapor
