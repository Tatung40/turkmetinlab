"""Türkçe metin ön işleme.

Bilimsel tasarım kararları
--------------------------
1. Türkçe büyük/küçük harf: Python'un str.lower() işlevi Türkçede hatalıdır
   ("I" -> "i", "İ" -> "i̇"). Burada "I" -> "ı" ve "İ" -> "i" dönüşümü
   açıkça yapılır.
2. Unicode NFKC normalleştirmesi: PDF'ten kopyalanan metinlerdeki bitişik
   harfler (ﬁ), ayrık yazılmış aksanlar ve tam genişlikte karakterler
   standart biçime getirilir. Yumuşak tire (U+00AD) silinir.
3. Kesme işareti: "Türkiye'nin" gibi özel ad + ek yapıları kullanıcı tercihine
   göre "türkiye" (ek atılır), "türkiyenin" (birleştirilir) veya
   "türkiye" + "nin" (ayrılır) biçiminde işlenir.
4. Bölütleme: Metin; nokta, soru/ünlem işareti, iki nokta, noktalı virgül,
   virgül, parantez, tırnak ve satır sonlarında bölütlere ayrılır. N-gramlar
   bölüt sınırlarını aşmaz.
5. Çıkarılan sözcükler (durak sözcük, sayı, çok kısa sözcük) bölüt içinde
   "boşluk" (None) olarak bırakılır. Böylece n-gram hesaplanırken, aralarında
   çıkarılmış bir sözcük bulunan iki sözcük yapay olarak yan yana
   getirilmez. Örneğin "eğitim ve öğretim" ifadesinden "eğitim öğretim"
   ikilisi üretilmez.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional

from .stopwords import stopword_set

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

KESME_SECENEKLERI = {
    "ek_at": "Eki at (Türkiye'nin → türkiye)",
    "birlestir": "Birleştir (Türkiye'nin → türkiyenin)",
    "ayir": "Ayrı sözcük say (Türkiye'nin → türkiye + nin)",
}


@dataclass(frozen=True)
class OnIslemeAyarlari:
    kucuk_harf: bool = True
    kisaltmalari_koru: bool = False
    url_eposta_kaldir: bool = True
    sayilari_kaldir: bool = True
    kesme_isareti: str = "ek_at"
    sapkali_harf_normallestir: bool = False
    tireli_birlesik_koru: bool = True
    durak_sozcuk_genel: bool = True
    durak_sozcuk_akademik: bool = False
    ozel_durak_sozcukler: tuple = field(default_factory=tuple)
    min_uzunluk: int = 2

    def __post_init__(self):
        if self.kesme_isareti not in KESME_SECENEKLERI:
            raise ValueError(f"Geçersiz kesme işareti seçeneği: {self.kesme_isareti}")
        if self.min_uzunluk < 1:
            raise ValueError("En kısa sözcük uzunluğu en az 1 olmalıdır.")

    def sozluk(self) -> dict:
        d = asdict(self)
        d["ozel_durak_sozcukler"] = ", ".join(self.ozel_durak_sozcukler)
        return d


# ---------------------------------------------------------------------------
# Temel karakter işlemleri
# ---------------------------------------------------------------------------

_KESME_VARYANTLARI = "’‘ʼ`´′ʹ"
_TIRE_VARYANTLARI = "‐‑‒–—−"
_SAPKALI = str.maketrans({"â": "a", "î": "i", "û": "u", "Â": "A", "Î": "İ", "Û": "U"})


def turkce_kucuk_harf(metin: str) -> str:
    """Türkçe kurallarına uygun küçük harfe çevirme.

    >>> turkce_kucuk_harf("IŞIK İSTANBUL")
    'ışık istanbul'
    """
    metin = unicodedata.normalize("NFC", metin)
    metin = metin.replace("İ", "i").replace("I", "ı")
    metin = metin.lower()
    # Ayrık yazılmış "i̇" (i + U+0307 birleşen nokta) durumunu temizle
    return metin.replace("i̇", "i")


def unicode_normallestir(metin: str) -> str:
    """NFKC normalleştirmesi ve tipografik işaretlerin sadeleştirilmesi."""
    metin = unicodedata.normalize("NFKC", metin)
    metin = metin.replace("­", "")  # yumuşak tire (PDF kopyalarında sık)
    metin = metin.replace("​", "").replace("﻿", "")  # sıfır genişlikli boşluk, BOM
    for k in _KESME_VARYANTLARI:
        metin = metin.replace(k, "'")
    for t in _TIRE_VARYANTLARI:
        metin = metin.replace(t, "-")
    # Satır sonunda tire ile bölünmüş sözcükleri birleştir: "eği-\ntim" -> "eğitim"
    metin = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", metin)
    return metin


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EPOSTA_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_DOI_RE = re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+", re.IGNORECASE)

# Bölüt sınırları: sayıların içindeki nokta/virgül (3,5 ; 1.250) bölmez.
_BOLUT_RE = re.compile(r"(?<!\d)[.,](?!\d)|[.,](?=\s)|[!?;:…()\[\]{}\"«»“”„\n\r\t|/]")

_SAYI_RE = re.compile(r"[\d.,\-]+")


def _token_deseni(tireli: bool) -> re.Pattern:
    sozcuk = r"[^\W_]+(?:'[^\W_]+)?"
    if tireli:
        sozcuk = r"[^\W_]+(?:-[^\W_]+)*(?:'[^\W_]+)?"
    return re.compile(r"\d+(?:[.,]\d+)+|" + sozcuk)


_TOKEN_TIRELI = _token_deseni(True)
_TOKEN_TIRESIZ = _token_deseni(False)


def ozel_durak_sozcukleri_ayristir(metin: str, sapkali_normallestir: bool = False) -> tuple:
    """Kullanıcının virgül, boşluk veya satır sonu ile girdiği sözcükleri ayırır."""
    if not metin:
        return tuple()
    parcalar = re.split(r"[,;\s]+", unicode_normallestir(metin))
    sonuc = []
    for p in parcalar:
        p = turkce_kucuk_harf(p.strip().strip("'"))
        if sapkali_normallestir:
            p = p.translate(_SAPKALI)
        if p:
            sonuc.append(p)
    return tuple(sorted(set(sonuc)))


# ---------------------------------------------------------------------------
# Ana ön işleme
# ---------------------------------------------------------------------------

Bolut = List[Optional[str]]


@dataclass
class IslenmisBelge:
    """Bir belgenin ön işlenmiş hali.

    bolutler: Her bölüt bir sözcük listesidir; çıkarılan sözcükler None'dır.
    """

    bolutler: List[Bolut]

    @property
    def sozcukler(self) -> List[str]:
        return [t for b in self.bolutler for t in b if t is not None]

    @property
    def ham_sozcuk_sayisi(self) -> int:
        return sum(len(b) for b in self.bolutler)


def _kisaltma_mi(tok: str) -> bool:
    """Tamamı büyük harfli, en az iki harfli kısaltma mı? (COVID-19, TÜİK, AB)"""
    harfler = [c for c in tok if c.isalpha()]
    return len(harfler) >= 2 and all(c.isupper() for c in harfler)


def _kesme_isle(token: str, secenek: str) -> List[str]:
    if "'" not in token:
        return [token]
    kok, _, ek = token.partition("'")
    if secenek == "ek_at":
        return [kok] if kok else []
    if secenek == "birlestir":
        return [kok + ek]
    return [p for p in (kok, ek) if p]  # ayir


def on_isle(metin, ayarlar: OnIslemeAyarlari = OnIslemeAyarlari(),
            durak_kumesi: Optional[frozenset] = None) -> IslenmisBelge:
    """Tek bir metni ön işler."""
    if metin is None or (isinstance(metin, float) and metin != metin):  # NaN
        return IslenmisBelge([])
    metin = unicode_normallestir(str(metin))

    if ayarlar.url_eposta_kaldir:
        metin = _URL_RE.sub(" \n ", metin)
        metin = _DOI_RE.sub(" \n ", metin)
        metin = _EPOSTA_RE.sub(" \n ", metin)

    if ayarlar.sapkali_harf_normallestir:
        metin = metin.translate(_SAPKALI)

    if durak_kumesi is None:
        durak_kumesi = stopword_set(ayarlar.durak_sozcuk_genel, ayarlar.durak_sozcuk_akademik,
                                    ayarlar.ozel_durak_sozcukler)
    if ayarlar.sapkali_harf_normallestir:
        durak_kumesi = frozenset(w.translate(_SAPKALI) for w in durak_kumesi)

    desen = _TOKEN_TIRELI if ayarlar.tireli_birlesik_koru else _TOKEN_TIRESIZ
    bolutler: List[Bolut] = []
    for parca in _BOLUT_RE.split(metin):
        if not parca or not parca.strip():
            continue
        bolut: Bolut = []
        for ham in desen.findall(parca):
            for tok in _kesme_isle(ham, ayarlar.kesme_isareti):
                tok = tok.strip("-")
                if not tok:
                    continue
                if _SAYI_RE.fullmatch(tok):
                    bolut.append(None if ayarlar.sayilari_kaldir else tok)
                    continue
                anahtar = turkce_kucuk_harf(tok)
                if ayarlar.kucuk_harf and not (
                    ayarlar.kisaltmalari_koru and _kisaltma_mi(tok)
                ):
                    tok = anahtar
                if anahtar in durak_kumesi or len(tok) < ayarlar.min_uzunluk:
                    bolut.append(None)
                else:
                    bolut.append(tok)
        if bolut:
            bolutler.append(bolut)
    return IslenmisBelge(bolutler)


def derlem_on_isle(metinler: Iterable, ayarlar: OnIslemeAyarlari = OnIslemeAyarlari()) -> List[IslenmisBelge]:
    """Bir metin listesini (derlemi) ön işler."""
    durak = stopword_set(ayarlar.durak_sozcuk_genel, ayarlar.durak_sozcuk_akademik,
                         ayarlar.ozel_durak_sozcukler)
    return [on_isle(m, ayarlar, durak) for m in metinler]
