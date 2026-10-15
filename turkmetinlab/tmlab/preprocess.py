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
4. Bölütleme (n-gramlar bölüt sınırını aşmaz). Üç düzey vardır:
   * "kati" (varsayılan): cümle sonu + ; : , + parantez, tırnak, eğik çizgi ve
     boşluklu tire. Gerekçe: UD Turkish (BOUN+IMST) ağaç bankalarında
     noktalamasız ardışık sözcük çiftlerinin %60,2'si aynı sözdizim öbeğinde
     doğrudan bağlıyken bu oran virgülün iki yanında %3,4, noktalı virgülde
     %1,3, iki noktada %2,4, boşluklu tirede %19,1'dir.
   * "orta": virgül sınır sayılmaz, diğerleri "kati" ile aynıdır.
   * "cumle": yalnızca cümle sonu (. ! ? …) ve paragraf.
   Sıra sayısı ("19. yüzyıl") ve yaygın kısaltmalardaki ("Prof. Dr.", "vb.")
   nokta cümle sonu sayılmaz. Paragraf (boş satır) her zaman sınırdır; tek
   satır sonu isteğe bağlıdır (PDF'ten kopyalanan metinlerde satır sonları
   cümle içinde olabilir).
5. Çıkarılan sözcükler (durak sözcük, sayı, çok kısa sözcük) bölüt içinde
   "boşluk" (None) olarak bırakılır. Böylece n-gram hesaplanırken, aralarında
   çıkarılmış bir sözcük bulunan iki sözcük yapay olarak yan yana
   getirilmez. Örneğin "eğitim ve öğretim" ifadesinden "eğitim öğretim"
   ikilisi üretilmez.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional

from .lemma import LEMMA_DURAK_SOZCUKLERI, Lemmatizer
from .stopwords import stopword_set

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

BOLUTLEME_SECENEKLERI = {
    "kati": "Katı — virgül, ; ve : da sınırdır (önerilen)",
    "orta": "Orta — virgül sınır değildir",
    "cumle": "Yalnızca cümle sonu ve paragraf",
}

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
    bolutleme: str = "kati"
    satir_sonu_sinir: bool = False
    lemmatizasyon: bool = False

    def __post_init__(self):
        if self.bolutleme not in BOLUTLEME_SECENEKLERI:
            raise ValueError(f"Geçersiz bölütleme seçeneği: {self.bolutleme}")
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

# Kesin sınır (paragraf, çıkarılan web adresi vb.) için kullanılan iç işaret
_SINIR = "\u2029"
# Sınır sayılmayacak noktaların yerine konan görünmez işaret (sözcük karakteri değildir)
_KORUMA = "\u2063"

# Noktası cümle sonu sayılmayan yaygın kısaltmalar
KISALTMALAR = ("prof", "doç", "dr", "yrd", "arş", "gör", "öğr", "uzm", "av", "op", "vb", "vs", "vd", "bkz", "bk",
               "örn", "krş", "yy", "çev", "haz", "ed", "yay", "no", "tel", "sf", "ss", "s", "bşk", "gn", "md")
_KISALTMA_RE = re.compile(r"(?<![^\W\d_])(" + "|".join(sorted(KISALTMALAR, key=len, reverse=True))
                          + r")\.(?=[ \t]*[^\s])", re.IGNORECASE)
# Sıra sayısı: "19. yüzyıl", "3. sınıf" (noktadan sonra küçük harfle devam eder)
_SIRA_SAYISI_RE = re.compile(r"(?<=\d)\.(?=[ \t]+[a-zçğıöşüâîû])")
_PARAGRAF_RE = re.compile(r"\n[ \t]*\n\s*")

# Bölüt sınırları (sayıların içindeki nokta, virgül ve iki nokta bölmez: 3,5 · 1.250 · 10:30)
_CUMLE_SONU = r"(?<!\d)\.|\.(?!\d)|[!?…" + _SINIR + "]"
_ORTA_EK = r"|(?<!\d):|:(?!\d)|[;()\[\]{}\"«»“”„|/\t]|(?<=\s)-(?=\s)"
_KATI_EK = r"|(?<!\d),|,(?!\d)"
_BOLUT_DESENLERI = {
    "cumle": re.compile(_CUMLE_SONU),
    "orta": re.compile(_CUMLE_SONU + _ORTA_EK),
    "kati": re.compile(_CUMLE_SONU + _ORTA_EK + _KATI_EK),
}

_SAYI_RE = re.compile(r"[\d.,:\-]+")


def _token_deseni(tireli: bool) -> re.Pattern:
    sozcuk = r"[^\W_]+(?:'[^\W_]+)?"
    if tireli:
        sozcuk = r"[^\W_]+(?:-[^\W_]+)*(?:'[^\W_]+)?"
    return re.compile(r"\d+(?:[.,:]\d+)+|" + sozcuk)


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


def bolutle(metin: str, bolutleme: str = "kati", satir_sonu_sinir: bool = False) -> List[str]:
    """Normalleştirilmiş metni n-gramların aşmayacağı bölütlere ayırır."""
    metin = metin.replace("\r\n", "\n").replace("\r", "\n")
    metin = _KISALTMA_RE.sub(lambda m: m.group(1) + _KORUMA, metin)
    metin = _SIRA_SAYISI_RE.sub(_KORUMA, metin)
    metin = _PARAGRAF_RE.sub(_SINIR, metin)
    metin = metin.replace("\n", _SINIR if satir_sonu_sinir else " ")
    return [p for p in _BOLUT_DESENLERI[bolutleme].split(metin) if p and p.strip()]


def on_isle(metin, ayarlar: OnIslemeAyarlari = OnIslemeAyarlari(),
            durak_kumesi: Optional[frozenset] = None, lemmatizer=None,
            lemma_kaydi: Optional[Counter] = None, _kucuk: Optional[dict] = None) -> IslenmisBelge:
    """Tek bir metni ön işler.

    lemma_kaydi verilirse, lemmatizasyonla değişen her (sözcük, lemma) çifti
    sayılır; kullanıcı dönüşümleri bu kayıttan denetleyebilir.
    """
    if metin is None or (isinstance(metin, float) and metin != metin):  # NaN
        return IslenmisBelge([])
    metin = unicode_normallestir(str(metin))

    if ayarlar.url_eposta_kaldir:
        sinir = f" {_SINIR} "
        metin = _URL_RE.sub(sinir, metin)
        metin = _DOI_RE.sub(sinir, metin)
        metin = _EPOSTA_RE.sub(sinir, metin)

    if ayarlar.sapkali_harf_normallestir:
        metin = metin.translate(_SAPKALI)

    if durak_kumesi is None:
        durak_kumesi = stopword_set(ayarlar.durak_sozcuk_genel, ayarlar.durak_sozcuk_akademik,
                                    ayarlar.ozel_durak_sozcukler)
    if ayarlar.sapkali_harf_normallestir:
        durak_kumesi = frozenset(w.translate(_SAPKALI) for w in durak_kumesi)
    if ayarlar.lemmatizasyon and lemmatizer is None:
        lemmatizer = Lemmatizer()

    desen = _TOKEN_TIRELI if ayarlar.tireli_birlesik_koru else _TOKEN_TIRESIZ
    bolutler: List[Bolut] = []
    for parca in bolutle(metin, ayarlar.bolutleme, ayarlar.satir_sonu_sinir):
        bolut: Bolut = []
        for ham in desen.findall(parca):
            for tok in _kesme_isle(ham, ayarlar.kesme_isareti):
                tok = tok.strip("-")
                if not tok:
                    continue
                if _SAYI_RE.fullmatch(tok):
                    bolut.append(None if ayarlar.sayilari_kaldir else sys.intern(tok))
                    continue
                if _kucuk is None:
                    anahtar = turkce_kucuk_harf(tok)
                else:
                    anahtar = _kucuk.get(tok)
                    if anahtar is None:
                        anahtar = _kucuk[tok] = turkce_kucuk_harf(tok)
                korunan_kisaltma = ayarlar.kisaltmalari_koru and _kisaltma_mi(tok)
                if ayarlar.kucuk_harf and not korunan_kisaltma:
                    tok = anahtar
                if anahtar in durak_kumesi or len(tok) < ayarlar.min_uzunluk:
                    bolut.append(None)
                    continue
                if lemmatizer is not None and not korunan_kisaltma:
                    lemma = lemmatizer(anahtar)
                    if lemma != anahtar and lemma_kaydi is not None:
                        lemma_kaydi[(anahtar, lemma)] += 1
                    if lemma in durak_kumesi or lemma in LEMMA_DURAK_SOZCUKLERI or len(lemma) < ayarlar.min_uzunluk:
                        bolut.append(None)
                        continue
                    tok = lemma
                # Aynı sözcüğün kopyaları bellekte tek nesne olarak tutulur
                bolut.append(sys.intern(tok))
        if bolut:
            bolutler.append(bolut)
    return IslenmisBelge(bolutler)


def derlem_on_isle(metinler: Iterable, ayarlar: OnIslemeAyarlari = OnIslemeAyarlari(),
                   lemma_kaydi: Optional[Counter] = None) -> List[IslenmisBelge]:
    """Bir metin listesini (derlemi) ön işler."""
    durak = stopword_set(ayarlar.durak_sozcuk_genel, ayarlar.durak_sozcuk_akademik,
                         ayarlar.ozel_durak_sozcukler)
    lem = Lemmatizer() if ayarlar.lemmatizasyon else None
    kucuk: dict = {}  # yalnızca bu çağrı süresince yaşar
    return [on_isle(m, ayarlar, durak, lem, lemma_kaydi, kucuk) for m in metinler]
