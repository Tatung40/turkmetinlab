"""Akademik terim sözlüğü ve çok sözcüklü terim tanıma (2.1).

Amaç
----
Türkçe akademik metinlerde sözlükteki (aday) terimleri, ek almış biçimleriyle birlikte tanımak ve
istenirse ön işlemede TEK BİR KAVRAMSAL BİRİM olarak korumak: "uzaktan eğitime" → `uzaktan_eğitim`.

Bilimsel tasarım kararları
--------------------------
1. Özgün metin değiştirilmez. Eşleştirme, karakter konumlarını özgün metne geri eşleyen normalleştirilmiş
   bir GÖRÜNÜM üzerinde yapılır (Türkçe küçük harf, şapka katlama â→a, kesme ve tire varyantları).
   Her eşleşme özgün metindeki başlangıç ve bitiş konumlarıyla saklanır: metin[bas:bit] = yüzey biçimi.
2. Terimin son sözcüğü DIŞINDAKİ sözcükler birebir eşleşmelidir. Sözcükler arasında yalnızca boşluk,
   tek satır sonu (isteğe bağlı) ya da tek bir tire olabilir; noktalama terimi böler. Tireyle önceki
   sözcüğe bağlanmış bir terim ("log-normal dağılım") daha büyük bir bileşiğin parçası sayılır, eşleşmez
   (tek harfli önekler hariç: "k-en yakın komşu algoritması").
3. Son sözcükte yalnızca ÇEKİM ekleri kabul edilir: çoğul, iyelik, hâl (yükleme, yönelme, bulunma,
   ayrılma, ilgi, araç), -ki ve ek-eylem (-DIr, -(y)DI, -(y)mIş, -(y)sA, -(y)ken). Ünlü uyumu, D/t
   değişimi ve kaynaştırma ünsüzleri (y, n, s) denetlenir. Ünsüz yumuşaması (p ç t k g → b c d ğ ğ;
   nk → ng) kabul edilir. Ünlü düşmesi ve ünsüz ikizleşmesi yalnızca kısa bir listedeki sözcüklerde kabul
   edilir; genel kural "eğitim → eğitme", "ölçüm → ölçme" gibi GERÇEK AMA FARKLI sözcükler üretir.
4. Belirtisiz isim tamlaması: son sözcük iyelik eki taşıyorsa (kuramı, dengesi, sağlığı) çoğul biçim de
   kabul edilir (kuramları, dengeleri, sağlıkları). Bu yolda ünlü uyumu son sözcüğün kendi iyelik
   ünlüsünden alınır; böylece "kontrolü → kontrolleri" gibi alıntı sözcük istisnaları da doğru işlenir.
4b. Zincirleme isim tamlamasında iyelik düşmesi: belirtisiz tamlama başka bir adın niteleyicisi olunca son
   sözcüğün iyelik eki düşebilir ("makine öğrenmesi" → "makine öğrenme yöntemleri", "Pearson korelasyonu"
   → "Pearson korelasyon katsayısı"). Eksiz biçim YALNIZCA ardından aynı bölütte bir sözcük geliyorsa
   kabul edilir; iyeliği düşmüş gövde ek de alabilir ("makine öğrenmenin", "yaşam kalitemizi"). Her iki
   durumda güven düzeyi daha düşüktür ("iyeliksiz"). Eşleşen bölüm terimin sözcükleridir.
5. Türetim ekleri (-CI, -lI, -lIk, -sIz, -sAl, -lAş, -CA …) eşleşmez: "uzaktan eğitimci", "yapay
   zekâsız" standartlaştırılmaz; "eşleştirilmeyen biçimler" tablosuna nedeniyle birlikte yazılır.
6. Kısaltmalar (YEM, NLP …) isteğe bağlıdır: büyük/küçük harfe duyarlı eşleşir ("yem" eşleşmez), ek
   yalnızca kesme işaretiyle kabul edilir (YEM'in) ve tamamı büyük harfli satırlarda (başlıklar) reddedilir.
7. Çakışmalar: uzun terim > kısa terim; eşitse standart > varyant > kısaltma ve ek almamış > kesme işaretli
   > çekimli; sonra kullanıcı terimi > alan sözlüğü > Genel araştırma ve istatistik; sonra metinde önce
   gelen. Reddedilen her aday (iç içe ya da kesişen) çakışma kaydına yazılır.
8. Güven düzeyi KURALA DAYALI SIRALI bir değerdir, olasılık değildir. Eşleşme türlerinin ölçülen
   doğruluğu için docs/bilimsel_degerlendirme_2.1.md'ye bakın.

Gizlilik: Yerleşik çekirdek sözlük (kullanıcı verisi değildir) süreç düzeyinde önbelleğe alınabilir.
Kullanıcı terimleri ve derlenmiş eşleştiriciler bu modülde hiçbir genel önbelleğe konmaz; uygulama
bunları yalnızca oturum belleğinde tutar. Ek kümesi önbelleği yalnızca ünlü uyumu profiline (ör. ("a",
ünlüyle bitmiyor, sert ünsüzle bitmiyor)) göre anahtarlanır, sözcüklere göre değil.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

KOK = os.path.dirname(os.path.abspath(__file__))
CEKIRDEK_YOLU = os.path.join(KOK, "sozluk", "akademik_terimler_cekirdek.csv")
SOZLUK_ADI = "TürkMetinLab çekirdek akademik terimler 2.1"
GENEL_ALAN = "Genel araştırma ve istatistik"
KULLANICI_ALANI = "Kullanıcı terimleri"
EN_FAZLA_KULLANICI_TERIMI = 5000
EN_FAZLA_TERIM_SOZCUGU = 10

TEMSILLER = {
    "birim": "Tek birim olarak koru (uzaktan eğitime → uzaktan_eğitim)",
    "isaretle": "Yalnızca işaretle (sözcükler değişmez, terim tablosu üretilir)",
}

ADAY_UYARISI = ("Sözlükteki terimler **uzman doğrulaması bekleyen aday terimlerdir**; doğrulanmış bilimsel "
                "terminoloji olarak sunulmaz. Tanıma kurala dayalıdır ve hata yapabilir.")

# Güven düzeyleri (sıralı; olasılık değildir)
GUVEN = {("standart", "yok"): 1.0, ("varyant", "yok"): 0.95, ("standart", "kesme"): 0.9,
         ("varyant", "kesme"): 0.9, ("standart", "çekim"): 0.85, ("varyant", "çekim"): 0.85,
         ("standart", "iyeliksiz"): 0.75, ("varyant", "iyeliksiz"): 0.75,
         ("kısaltma", "yok"): 0.7, ("kısaltma", "kesme"): 0.65}
_BICIM_SIRA = {"standart": 0, "varyant": 1, "kısaltma": 2}
_EK_SIRA = {"yok": 0, "kesme": 1, "çekim": 2, "iyeliksiz": 3}


# ---------------------------------------------------------------------------
# Sözlük kayıtları
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Terim:
    terim_id: str
    alan: str
    standart: str
    varyantlar: Tuple[str, ...] = ()
    kaynak: str = "çekirdek"  # "çekirdek" | "kullanıcı"
    durum: str = ""


def _temiz(s) -> str:
    s = unicodedata.normalize("NFKC", str(s if s is not None else ""))
    return re.sub(r"\s+", " ", s).strip()


def _bos(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or not str(x).strip()


def _varyant_ayir(s) -> Tuple[str, ...]:
    if _bos(s):
        return ()
    return tuple(v for v in (_temiz(x) for x in re.split(r"[;|]", str(s or ""))) if v)


@lru_cache(maxsize=1)
def _cekirdek_ham() -> Tuple[bytes, Tuple[Terim, ...]]:
    with open(CEKIRDEK_YOLU, "rb") as f:
        veri = f.read()
    okuyucu = csv.DictReader(io.StringIO(veri.decode("utf-8-sig")))
    terimler = tuple(Terim(r["terim_id"].strip(), _temiz(r["alan"]), _temiz(r["standart_terim"]),
                           _varyant_ayir(r.get("yazim_varyantlari")), "çekirdek", _temiz(r.get("dogrulama_durumu")))
                     for r in okuyucu)
    return veri, terimler


def cekirdek_sozluk() -> Tuple[Terim, ...]:
    """Projeyle gelen çekirdek sözlük (637 aday terim)."""
    return _cekirdek_ham()[1]


def sozluk_surumu() -> str:
    veri, terimler = _cekirdek_ham()
    return f"{SOZLUK_ADI} ({len(terimler)} aday terim; SHA-256 {hashlib.sha256(veri).hexdigest()[:12]})"


def alanlar() -> List[str]:
    sira = []
    for t in cekirdek_sozluk():
        if t.alan not in sira:
            sira.append(t.alan)
    return sorted(sira, key=lambda a: (a == GENEL_ALAN, a))


def sozluk_olustur(secili_alanlar: Optional[Iterable[str]] = None, kapali: Iterable[str] = (),
                   kullanici: Sequence[Terim] = ()) -> List[Terim]:
    """Etkin sözlük: seçilen alanlardaki çekirdek terimler + kullanıcı terimleri, devre dışı olanlar hariç."""
    secili = None if secili_alanlar is None else set(secili_alanlar)
    kapali = set(kapali)
    sonuc = [t for t in cekirdek_sozluk() if (secili is None or t.alan in secili) and t.terim_id not in kapali]
    sonuc += [t for t in kullanici if t.terim_id not in kapali]
    return sonuc


class SozlukHatasi(ValueError):
    pass


def kullanici_terimi(standart: str, varyantlar="", terim_id: str = "", alan: str = KULLANICI_ALANI) -> Terim:
    """Kullanıcının elle eklediği ya da dosyadan yüklediği terimi doğrular."""
    st_ = _temiz(standart)
    if not st_:
        raise SozlukHatasi("Terim boş olamaz.")
    if not re.search(r"[^\W\d_]", st_):
        raise SozlukHatasi(f"“{st_[:40]}” harf içermiyor.")
    if len(st_) > 120 or len(_sozcukler(katla(st_))) > EN_FAZLA_TERIM_SOZCUGU:
        raise SozlukHatasi(f"“{st_[:40]}…” çok uzun (en fazla {EN_FAZLA_TERIM_SOZCUGU} sözcük, 120 karakter).")
    vars_ = varyantlar if isinstance(varyantlar, tuple) else _varyant_ayir(varyantlar)
    vars_ = tuple(v for v in vars_ if katla(v) != katla(st_) and re.search(r"[^\W\d_]", v))
    return Terim(terim_id or "", _temiz(alan) or KULLANICI_ALANI, st_, vars_, "kullanıcı", "kullanıcı ekledi")


def sozluk_tablosundan(tablo, onek: str = "Y") -> Tuple[List[Terim], List[str]]:
    """Kullanıcının yüklediği CSV/Excel tablosundan terim listesi.

    Sütunlar (büyük/küçük harf duyarsız): standart_terim (ya da terim; yoksa ilk sütun) zorunlu;
    yazim_varyantlari (ya da varyantlar), alan, terim_id isteğe bağlı.
    """
    if tablo is None or len(tablo) == 0:
        raise SozlukHatasi("Sözlük dosyasında terim bulunamadı.")
    adlar = {str(c).strip().lower(): c for c in tablo.columns}

    def sutun(*adaylar):
        for a in adaylar:
            if a in adlar:
                return adlar[a]
        return None

    s_terim = sutun("standart_terim", "terim", "standart terim") or tablo.columns[0]
    s_var = sutun("yazim_varyantlari", "yazım_varyantları", "varyantlar", "varyant")
    s_alan = sutun("alan")
    s_id = sutun("terim_id", "id", "kimlik")
    cekirdek_idler = {t.terim_id for t in cekirdek_sozluk()}
    terimler, uyarilar, gorulen, hatalar = [], [], {}, 0
    for i, satir in enumerate(tablo.itertuples(index=False), start=2):
        r = dict(zip(tablo.columns, satir))
        ham = r.get(s_terim)
        if _bos(ham):
            continue
        try:
            tid = "" if s_id is None or _bos(r.get(s_id)) else _temiz(r.get(s_id))
            if tid in cekirdek_idler or not tid:
                tid = f"{onek}{len(terimler) + 1:04d}"
            t = kullanici_terimi(ham, _varyant_ayir(r.get(s_var)) if s_var is not None else (), tid,
                                 KULLANICI_ALANI if s_alan is None or _bos(r.get(s_alan)) else _temiz(r.get(s_alan)))
        except SozlukHatasi:
            hatalar += 1
            continue
        anahtar = katla(t.standart)
        if anahtar in gorulen:
            uyarilar.append(f"“{t.standart}” birden fazla kez geçiyor; ilki kullanıldı.")
            continue
        gorulen[anahtar] = t
        terimler.append(t)
        if len(terimler) >= EN_FAZLA_KULLANICI_TERIMI:
            uyarilar.append(f"En fazla {EN_FAZLA_KULLANICI_TERIMI} terim yüklenebilir; kalan satırlar alınmadı.")
            break
    if hatalar:
        uyarilar.append(f"{hatalar} satır geçersiz olduğu için alınmadı (boş, harf içermeyen ya da çok uzun).")
    if not terimler:
        raise SozlukHatasi("Sözlük dosyasında geçerli terim bulunamadı. İlk sütunda ya da 'standart_terim' "
                           "sütununda terimler olmalı.")
    cekirdek = {katla(t.standart) for t in cekirdek_sozluk()}
    ortak = sum(katla(t.standart) in cekirdek for t in terimler)
    if ortak:
        uyarilar.append(f"{ortak} terim çekirdek sözlükte de var; çakışmada sizin kaydınız önceliklidir.")
    return terimler, uyarilar


def sozluk_disa_aktar(terimler: Sequence[Terim]):
    """Etkin sözlüğü, yeniden yüklenebilecek tablo biçiminde döndürür."""
    import pandas as pd

    return pd.DataFrame([{"terim_id": t.terim_id, "alan": t.alan, "standart_terim": t.standart,
                          "yazim_varyantlari": "; ".join(t.varyantlar), "kaynak": t.kaynak,
                          "dogrulama_durumu": t.durum} for t in terimler])


# ---------------------------------------------------------------------------
# Normalleştirme ve özgün metne geri eşleme
# ---------------------------------------------------------------------------

_KESME = "’‘ʼ`´′ʹ"
_TIRE = "‐‑‒–—−"
_CEVIRI = str.maketrans({**{k: "'" for k in _KESME}, **{t: "-" for t in _TIRE},
                         "I": "ı", "İ": "i", "â": "a", "î": "i", "û": "u", "Â": "a", "Î": "i", "Û": "u"})
_OZEL_RE = re.compile("[̇­​‌‍﻿⁠]|-[ \t]*\r?\n")
_SATIR_TIRESI_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\r?\n[ \t]*(?=[^\W\d_])")
_SILINEN = frozenset("­​‌‍﻿⁠")


def katla(s: str) -> str:
    """Eşleştirme için katlama: Türkçe küçük harf, şapka (â→a), kesme ve tire varyantları."""
    return s.translate(_CEVIRI).lower().replace("i̇", "i")


def _gorunum(metin: str):
    """(görünüm, başlangıç haritası, bitiş haritası). Haritalar None ise görünüm özgün metinle hizalıdır."""
    if unicodedata.is_normalized("NFKC", metin) and not _OZEL_RE.search(metin):
        v = katla(metin)
        if len(v) == len(metin):
            return v, None, None
    atla = set()
    for m in _SATIR_TIRESI_RE.finditer(metin):
        atla.update(range(m.start(), m.end()))
    parcalar, bas, bit = [], [], []
    i, n = 0, len(metin)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(metin[j]):
            j += 1
        if i in atla or metin[i] in _SILINEN:
            i = j
            continue
        norm = katla(unicodedata.normalize("NFKC", metin[i:j]))
        for c in norm:
            if unicodedata.combining(c):
                continue
            parcalar.append(c)
            bas.append(i)
            bit.append(j)
        i = j
    return "".join(parcalar), bas, bit


_KELIME_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
_BAG_RE = re.compile(r"[ \t]+|[ \t\r]*\n[ \t]*|-")
_BAG_SATIRSIZ_RE = re.compile(r"[ \t]+|-")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b|\b(?:doi:\s*)?10\.\d{4,9}/\S+")


def _sozcukler(katli: str) -> Tuple[str, ...]:
    return tuple(m.group() for m in re.finditer(r"[^\W_]+", katli))


# ---------------------------------------------------------------------------
# Türkçe çekim ekleri (ünlü uyumuna göre üretilen sonlu küme)
# ---------------------------------------------------------------------------

_UNLU = "aeıioöuü"
_SERT = "fstkçşhp"


def _son_unlu(s: str) -> str:
    for c in reversed(s):
        if c in _UNLU:
            return c
    return "e"


def _A(s):
    return "a" if _son_unlu(s) in "aıou" else "e"


def _I(s):
    v = _son_unlu(s)
    return "ı" if v in "aı" else "u" if v in "ou" else "i" if v in "ei" else "ü"


def _unlu_bitis(s):
    return s[-1] in _UNLU


def _D(s):
    return "t" if s[-1] in _SERT else "d"


def _hal_normal(s):
    """Yükleme, yönelme, bulunma, ayrılma, ilgi, araç (sırası önemli: 2 bulunma, 4 ilgi)."""
    y = "y" if _unlu_bitis(s) else ""
    return [y + _I(s), y + _A(s), _D(s) + _A(s), _D(s) + _A(s) + "n",
            ("n" if y else "") + _I(s) + "n", y + "l" + _A(s)]


def _hal_zamir(s):
    """3. kişi iyelikten (ve -ki'den) sonra n kaynaştırmalı hâller."""
    return ["n" + _I(s), "n" + _A(s), "nd" + _A(s), "nd" + _A(s) + "n", "n" + _I(s) + "n", "yl" + _A(s)]


def _kopulalar(s):
    y = "y" if _unlu_bitis(s) else ""
    dir_ = _D(s) + _I(s) + "r"
    return [dir_, dir_ + "l" + _A(dir_) + "r", (y + "d" if y else _D(s)) + _I(s), y + "m" + _I(s) + "ş",
            y + "s" + _A(s), y + "ken"]


def _ek_uret(govde: str, iyelik_sonrasi: bool) -> frozenset:
    sonuc = set()

    def ekle(s, ek):
        sonuc.add(ek)
        for k in _kopulalar(s):
            sonuc.add(ek + k)

    def hal_ve_ki(s, ek, zamir):
        ekle(s, ek)
        for i, h in enumerate(_hal_zamir(s) if zamir else _hal_normal(s)):
            ekle(s + h, ek + h)
            if i in (2, 4):  # bulunma ve ilgi hâlinden sonra -ki
                s2, e2 = s + h + "ki", ek + h + "ki"
                ekle(s2, e2)
                for h2 in _hal_zamir(s2):
                    ekle(s2 + h2, e2 + h2)
                s3, e3 = s2 + "ler", e2 + "ler"
                ekle(s3, e3)
                for h3 in _hal_normal(s3):
                    ekle(s3 + h3, e3 + h3)

    if iyelik_sonrasi:
        hal_ve_ki(govde, "", True)
    else:
        for cogul in (False, True):
            s = govde + ("l" + _A(govde) + "r" if cogul else "")
            e = s[len(govde):]
            hal_ve_ki(s, e, False)
            u = _unlu_bitis(s)
            k = "" if u else _I(s)
            iyelikler = [(k + "m", False), (k + "n", False), (("s" if u else "") + _I(s), True),
                         (k + "m" + _I(s) + "z", False), (k + "n" + _I(s) + "z", False)]
            if not cogul:
                lar = "l" + _A(s) + "r"
                iyelikler.append((lar + _I(s + lar), True))
            for iy, ucuncu in iyelikler:
                hal_ve_ki(s + iy, e + iy, ucuncu)
    sonuc.discard("")
    return frozenset(sonuc)


@lru_cache(maxsize=256)
def _ek_kumesi(unlu: str, unlu_bitis: bool, sert_bitis: bool, iyelik_sonrasi: bool) -> frozenset:
    """Ek kümesi yalnızca ses profiline bağlıdır; temsilci bir gövdeyle üretilir (sözcük önbelleğe girmez)."""
    temsil = "b" + unlu if unlu_bitis else "b" + unlu + ("t" if sert_bitis else "l")
    return _ek_uret(temsil, iyelik_sonrasi)


def _profil(s: str, unlu: Optional[str] = None):
    return (unlu or _son_unlu(s), _unlu_bitis(s), s[-1] in _SERT)


@lru_cache(maxsize=1)
def _tum_ekler() -> frozenset:
    """Kısaltmalar ve ünlüsüz sözcükler için: bütün profillerin birleşimi (kesme işaretinden sonra)."""
    k = set()
    for v in _UNLU:
        for ub, sb in ((True, False), (False, False), (False, True)):
            k |= _ek_kumesi(v, ub, sb, False) | _ek_kumesi(v, ub, sb, True)
    return frozenset(k)


_YUMUSAMA = {"p": "b", "ç": "c", "t": "d", "g": "ğ"}
_SERTLESME = {"b": "p", "c": "ç", "d": "t", "ğ": "k"}
# Ünlü düşmesi ve ünsüz ikizleşmesi: yalnızca bu listedeki sözcüklerde (genel kural yanlış pozitif üretir)
_UNLU_DUSMESI = {"kayıt": "kayd", "akıl": "akl", "beyin": "beyn", "zihin": "zihn", "fikir": "fikr",
                 "resim": "resm", "isim": "ism", "şehir": "şehr", "ömür": "ömr", "nehir": "nehr", "sabır": "sabr",
                 "hüküm": "hükm", "ağız": "ağz", "burun": "burn", "oğul": "oğl", "boyun": "boyn", "gönül": "gönl",
                 "karın": "karn", "alın": "aln", "emir": "emr", "zehir": "zehr", "metin": "metn", "keyif": "keyf",
                 "nakit": "nakd", "vakit": "vakt", "bağır": "bağr", "devir": "devr", "seyir": "seyr",
                 "kahır": "kahr", "nesil": "nesl", "asıl": "asl", "fasıl": "fasl", "şekil": "şekl", "kısım": "kısm"}
_IKIZLESME = {"tıp": "tıbb", "hak": "hakk", "his": "hiss", "ret": "redd", "af": "aff", "sır": "sırr", "hat": "hatt",
              "zıt": "zıdd", "üs": "üss", "şık": "şıkk", "zan": "zann", "hac": "hacc", "had": "hadd",
              "zam": "zamm", "fen": "fenn"}
_GERI_DUSME = {kisa: tam for tam, kisa in list(_UNLU_DUSMESI.items()) + list(_IKIZLESME.items())}


def _yumusat(w: str) -> Optional[str]:
    if len(w) < 2:
        return None
    c = w[-1]
    if c == "k":
        return w[:-1] + ("g" if w[-2] == "n" else "ğ")
    if c in _YUMUSAMA:
        return w[:-1] + _YUMUSAMA[c]
    return None


def _on_uyum_istisnasi(w: str) -> Optional[str]:
    """kontrol → kontrolü, saat → saati: kalın ünlülü ama ince ünlülü ek alan alıntı sözcükler için ek profil."""
    v = _son_unlu(w)
    if v in "aıou" and (w.endswith("l") or w.endswith("at") or w.endswith("alp")):
        return {"a": "e", "ı": "i", "o": "ö", "u": "ü"}[v]
    return None


def _iyelikli_mi(w: str, sozcuk_sayisi: int) -> Optional[str]:
    """Belirtisiz isim tamlamasının son sözcüğü iyelik ekli görünüyorsa iyeliksiz gövdeyi döndürür."""
    if sozcuk_sayisi < 2 or len(w) < 3 or w[-1] not in "ıiuü":
        return None
    if w[-2] == "s" and w[-3] in _UNLU:
        return w[:-2]
    if w[-2] not in _UNLU:
        return w[:-1]
    return None


def _iyeliksiz_govdeler(b: str) -> set:
    adaylar = {b}
    if b[-1] in _SERTLESME:
        adaylar.add(b[:-1] + _SERTLESME[b[-1]])
    if b in _GERI_DUSME:
        adaylar.add(_GERI_DUSME[b])
    return adaylar


def _iyeliksiz_mi(token: str, w: str, sozcuk_sayisi: int) -> bool:
    """Zincirleme tamlamada iyeliği düşmüş son sözcük mü? (öğrenmesi → öğrenme, sağlığı → sağlık)"""
    b = _iyelikli_mi(w, sozcuk_sayisi)
    if not b or len(b) < 3:
        return False
    return token in _iyeliksiz_govdeler(b)


@dataclass(frozen=True)
class _GovdeAday:
    govde: str
    profil: tuple
    zamir: bool  # 3. kişi iyelikten sonraki hâl ekleri
    sadece_unlu: bool = False  # yumuşama, ünlü düşmesi: ek ünlüyle başlamalı
    bos_olabilir: bool = False  # çoğul iyelik gövdesi ("kuramları") eksiz de geçerli


def _govde_adaylari(w: str, sozcuk_sayisi: int) -> Tuple[_GovdeAday, ...]:
    adaylar = [_GovdeAday(w, _profil(w), False)]
    alt = _on_uyum_istisnasi(w)
    if alt:
        adaylar.append(_GovdeAday(w, _profil(w, alt), False))
    y = _yumusat(w)
    if y and y != w:
        adaylar.append(_GovdeAday(y, _profil(y), False, sadece_unlu=True))
        if alt:
            adaylar.append(_GovdeAday(y, _profil(y, alt), False, sadece_unlu=True))
    for sozluk_ in (_UNLU_DUSMESI, _IKIZLESME):
        k = sozluk_.get(w)
        if k and k != w:
            adaylar.append(_GovdeAday(k, _profil(k, _son_unlu(w)), False, sadece_unlu=True))
            yk = _yumusat(k)
            if yk and yk != k and sozluk_ is _UNLU_DUSMESI:
                adaylar.append(_GovdeAday(yk, _profil(yk, _son_unlu(w)), False, sadece_unlu=True))
    b = _iyelikli_mi(w, sozcuk_sayisi)
    if b:
        adaylar.append(_GovdeAday(w, _profil(w), True))  # kuramı-na, dengesi-nde
        gövdeler = {b}
        if b[-1] in _SERTLESME:
            gövdeler.add(b[:-1] + _SERTLESME[b[-1]])
        if b.endswith("ng"):
            gövdeler.add(b[:-1] + "k")
        if b in _GERI_DUSME:
            gövdeler.add(_GERI_DUSME[b])
        ince = w[-1] in "iü"
        lar = ("ler" if ince else "lar") + ("i" if ince else "ı")
        for g in gövdeler:
            p = g + lar
            adaylar.append(_GovdeAday(p, _profil(p), True, bos_olabilir=True))
    return tuple(adaylar)


_TURETIM = ("cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü", "lı", "li", "lu", "lü", "sız", "siz", "suz", "süz",
            "sal", "sel", "laş", "leş", "lan", "len", "msı", "msi", "cık", "cik", "ca", "ce", "ça", "çe")


def son_sozcuk_cozumle(token: str, ek: Optional[str], w: str, sozcuk_sayisi: int) -> Tuple[Optional[str], str]:
    """Son sözcüğün terimin son sözcüğünün çekimli biçimi olup olmadığına karar verir.

    Döndürür: (ek_turu, neden). ek_turu: "yok" | "kesme" | "çekim" | "iyeliksiz" ya da None (eşleşmez).
    neden: eşleşmezse "türetim eki" | "tanınmayan ek" | "" (ilgisiz sözcük).
    """
    adaylar = _govde_adaylari(w, sozcuk_sayisi)
    if ek is not None:  # kesme işaretli: kesmeden önceki kısım terim sözcüğü ya da onun çekimli biçimi olmalı
        if token != w:
            ic, _ = son_sozcuk_cozumle(token, None, w, sozcuk_sayisi)
            if ic in ("çekim", "iyeliksiz") and ek in _tum_ekler():  # Sinir Ağları'dır
                return ic, ""
            return None, ""
        if not re.search(f"[{_UNLU}]", w) or not w[-1].isalpha():
            return ("kesme", "") if ek in _tum_ekler() else (None, "tanınmayan ek")
        for a in adaylar:
            if a.govde == w and ek in _ek_kumesi(*a.profil, a.zamir):
                return "kesme", ""
        return None, "tanınmayan ek"
    if token == w:
        return "yok", ""
    ilgili = None
    for a in adaylar:
        g = a.govde
        if not token.startswith(g):
            continue
        kalan = token[len(g):]
        if not kalan:
            if a.bos_olabilir:
                return "çekim", ""
            continue
        if a.sadece_unlu and kalan[0] not in _UNLU:
            continue
        if kalan in _ek_kumesi(*a.profil, a.zamir):
            return "çekim", ""
        if ilgili is None or len(g) > len(ilgili[0]):
            ilgili = (g, kalan)
    b = _iyelikli_mi(w, sozcuk_sayisi)
    if b and len(b) >= 3:  # iyeliği düşmüş gövde + ek: "makine öğrenmenin", "yaşam kalitemizi"
        for g in _iyeliksiz_govdeler(b):
            if token.startswith(g) and len(token) > len(g) and token[len(g):] in _ek_kumesi(*_profil(g), False):
                return "iyeliksiz", ""
    if ilgili is None:
        return None, ""
    kalan = ilgili[1]
    if any(kalan.startswith(t) for t in _TURETIM):
        return None, "türetim eki"
    return None, "tanınmayan ek"


# ---------------------------------------------------------------------------
# Eşleştirici
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Eslesme:
    terim_id: str
    alan: str
    standart: str
    yuzey: str
    bas: int
    bit: int
    bicim: str  # standart | varyant | kısaltma
    ek: str  # yok | kesme | çekim
    guven: float
    kaynak: str
    sozcuk_sayisi: int


@dataclass(frozen=True)
class Cakisma:
    secilen_id: str
    secilen_standart: str
    secilen_yuzey: str
    reddedilen_id: str
    reddedilen_standart: str
    reddedilen_yuzey: str
    tur: str  # iç içe | kesişen | aynı konum
    bas: int
    bit: int


@dataclass(frozen=True)
class Eslesmeyen:
    yuzey: str
    terim_id: str
    standart: str
    neden: str
    bas: int
    bit: int


@dataclass
class BelgeTerimleri:
    eslesmeler: List[Eslesme] = field(default_factory=list)
    cakismalar: List[Cakisma] = field(default_factory=list)
    eslesmeyenler: List[Eslesmeyen] = field(default_factory=list)


@dataclass(frozen=True)
class _Kalip:
    sozcukler: Tuple[str, ...]
    sira: int  # terim sırası
    bicim: str
    ozgun: str  # kısaltmalar için büyük/küçük harfe duyarlı biçim
    oncelik: int


def _buyuk_harfli_satir(satir: str) -> bool:
    """Tamamı büyük harfli başlık satırı mı? Kısaltmalar (≤5 harf) sayılmaz: "KEKK-YEM yöntemiyle" başlık değildir."""
    sozcukler = re.findall(r"[^\W\d_]{2,}", satir)
    if len(sozcukler) < 3:
        return False
    buyuk = [s for s in sozcukler if s.isupper()]
    return len(buyuk) / len(sozcukler) >= 0.6 and sum(len(s) >= 6 for s in buyuk) >= 2


def _kisaltma_mi(v: str) -> bool:
    return " " not in v and "-" not in v and len(v) <= 8 and sum(c.isupper() for c in v) >= 2


class TerimEslestirici:
    """Derlenmiş terim sözlüğü. Oturum belleğinde tutulur; genel önbelleğe konmaz."""

    def __init__(self, terimler: Sequence[Terim], kisaltmalar: bool = False, temsil: str = "birim"):
        if temsil not in TEMSILLER:
            raise ValueError(f"Geçersiz temsil: {temsil}")
        self.terimler: List[Terim] = list(terimler)
        self.kisaltmalar = kisaltmalar
        self.temsil = temsil
        self._ilk: Dict[str, List[_Kalip]] = {}
        self._kisaltma: Dict[str, List[_Kalip]] = {}
        self._tek: Dict[str, List[_Kalip]] = {}  # tek sözcüklü terimlerin gövde adayları
        h = hashlib.sha256(f"{kisaltmalar}|{temsil}".encode())
        goruldu = set()
        for i, t in enumerate(self.terimler):
            h.update(repr((t.terim_id, t.alan, t.standart, t.varyantlar, t.kaynak)).encode("utf-8"))
            oncelik = 0 if t.kaynak != "çekirdek" else (2 if t.alan == GENEL_ALAN else 1)
            for bicim, yuzey in [("standart", t.standart)] + [("varyant", v) for v in t.varyantlar]:
                if bicim == "varyant" and _kisaltma_mi(yuzey):
                    if kisaltmalar:
                        k = _Kalip((katla(yuzey),), i, "kısaltma", yuzey, oncelik)
                        self._kisaltma.setdefault(katla(yuzey), []).append(k)
                    continue
                sozcukler = _sozcukler(katla(yuzey))
                if not sozcukler or (i, sozcukler) in goruldu:
                    continue  # katlandıktan sonra standartla aynı olan varyant (zekâ / zeka) bir kez derlenir
                goruldu.add((i, sozcukler))
                k = _Kalip(sozcukler, i, bicim, yuzey, oncelik)
                self._ilk.setdefault(sozcukler[0], []).append(k)
                if len(sozcukler) == 1:
                    for a in _govde_adaylari(sozcukler[0], 1):
                        self._tek.setdefault(a.govde, []).append(k)
        self.parmak_izi = h.hexdigest()
        govdeler = sorted(self._tek, key=len, reverse=True)
        self._tek_re = (re.compile(r"(?<![^\W_])(?:" + "|".join(map(re.escape, govdeler)) + ")")
                        if govdeler else None)

    def __len__(self):
        return len(self.terimler)

    def belirtec(self, eslesme: Eslesme, kucuk_harf: bool = True, sapka_sadelestir: bool = False) -> str:
        """Tek birim temsili: standart terim, sözcükleri alt çizgiyle birleştirilmiş."""
        s = "_".join(eslesme.standart.split())
        if kucuk_harf:
            s = s.replace("İ", "i").replace("I", "ı").lower()
        if sapka_sadelestir:
            s = s.translate(str.maketrans({"â": "a", "î": "i", "û": "u", "Â": "A", "Î": "İ", "Û": "U"}))
        return s

    # -- eşleştirme ---------------------------------------------------------
    def eslestir(self, metin, satir_sonu_sinir: bool = False) -> BelgeTerimleri:
        if metin is None or (isinstance(metin, float) and metin != metin) or not self.terimler:
            return BelgeTerimleri()
        metin = str(metin)
        gor, hbas, hbit = _gorunum(metin)

        def ozgun(s, e):  # görünüm aralığı → özgün metin aralığı
            if hbas is None:
                return s, e
            return hbas[s], hbit[e - 1]

        kelimeler = []  # (bas, bit, govde, ek)
        for m in _KELIME_RE.finditer(gor):
            t = m.group()
            g, _, ek = t.partition("'")
            kelimeler.append((m.start(), m.end(), g, ek if "'" in t else None))
        if not kelimeler:
            return BelgeTerimleri()
        bag_re = _BAG_SATIRSIZ_RE if satir_sonu_sinir else _BAG_RE
        n = len(kelimeler)

        def bagli(i):  # i ile i+1 arası terim içinde olabilir mi?
            return bag_re.fullmatch(gor, kelimeler[i][1], kelimeler[i + 1][0]) is not None

        engelli = set()
        if "http" in gor or "www." in gor or "@" in gor or "10." in gor:
            araliklar = [(m.start(), m.end()) for m in _URL_RE.finditer(gor)]
            if araliklar:
                for i, (s, e, _, _) in enumerate(kelimeler):
                    if any(a <= s < b for a, b in araliklar):
                        engelli.add(i)

        adaylar = []  # (i, j, kalip, ek_turu)
        sonuc = BelgeTerimleri()

        def dene(i, k: _Kalip):
            m = len(k.sozcukler)
            j = i + m - 1
            if j >= n:
                return
            for x in range(i, j):
                if x in engelli or kelimeler[x][3] is not None or kelimeler[x][2] != k.sozcukler[x - i] \
                        or not bagli(x):
                    return
            if j in engelli:
                return
            if i > 0 and gor[kelimeler[i - 1][1]:kelimeler[i][0]] == "-" and len(kelimeler[i - 1][2]) > 1:
                # "log-normal dağılım": tireyle önceki sözcüğe bağlı terim, daha büyük bir bileşiğin parçasıdır
                s, e = ozgun(kelimeler[i - 1][0], kelimeler[j][1])
                t = self.terimler[k.sira]
                sonuc.eslesmeyenler.append(Eslesmeyen(metin[s:e], t.terim_id, t.standart,
                                                      "tireli bileşiğin parçası", s, e))
                return
            _, _, g, ek = kelimeler[j]
            tur, neden = son_sozcuk_cozumle(g, ek, k.sozcukler[-1], m)
            if tur is None and ek is None and j + 1 < n and (j + 1) not in engelli and bagli(j) \
                    and _iyeliksiz_mi(g, k.sozcukler[-1], m) and re.search(r"[^\W\d_]", kelimeler[j + 1][2]):
                tur, neden = "iyeliksiz", ""
            if tur is not None:
                adaylar.append((i, j, k, tur))
            elif neden:
                s, e = ozgun(kelimeler[i][0], kelimeler[j][1])
                t = self.terimler[k.sira]
                sonuc.eslesmeyenler.append(Eslesmeyen(metin[s:e], t.terim_id, t.standart, neden, s, e))

        for i in range(n):
            for k in self._ilk.get(kelimeler[i][2], ()):
                if len(k.sozcukler) > 1:
                    dene(i, k)
        if self._tek_re is not None:
            konum = {s: i for i, (s, _, _, _) in enumerate(kelimeler)}
            for m in self._tek_re.finditer(gor):
                i = konum.get(m.start())
                if i is None:
                    continue
                g = kelimeler[i][2]
                denenen = set()
                for uz in range(len(g), 1, -1):  # sözcüğün önek olarak içerdiği bütün gövdeler
                    for k in self._tek.get(g[:uz], ()):
                        if (k.sira, k.bicim, k.sozcukler) not in denenen:
                            denenen.add((k.sira, k.bicim, k.sozcukler))
                            dene(i, k)
        if self._kisaltma:
            for i, (s, e, g, ek) in enumerate(kelimeler):
                for k in self._kisaltma.get(g, ()):
                    if i in engelli:
                        continue
                    os_, oe = ozgun(s, s + len(g))
                    if metin[os_:oe] not in (k.ozgun, k.ozgun.upper()):  # IoT ya da IOT; "yem" değil
                        continue
                    if ek is not None and ek not in _tum_ekler():
                        continue
                    satir_bas = metin.rfind("\n", 0, os_) + 1
                    satir_bit = metin.find("\n", oe)
                    if _buyuk_harfli_satir(metin[satir_bas: satir_bit if satir_bit >= 0 else len(metin)]):
                        sonuc.eslesmeyenler.append(Eslesmeyen(metin[os_:oe], self.terimler[k.sira].terim_id,
                                                              self.terimler[k.sira].standart,
                                                              "büyük harfli başlık satırı", os_, oe))
                        continue
                    adaylar.append((i, i, k, "kesme" if ek is not None else "yok"))

        # Çakışma çözümü: uzun > kısa; standart > varyant > kısaltma; eksiz > kesme > çekim; öncelik; konum
        adaylar.sort(key=lambda a: (-(a[1] - a[0] + 1), _BICIM_SIRA[a[2].bicim], _EK_SIRA[a[3]], a[2].oncelik,
                                    a[0], a[2].sira))
        dolu: Dict[int, int] = {}  # kelime → seçilen aday sırası
        secilen = []
        for a in adaylar:
            i, j, k, tur = a
            catisan = {dolu[x] for x in range(i, j + 1) if x in dolu}
            if not catisan:
                for x in range(i, j + 1):
                    dolu[x] = len(secilen)
                secilen.append(a)
                continue
            s, e = ozgun(kelimeler[i][0], kelimeler[j][1])
            for c in sorted(catisan):
                ci, cj, ck, _ = secilen[c]
                if (ci, cj) == (i, j):
                    if ck.sira == k.sira:
                        continue  # aynı terimin başka biçimi (ör. varyant): kayda gerek yok
                    tur_ = "aynı konum"
                elif ci <= i and j <= cj:
                    tur_ = "iç içe"
                else:
                    tur_ = "kesişen"
                cs, ce = ozgun(kelimeler[ci][0], kelimeler[cj][1])
                st_, rt = self.terimler[ck.sira], self.terimler[k.sira]
                sonuc.cakismalar.append(Cakisma(st_.terim_id, st_.standart, metin[cs:ce], rt.terim_id,
                                                rt.standart, metin[s:e], tur_, s, e))
        for i, j, k, tur in sorted(secilen, key=lambda a: a[0]):
            s, e = ozgun(kelimeler[i][0], kelimeler[j][1])
            t = self.terimler[k.sira]
            sonuc.eslesmeler.append(Eslesme(t.terim_id, t.alan, t.standart, metin[s:e], s, e, k.bicim, tur,
                                            GUVEN[(k.bicim, tur)], t.kaynak, len(k.sozcukler)))
        # Eşleşen bir terimin içinde kalan "eşleşmeyen" kayıtlar gürültüdür (ör. daha uzun terimin parçası)
        if sonuc.eslesmeyenler and sonuc.eslesmeler:
            araliklar = [(e.bas, e.bit) for e in sonuc.eslesmeler]
            sonuc.eslesmeyenler = [x for x in sonuc.eslesmeyenler
                                   if not any(a <= x.bas and x.bit <= b for a, b in araliklar)]
        return sonuc


def yer_degistir(metin: str, eslesmeler: Sequence[Eslesme], isaret: str = "ǂ") -> str:
    """Eşleşen bölümleri boşlukla ayrılmış yer tutucularla (ǂ<sıra>ǂ) değiştirir (ön işleme için)."""
    parcalar, son = [], 0
    for k, e in enumerate(eslesmeler):
        parcalar.append(metin[son:e.bas])
        parcalar.append(f" {isaret}{k}{isaret} ")
        son = e.bit
    parcalar.append(metin[son:])
    return "".join(parcalar)


# ---------------------------------------------------------------------------
# Sözlük bilgisi
# ---------------------------------------------------------------------------

def alanlar_arasi_belirsiz(terimler: Sequence[Terim]) -> Dict[str, str]:
    """Başka bir alandaki daha uzun bir terimin içinde geçen terimler (ör. eğitimdeki "derin öğrenme" ve
    mühendislikteki "derin öğrenme modeli"). Anlam alana göre değişebilir; kullanıcıya bildirilir."""
    sozcukler = [(t, _sozcukler(katla(t.standart))) for t in terimler]
    sonuc: Dict[str, str] = {}
    for a, sa in sozcukler:
        for b, sb in sozcukler:
            if a is b or a.alan == b.alan or len(sb) <= len(sa):
                continue
            if any(sb[i:i + len(sa)] == sa for i in range(len(sb) - len(sa) + 1)):
                sonuc.setdefault(a.terim_id, f"“{b.standart}” ({b.alan})")
    return sonuc


# ---------------------------------------------------------------------------
# Tablolar (arayüz ve dışa aktarım)
# ---------------------------------------------------------------------------

def _adlar(belgeler, adlar):
    return list(adlar) if adlar is not None else [f"Belge {i + 1}" for i in range(len(belgeler))]


def eslesme_tablosu(belgeler, adlar=None, satir_nolari=None):
    """Her eşleşme bir satır; konumlar özgün metindeki karakter konumlarıdır (0'dan başlar, bitiş hariç)."""
    import pandas as pd

    adlar = _adlar(belgeler, adlar)
    satirlar = []
    for i, b in enumerate(belgeler):
        for e in b.terimler:
            satirlar.append((adlar[i], satir_nolari[i] if satir_nolari is not None else i + 1, e.terim_id, e.alan,
                             e.standart, e.yuzey, e.bicim, e.ek, e.guven, e.bas, e.bit, e.kaynak))
    return pd.DataFrame(satirlar, columns=["Belge", "Satır", "Terim ID", "Alan", "Standart terim", "Metindeki biçim",
                                           "Eşleşme biçimi", "Ek", "Güven düzeyi", "Başlangıç", "Bitiş", "Kaynak"])


def cakisma_tablosu(belgeler, adlar=None):
    import pandas as pd

    adlar = _adlar(belgeler, adlar)
    satirlar = [(adlar[i], c.tur, c.secilen_standart, c.secilen_yuzey, c.reddedilen_standart, c.reddedilen_yuzey,
                 c.secilen_id, c.reddedilen_id, c.bas, c.bit)
                for i, b in enumerate(belgeler) if b.terim_sonucu is not None for c in b.terim_sonucu.cakismalar]
    return pd.DataFrame(satirlar, columns=["Belge", "Çakışma türü", "Seçilen terim", "Seçilen biçim",
                                           "Reddedilen terim", "Reddedilen biçim", "Seçilen ID", "Reddedilen ID",
                                           "Başlangıç", "Bitiş"])


def eslesmeyen_tablosu(belgeler, adlar=None):
    """Terime çok benzeyen ama standartlaştırılmayan biçimler (türetim eki, tanınmayan ek, başlık satırı)."""
    import pandas as pd

    adlar = _adlar(belgeler, adlar)
    satirlar = [(adlar[i], x.yuzey, x.standart, x.terim_id, x.neden, x.bas, x.bit)
                for i, b in enumerate(belgeler) if b.terim_sonucu is not None for x in b.terim_sonucu.eslesmeyenler]
    return pd.DataFrame(satirlar, columns=["Belge", "Metindeki biçim", "Benzediği terim", "Terim ID", "Neden",
                                           "Başlangıç", "Bitiş"])


def terim_ozeti(belgeler, belirsiz: Optional[Dict[str, str]] = None):
    """Terim başına sıklık, belge sayısı, çekimli geçiş oranı ve metindeki biçimler."""
    import pandas as pd
    from collections import Counter, defaultdict

    siklik, belge, cekimli, bicimler, bilgi = Counter(), Counter(), Counter(), defaultdict(Counter), {}
    for b in belgeler:
        gorulen = set()
        for e in b.terimler:
            siklik[e.terim_id] += 1
            cekimli[e.terim_id] += e.ek != "yok"
            bicimler[e.terim_id][e.yuzey.lower()] += 1
            bilgi[e.terim_id] = (e.standart, e.alan, e.kaynak)
            gorulen.add(e.terim_id)
        for t in gorulen:
            belge[t] += 1
    belirsiz = belirsiz or {}
    satirlar = [(bilgi[t][0], t, bilgi[t][1], n, belge[t], round(cekimli[t] / n, 3),
                 ", ".join(b for b, _ in bicimler[t].most_common(5)), belirsiz.get(t, ""))
                for t, n in siklik.most_common()]
    return pd.DataFrame(satirlar, columns=["Standart terim", "Terim ID", "Alan", "Sıklık", "Belge sayısı",
                                           "Ekli geçiş oranı", "Metindeki biçimler (en sık 5)",
                                           "Başka alanda daha uzun terimin parçası"])


def temizlenmis_metinler(belgeler, metinler, adlar=None, satir_nolari=None):
    """Özgün metin, temizlenmiş metin ve normalize edilmiş terimler yan yana."""
    import pandas as pd

    adlar = _adlar(belgeler, adlar)
    return pd.DataFrame({
        "Belge": adlar,
        "Satır": list(satir_nolari) if satir_nolari is not None else list(range(1, len(belgeler) + 1)),
        "Özgün metin": list(metinler),
        "Temizlenmiş metin": [" ".join(b.sozcukler) for b in belgeler],
        "Normalize terimler": ["; ".join(e.standart for e in b.terimler) for b in belgeler],
    })


def temizlenmis_txt(belgeler, adlar=None) -> bytes:
    """Her belge için bir başlık satırı ve temizlenmiş metin (UTF-8)."""
    adlar = _adlar(belgeler, adlar)
    parcalar = []
    for ad, b in zip(adlar, belgeler):
        parcalar.append(f"### {ad}\n" + " ".join(b.sozcukler) + "\n")
    return "\n".join(parcalar).encode("utf-8")


def ilk_n_karsilastir(kapali, acik, ad_sutunu: str, deger_sutunu: str, n: int = 20):
    """Sözlük kapalı/açık iki sıralı tabloyu (sıklık, n-gram, TF-IDF) yan yana koyar.

    Döndürür: (tablo, ölçütler). Ölçütler: ilk n listelerinin ortak öğe sayısı ve Jaccard benzerliği,
    açık listede tek birim terim (alt çizgili) sayısı.
    """
    import pandas as pd

    a = kapali.head(n).reset_index(drop=True)
    b = acik.head(n).reset_index(drop=True)
    uz = max(len(a), len(b))
    tablo = pd.DataFrame({
        "Sıra": range(1, uz + 1),
        "Sözlük kapalı": list(a[ad_sutunu]) + [""] * (uz - len(a)),
        f"{deger_sutunu} (kapalı)": list(a[deger_sutunu]) + [None] * (uz - len(a)),
        "Sözlük açık": list(b[ad_sutunu]) + [""] * (uz - len(b)),
        f"{deger_sutunu} (açık)": list(b[deger_sutunu]) + [None] * (uz - len(b)),
    })
    sa, sb = set(a[ad_sutunu]), set(b[ad_sutunu])
    olcut = {"ilk_n": n, "ortak": len(sa & sb), "jaccard": round(len(sa & sb) / len(sa | sb), 3) if sa | sb else 1.0,
             "acik_listede_terim_birimi": int(sum("_" in str(x) for x in b[ad_sutunu]))}
    return tablo, olcut
