"""TürkMetinLab 2.1 — terim tanıma validasyonu.

Bölümler
  1. Yapay test kümesi (degerlendirme/veri/terim_test_yapay.tsv): eşleştirici yazılmadan önce dondurulmuş,
     elle yazılmış kurgusal cümleler. Kesinlik, duyarlılık, F1; alan ve ek türüne göre; yanlış pozitifler.
  2. Gerçek akademik metin (DergiPark, 1.417 Türkçe mühendislik özeti; lisansı belirtilmediği için depoda
     yoktur, çalışma sırasında indirilir). Belgeler sabit tohumla geliştirme/test yarılarına ayrılır.
     Altın standart: test yarısında gevşek (yüksek duyarlıklı) aday arama + elle etiketleme
     (degerlendirme/veri/terim_altin_dergipark.tsv; etiketleyen geliştirici, uzman değil).
  3. Genel Türkçe metinde (UD Turkish-BOUN, 400 cümle) yanlış alarm.
  4. Ön işlemenin (durak sözcük, lemmatizasyon) terim bütünlüğüne etkisi.
  5. Sözlüğün TF-IDF, n-gram, NMF, LDA ve benzerlik sonuçlarına etkisi (DergiPark).

Kullanım:
    python degerlendirme/v21_terim_dogrulama.py yapay
    python degerlendirme/v21_terim_dogrulama.py adaylar      # gerçek veri: etiketlenecek adayları üretir
    python degerlendirme/v21_terim_dogrulama.py gercek       # etiketlere göre ölçüm
    python degerlendirme/v21_terim_dogrulama.py ud
    python degerlendirme/v21_terim_dogrulama.py onisleme
    python degerlendirme/v21_terim_dogrulama.py analiz
    python degerlendirme/v21_terim_dogrulama.py hepsi
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from tmlab import terimler as tm  # noqa: E402

VERI = os.path.join(KOK, "degerlendirme", "veri")
YAPAY = os.path.join(VERI, "terim_test_yapay.tsv")
ALTIN = os.path.join(VERI, "terim_altin_dergipark.tsv")
SONUC = os.path.join(KOK, "degerlendirme", "v21_terim_sonuclari.json")
TOHUM = 2101


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "kesinlik": round(p, 4), "duyarlilik": round(r, 4), "f1": round(f, 4)}


def _alan(tid):
    for t in tm.cekirdek_sozluk():
        if t.terim_id == tid:
            return t.alan
    return "?"


def _ekli_mi(yuzey: str, tid: str) -> bool:
    t = next(t for t in tm.cekirdek_sozluk() if t.terim_id == tid)
    bicimler = {tm._sozcukler(tm.katla(x)) for x in (t.standart,) + t.varyantlar}
    return tm._sozcukler(tm.katla(yuzey)) not in bicimler


def _ic_ice_mi(tid: str) -> bool:
    """Bu terimin içinde sözlükteki başka bir terim geçiyor mu?"""
    hedef = next(t for t in tm.cekirdek_sozluk() if t.terim_id == tid)
    sh = tm._sozcukler(tm.katla(hedef.standart))
    for t in tm.cekirdek_sozluk():
        if t is hedef:
            continue
        s = tm._sozcukler(tm.katla(t.standart))
        if len(s) < len(sh) and any(sh[i:i + len(s)] == s for i in range(len(sh) - len(s) + 1)):
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Yapay test kümesi
# ---------------------------------------------------------------------------

def yapay_kumeyi_oku(yol: str = YAPAY):
    for satir in open(yol, encoding="utf-8"):
        if satir.startswith("#") or not satir.strip():
            continue
        cumle, beklenen = satir.rstrip("\n").split("\t")
        kisaltma = cumle.startswith("[kisaltma] ")
        cumle = cumle.removeprefix("[kisaltma] ")
        gruplar = []
        if beklenen.strip() != "-":
            for oge in beklenen.split(";"):
                gruplar.append([tuple(x.rsplit("=", 1)) for x in oge.split("|")])
        yield cumle, gruplar, kisaltma


def yapay_kumeyi_degerlendir(yol: str = YAPAY) -> dict:
    E = tm.TerimEslestirici(tm.cekirdek_sozluk())
    EK = tm.TerimEslestirici(tm.cekirdek_sozluk(), kisaltmalar=True)
    genel = Counter()
    alan, ek, ice = defaultdict(Counter), defaultdict(Counter), Counter()
    hatalar, negatif_satir, negatif_yp = [], 0, 0
    for cumle, gruplar, kisaltma in yapay_kumeyi_oku(yol):
        sistem = [(e.yuzey, e.terim_id) for e in (EK if kisaltma else E).eslestir(cumle).eslesmeler]
        kalan = list(sistem)
        for grup in gruplar:
            bulunan = next((g for g in grup if g in kalan), None)
            ana = bulunan or grup[0]
            a = "Kısaltma" if kisaltma else _alan(ana[1])
            e_ = "kısaltma" if kisaltma else ("ekli" if _ekli_mi(*ana) else "eksiz")
            if bulunan:
                kalan.remove(bulunan)
                genel["tp"] += 1; alan[a]["tp"] += 1; ek[e_]["tp"] += 1
                if not kisaltma and _ic_ice_mi(ana[1]):
                    ice["tp"] += 1
            else:
                genel["fn"] += 1; alan[a]["fn"] += 1; ek[e_]["fn"] += 1
                if not kisaltma and _ic_ice_mi(ana[1]):
                    ice["fn"] += 1
                hatalar.append({"tür": "kaçırılan", "cümle": cumle, "beklenen": grup, "sistem": sistem})
        for y in kalan:
            genel["fp"] += 1
            a = "Kısaltma" if kisaltma else _alan(y[1])
            alan[a]["fp"] += 1
            ek["kısaltma" if kisaltma else ("ekli" if _ekli_mi(*y) else "eksiz")]["fp"] += 1
            hatalar.append({"tür": "yanlış pozitif", "cümle": cumle, "sistem": y})
        if not gruplar:
            negatif_satir += 1
            negatif_yp += bool(kalan)
    s = {"genel": _prf(genel["tp"], genel["fp"], genel["fn"]) | {"yanlis_pozitif": genel["fp"]},
         "alanlar": {k: _prf(v["tp"], v["fp"], v["fn"]) for k, v in sorted(alan.items())},
         "ek_turu": {k: _prf(v["tp"], v["fp"], v["fn"]) for k, v in sorted(ek.items())},
         "ic_ice_terimler": _prf(ice["tp"], 0, ice["fn"]),
         "negatif_cumleler": {"sayi": negatif_satir, "yanlis_pozitif_iceren": negatif_yp},
         "hatalar": hatalar}
    return s


def _yazdir_prf(baslik, d):
    print(f"  {baslik:40s} TP {d['tp']:4d}  YP {d['fp']:3d}  YN {d['fn']:3d}  "
          f"K {d['kesinlik']:.3f}  D {d['duyarlilik']:.3f}  F1 {d['f1']:.3f}")


def _kaydet(anahtar, deger):
    tum = {}
    if os.path.exists(SONUC):
        with open(SONUC, encoding="utf-8") as f:
            tum = json.load(f)
    tum[anahtar] = deger
    with open(SONUC, "w", encoding="utf-8") as f:
        json.dump(tum, f, ensure_ascii=False, indent=1)


def bolum_yapay():
    s = yapay_kumeyi_degerlendir()
    print("== Yapay test kümesi ==")
    _yazdir_prf("Genel", s["genel"])
    for k, v in s["alanlar"].items():
        _yazdir_prf(k, v)
    for k, v in s["ek_turu"].items():
        _yazdir_prf("Ek türü: " + k, v)
    _yazdir_prf("İç içe terimler (en uzunu bulma)", s["ic_ice_terimler"])
    print("  Negatif cümleler:", s["negatif_cumleler"])
    for h in s["hatalar"]:
        print("  HATA", h)
    _kaydet("yapay", s)
    return s


# ---------------------------------------------------------------------------
# 2. Gerçek akademik metin: DergiPark
# ---------------------------------------------------------------------------

def dergipark_belgeleri():
    """[(kimlik, metin, anahtar_sözcükler)]; metin = başlık + ". " + özet. Yalnızca bellekte.

    Not: Veri setindeki kimlik sütunu benzersiz değildir (414 kimlik 929 farklı özette tekrar eder) ve 20 özet
    birebir yinelenir. Bu yüzden kimlik olarak dosyadaki sıra kullanılır (D0001…) ve yinelenen metinler bir kez
    alınır; böylece aynı metin hem geliştirme hem test yarısına düşemez.
    """
    sys.path.insert(0, os.path.join(KOK, "degerlendirme"))
    from v2_dogrulama import dergipark

    sonuc, gorulen = [], set()
    for i, k in enumerate(dergipark(), start=1):
        metin = k["baslik"].rstrip(" .") + ". " + k["ozet"]
        if metin in gorulen:
            continue
        gorulen.add(metin)
        sonuc.append((f"D{i:04d}", metin, k["anahtar"]))
    return sonuc


def bolumle(belgeler):
    """Belgeleri sabit tohumla geliştirme ve test yarılarına ayırır (kimliğe göre)."""
    kimlikler = sorted({b[0] for b in belgeler})
    random.Random(TOHUM).shuffle(kimlikler)
    test = set(kimlikler[len(kimlikler) // 2:])
    return [b for b in belgeler if b[0] not in test], [b for b in belgeler if b[0] in test]


def _gevsek_katla(metin: str):
    """Eşleştiriciden BAĞIMSIZ basit katlama: (katlanmış metin, özgün konum listesi)."""
    harf = {"I": "ı", "İ": "i", "â": "a", "î": "i", "û": "u", "Â": "a", "Î": "i", "Û": "u", "’": "'", "‘": "'",
            "–": "-", "‐": "-", "‑": "-", "—": "-"}
    cikti, konum = [], []
    for i, c in enumerate(metin):
        if c in "̇­​":
            continue
        c = harf.get(c, c).lower()
        for x in c:
            cikti.append(x)
            konum.append(i)
    konum.append(len(metin))
    return "".join(cikti), konum


def _gevsek_desenler(kisaltmalar: bool = True):
    """Yüksek duyarlıklı aday desenleri: ilk sözcükler birebir, son sözcük önekle (son 2 harf serbest)."""
    desenler = []
    for t in tm.cekirdek_sozluk():
        for bicim in (t.standart,) + t.varyantlar:
            if tm._kisaltma_mi(bicim):
                if kisaltmalar:
                    desenler.append((t.terim_id, re.compile(r"(?<!\w)" + re.escape(_gevsek_katla(bicim)[0])
                                                            + r"(?:'\w+)?(?!\w)")))
                continue
            sozcukler = re.findall(r"\w+", _gevsek_katla(bicim)[0])
            son = sozcukler[-1]
            onek = son if len(son) <= 3 else son[:max(3, len(son) - 2)]
            govde = "".join(re.escape(s) + r"[\s\-]+" for s in sozcukler[:-1])
            desenler.append((t.terim_id, re.compile(r"(?<!\w)" + govde + re.escape(onek) + r"\w*(?:'\w+)?")))
    return desenler


def adaylari_bul(belgeler, desenler=None):
    """Gevşek desenlerle aday bölgeler: çakışan adaylar tek bölgede birleştirilir.

    Döndürür: [{kimlik, bas, bit, metin, baglam, adaylar: [terim_id…]}] (konumlar özgün metinde).
    """
    desenler = desenler or _gevsek_desenler()
    bolgeler = []
    for kimlik, metin, _ in belgeler:
        k, konum = _gevsek_katla(metin)
        bulunan = []
        for tid, d in desenler:
            for m in d.finditer(k):
                bulunan.append((konum[m.start()], konum[m.end() - 1] + 1, tid))
        bulunan.sort()
        grup = []
        for b in bulunan + [(10 ** 9, 10 ** 9, None)]:
            if grup and b[0] >= max(x[1] for x in grup):
                bas, bit = grup[0][0], max(x[1] for x in grup)
                bolgeler.append({"kimlik": kimlik, "bas": bas, "bit": bit, "metin": metin[bas:bit],
                                 "baglam": metin[max(0, bas - 60):bas] + "【" + metin[bas:bit] + "】"
                                 + metin[bit:bit + 40],
                                 "adaylar": sorted({x[2] for x in grup})})
                grup = []
            if b[2] is not None:
                grup.append(b)
    return bolgeler


def altin_oku(yol: str = ALTIN):
    """Altın standart: [(kimlik, bas, bit, etiket, terim_id, yüzey_bas, yüzey_bit, alternatifler)].

    Etiketler: T = terim gerçekleşmiş (en uzun sözlük terimi); A = sözlükteki dizi geçiyor ama anlam/alan farklı
    (ör. eğitimdeki "derin öğrenme" terimi makine öğrenmesi anlamında); D = türetilmiş biçim; V = sözlükte olmayan
    yazım/biçim varyantı (bilgisayarlı görme); N = terim değil.
    """
    satirlar = []
    for s in open(yol, encoding="utf-8"):
        if s.startswith("#") or not s.strip():
            continue
        kimlik, bas, bit, etiket, alternatifler = s.rstrip("\n").split("\t")
        alts = []
        for a in alternatifler.split("|") if alternatifler else []:
            tid, yb, ye = a.split(",")
            alts.append((tid, int(yb), int(ye)))
        satirlar.append((kimlik, int(bas), int(bit), etiket, alts))
    return satirlar


def gercek_degerlendir(kisaltmalar: bool = False, belgeler=None, alanlar_=None) -> dict:
    """DergiPark test yarısında altın standarda karşı ölçüm.

    İki düzey:
      * yüzey düzeyi: sözlük dizisinin (anlamı ne olursa olsun) doğru konum ve kimlikle bulunması (T + A pozitif);
      * kavram düzeyi: yalnızca terimin sözlükteki anlamıyla geçtiği yerler pozitif (A eşleşmeleri yanlış pozitif).
    Kısaltmalar kapalıyken kısaltma biçimindeki altın öğeler (IoT, NLP…) kapsam dışı sayılır.
    alanlar_: yalnızca bu alanlar seçilmiş gibi ölçer (seçilmeyen alanların terimleri kapsam dışıdır).
    """
    if belgeler is None:
        _, belgeler = bolumle(dergipark_belgeleri())
    metin = {k: m for k, m, _ in belgeler}
    altin = altin_oku()
    sozluk = tm.sozluk_olustur(alanlar_)
    kapsam = {t.terim_id for t in sozluk}
    E = tm.TerimEslestirici(sozluk, kisaltmalar=kisaltmalar)
    sistem = {k: E.eslestir(m).eslesmeler for k, m in metin.items()}

    def kisaltma_ogesi(k, yb, ye):
        return tm._kisaltma_mi(metin[k][yb:ye].split("'")[0].split("’")[0])

    pozitif = []  # (kimlik, etiket, alternatifler)
    for k, b, e, etiket, alts in altin:
        if etiket in "TA" and alts:
            alts = [a for a in alts if a[0] in kapsam]
            if not alts:
                continue
            if not kisaltmalar and all(kisaltma_ogesi(k, yb, ye) for _, yb, ye in alts):
                continue
            pozitif.append((k, b, e, etiket, alts))
    bolgeler = defaultdict(list)
    kapsam_disi = defaultdict(list)  # altın terimi seçilmeyen alanda olan T bölgeleri
    for k, b, e, etiket, alts in altin:
        bolgeler[k].append((b, e, etiket))
        if etiket in "TA" and alts and not any(a[0] in kapsam for a in alts):
            kapsam_disi[k].append((b, e))

    sonuc = {}
    for duzey, sayilan in (("yuzey", "TA"), ("kavram", "T")):
        tp = fp = fn = 0
        kategori = Counter()
        ek_tp, ek_fp, ek_fn = Counter(), Counter(), Counter()
        alan_ = defaultdict(Counter)
        kullanildi = set()
        for k, b, e, etiket, alts in pozitif:
            if etiket not in sayilan:
                continue
            bulunan = None
            for s_ in sistem[k]:
                if any((s_.terim_id, s_.bas, s_.bit) == a for a in alts) and (k, s_.bas) not in kullanildi:
                    bulunan = s_
                    break
            tid = alts[0][0]
            if bulunan:
                kullanildi.add((k, bulunan.bas))
                tp += 1
                ek_tp[bulunan.ek] += 1
                alan_[_alan(tid)]["tp"] += 1
            else:
                fn += 1
                ek_fn["ekli" if _ekli_mi(metin[k][alts[0][1]:alts[0][2]], tid) else "eksiz"] += 1
                alan_[_alan(tid)]["fn"] += 1
                kategori["kaçırılan: " + ("ekli" if _ekli_mi(metin[k][alts[0][1]:alts[0][2]], tid) else "eksiz")] += 1
        hatalar = []
        for k, liste in sistem.items():
            for s_ in liste:
                if (k, s_.bas) in kullanildi:
                    continue
                if not kisaltmalar and s_.bicim == "kısaltma":
                    continue
                if any(b <= s_.bas and s_.bit <= e for b, e in kapsam_disi[k]):
                    continue  # kapsam dışı (seçilmeyen alandaki) daha uzun terimin içindeki geçerli alt terim
                fp += 1
                ek_fp[s_.ek] += 1
                alan_[_alan(s_.terim_id)]["fp"] += 1
                ortusen = [et for b, e, et in bolgeler[k] if b < s_.bit and s_.bas < e]
                neden = ("anlam/alan farkı (A)" if "A" in ortusen else "sınır/kimlik hatası" if "T" in ortusen
                         else f"etiket {ortusen[0]}" if ortusen else "etiketlenmemiş bölge")
                kategori["yanlış pozitif: " + neden] += 1
                hatalar.append({"kimlik": k, "yuzey": s_.yuzey, "standart": s_.standart, "ek": s_.ek, "neden": neden})
        tur = {}
        for t in sorted(set(ek_tp) | set(ek_fp)):
            tur[t] = {"tp": ek_tp[t], "fp": ek_fp[t],
                      "kesinlik": round(ek_tp[t] / (ek_tp[t] + ek_fp[t]), 4) if ek_tp[t] + ek_fp[t] else None}
        sonuc[duzey] = {"genel": _prf(tp, fp, fn), "eslesme_turune_gore_kesinlik": tur,
                        "kacirilan_ek_durumu": dict(ek_fn), "kategoriler": dict(kategori),
                        "alanlar": {a: _prf(v["tp"], v["fp"], v["fn"]) for a, v in sorted(alan_.items())},
                        "hatalar": hatalar}
    sonuc["altin_ozet"] = dict(Counter(et for _, _, _, et, _ in altin))
    sonuc["belge_sayisi"] = len(belgeler)
    return sonuc


def bolum_gercek():
    print("== Gerçek veri (DergiPark test yarısı) ==")
    tum = {}
    for kis in (False, True):
        s = gercek_degerlendir(kisaltmalar=kis)
        ad = "kısaltmalar açık" if kis else "varsayılan (kısaltmalar kapalı)"
        print(f"-- {ad}; altın: {s['altin_ozet']}")
        for duzey in ("yuzey", "kavram"):
            _yazdir_prf(f"{duzey} düzeyi", s[duzey]["genel"])
        print("  eşleşme türüne göre kesinlik (kavram):", s["kavram"]["eslesme_turune_gore_kesinlik"])
        print("  kategoriler (kavram):", s["kavram"]["kategoriler"])
        print("  kategoriler (yüzey):", s["yuzey"]["kategoriler"])
        for a, v in s["kavram"]["alanlar"].items():
            _yazdir_prf("  " + a, v)
        tum["kisaltmali" if kis else "varsayilan"] = s
    secim = ["Mühendislik, bilgisayar ve teknoloji", "Genel araştırma ve istatistik"]
    s = gercek_degerlendir(alanlar_=secim)
    print("-- yalnızca mühendislik + genel araştırma alanları seçili (mühendislik araştırmacısı senaryosu)")
    for duzey in ("yuzey", "kavram"):
        _yazdir_prf(f"{duzey} düzeyi", s[duzey]["genel"])
    tum["alan_secimi_muhendislik_genel"] = s
    _kaydet("gercek_dergipark", tum)
    return tum



# ---------------------------------------------------------------------------
# 3. Genel Türkçe metinde yanlış alarm (UD Turkish-BOUN, 400 cümle)
# ---------------------------------------------------------------------------

UD_YOLU = os.path.join(KOK, "tests", "veri", "ud_boun_test_ornek.conllu")


def ud_cumleleri():
    return [s.split("=", 1)[1].strip() for s in open(UD_YOLU, encoding="utf-8") if s.startswith("# text")]


def bolum_ud():
    """Akademik olmayan metinde sözlük eşleşmeleri; her biri elle değerlendirilmek üzere listelenir."""
    E = tm.TerimEslestirici(tm.cekirdek_sozluk(), kisaltmalar=True)
    cumleler = ud_cumleleri()
    kayit = []
    for c in cumleler:
        for e in E.eslestir(c).eslesmeler:
            kayit.append({"cumle_baglami": c[max(0, e.bas - 50):e.bit + 30], "yuzey": e.yuzey, "standart": e.standart})
    print(f"== UD BOUN: {len(cumleler)} cümle, {sum(len(c.split()) for c in cumleler)} sözcük, "
          f"{len(kayit)} eşleşme ==")
    for k in kayit:
        print("  ", k)
    s = {"cumle": len(cumleler), "sozcuk": sum(len(c.split()) for c in cumleler), "eslesme": kayit,
         "elle_degerlendirme": "Tek eşleşme ('sivil toplum kuruluşları') gerçek terimdir; yanlış alarm 0 "
                               "(15 Ekim 2026 ölçümü, kısaltmalar açık)."}
    _kaydet("ud_genel_metin", s)
    return s



# ---------------------------------------------------------------------------
# 4. Ön işlemenin terim bütünlüğüne etkisi
# ---------------------------------------------------------------------------

ONISLEME_DURUMLARI = {
    "hiçbiri": dict(durak_sozcuk_genel=False),
    "durak sözcük (varsayılan)": dict(),
    "durak + akademik kalıp": dict(durak_sozcuk_akademik=True),
    "lemmatizasyon": dict(durak_sozcuk_genel=False, lemmatizasyon=True),
    "lemma + durak": dict(lemmatizasyon=True),
}


def onisleme_etkisi(belgeler=None) -> dict:
    """Altın standarttaki her terim geçişi için (DergiPark test yarısı, T etiketli, kısaltmalar hariç):

    * bölünme: ön işleme terimin içine boşluk (çıkarılmış sözcük) koyuyor mu? (n-gram olarak sayılamaz)
    * standart biçim: işlenmiş dizi, terimin standart biçimiyle (ör. "uzaktan eğitim") aynı mı?
    * tutarlı temsil: işlenmiş dizi, standart biçimin AYNI ayarlarla işlenmiş haliyle aynı mı?
      (tüm geçişler tek bir n-gramda toplanabilir mi?)
    Sözlük açıkken ("birim") her geçiş tek belirteç olmalıdır.
    """
    from tmlab.preprocess import OnIslemeAyarlari, on_isle

    if belgeler is None:
        _, belgeler = bolumle(dergipark_belgeleri())
    metin = {k: m for k, m, _ in belgeler}
    standart = {t.terim_id: t.standart for t in tm.cekirdek_sozluk()}
    ogeler = []
    for k, b, e, etiket, alts in altin_oku():
        if etiket != "T" or not alts:
            continue
        tid, yb, ye = alts[0]
        yuzey = metin[k][yb:ye]
        if tm._kisaltma_mi(yuzey.split("'")[0]):
            continue
        ogeler.append((yuzey, tid))
    E = tm.TerimEslestirici(tm.cekirdek_sozluk())
    sonuc = {"terim_gecisi": len(ogeler)}
    for ad, kw in ONISLEME_DURUMLARI.items():
        ayar = OnIslemeAyarlari(**kw)
        bolunen = std = tutarli = birim = 0
        for yuzey, tid in ogeler:
            bolutler = on_isle(yuzey, ayar).bolutler
            dizi = [t for b in bolutler for t in b]
            if None in dizi[1:-1] or len(bolutler) > 1:
                bolunen += 1
            temiz = [t for t in dizi if t is not None]
            std += temiz == standart[tid].replace("â", "a").lower().split() or temiz == standart[tid].lower().split()
            hedef = [t for b in on_isle(standart[tid], ayar).bolutler for t in b if t is not None]
            tutarli += temiz == hedef and None not in dizi[1:-1]
            # Bağlam: iyeliği düşmüş biçim ("makine öğrenme") ancak ardından bir sözcük gelince geçerlidir
            r_ = E.eslestir(yuzey + " yöntemi")
            acik = on_isle(yuzey + " yöntemi", ayar, terimci=E).sozcukler
            birim += bool(r_.eslesmeler) and acik[:1] == [E.belirtec(r_.eslesmeler[0])] \
                and r_.eslesmeler[0].bit == len(yuzey)
        n = len(ogeler)
        sonuc[ad] = {"sözlük kapalı: içine boşluk giren (bölünen)": round(bolunen / n, 4),
                     "sözlük kapalı: standart biçimde kalan": round(std / n, 4),
                     "sözlük kapalı: tutarlı temsil (tek n-gramda toplanabilir)": round(tutarli / n, 4),
                     "sözlük açık: tek birim olarak korunan": round(birim / n, 4)}
    return sonuc


def bolum_onisleme():
    s = onisleme_etkisi()
    print(f"== Ön işlemenin terimlere etkisi ({s['terim_gecisi']} gerçek terim geçişi) ==")
    for ad, v in s.items():
        if isinstance(v, dict):
            print(f"  {ad:28s}", v)
    _kaydet("onisleme_etkisi", s)
    return s


# ---------------------------------------------------------------------------
# 5. Sözlüğün analizlere etkisi (DergiPark özetleri)
# ---------------------------------------------------------------------------

def analiz_etkisi() -> dict:
    import time

    import numpy as np

    from tmlab.analysis import ngram_sikligi, tfidf_hesapla
    from tmlab.benzerlik import kosinus_matrisi
    from tmlab.konu import KonuAyarlari, konu_modeli
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

    sys.path.insert(0, os.path.join(KOK, "degerlendirme"))
    from v2_dogrulama import _auc, dergipark

    kayitlar, gorulen = [], set()
    for k in dergipark():
        if k["ozet"] not in gorulen:
            gorulen.add(k["ozet"])
            kayitlar.append(k)
    ozetler = [k["ozet"] for k in kayitlar]
    ayar = OnIslemeAyarlari(durak_sozcuk_ingilizce=True)
    E = tm.TerimEslestirici(tm.cekirdek_sozluk())
    rng = np.random.default_rng(0)
    sonuc = {"belge": len(kayitlar), "ayarlar": "durak sözcük (Türkçe + İngilizce), lemma kapalı; sözlük: 6 alan, birim"}
    temsil = {}
    for ad, terimci in (("kapali", None), ("acik", E)):
        t = time.perf_counter()
        belgeler = derlem_on_isle(ozetler, ayar, terimci=terimci)
        sure = time.perf_counter() - t
        anahtar = [set(x for terim in k["anahtar"] for x in derlem_on_isle([terim], ayar, terimci=terimci)[0].sozcukler)
                   for k in kayitlar]
        tf = tfidf_hesapla(belgeler, [str(i) for i in range(len(belgeler))])
        ng = ngram_sikligi(belgeler, 2)
        r = {"on_isleme_sn": round(sure, 2), "sozcuk": sum(len(b.sozcukler) for b in belgeler),
             "farkli_sozcuk": len({w for b in belgeler for w in b.sozcukler})}
        # Benzerlik: yazar anahtar sözcüğü paylaşan çiftler (anahtar sözcükler HAM, ön işlemeden bağımsız)
        S = kosinus_matrisi(tf.matris)
        ham = [k["anahtar"] for k in kayitlar]
        i, j = np.triu_indices(len(kayitlar), k=1)
        ortak = np.array([bool(ham[a] & ham[b]) for a, b in zip(i, j)])
        r["benzerlik_auc_ortak_anahtar"] = round(_auc(S[i, j][ortak], S[i, j][~ortak]), 4)
        for yontem in ("nmf", "lda"):
            t = time.perf_counter()
            m = konu_modeli(belgeler, KonuAyarlari(yontem, 10), kararlilik_tekrari=3)
            ust = [set(u) for u in m.ust_sozcukler(10)]
            baskin = np.argmax(m.belge_konu, axis=1)
            idx = m.belge_indisleri
            ortusme = np.mean([len(ust[baskin[q]] & anahtar[d]) > 0 for q, d in enumerate(idx)])
            taban = np.mean([np.mean([len(ust[b] & anahtar[d]) > 0 for b, d in zip(rng.permutation(baskin), idx)])
                             for _ in range(50)])
            r[yontem] = {**{a: v for a, v in m.olcutler.items()}, "anahtar_ortusme": round(float(ortusme), 4),
                         "anahtar_ortusme_rastgele": round(float(taban), 4), "sure_sn": round(time.perf_counter() - t, 1),
                         "ornek_konular": [" ".join(u) for u in m.ust_sozcukler(8)[:4]]}
        temsil[ad] = (tf.derlem_tablosu, ng)
        sonuc[ad] = r
    for ad_, (tablo_ad, sutun, deger) in {"tfidf_ilk30": (0, "Terim", "Ortalama TF-IDF"),
                                          "bigram_ilk30": (1, "N-gram", "Sıklık")}.items():
        _, olcut = tm.ilk_n_karsilastir(temsil["kapali"][tablo_ad], temsil["acik"][tablo_ad], sutun, deger, 30)
        olcut["acik_ilk10"] = list(temsil["acik"][tablo_ad][sutun].head(10))
        olcut["kapali_ilk10"] = list(temsil["kapali"][tablo_ad][sutun].head(10))
        sonuc[ad_] = olcut
    return sonuc


def bolum_analiz():
    s = analiz_etkisi()
    print("== Sözlüğün analizlere etkisi ==")
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
    _kaydet("analiz_etkisi", s)
    return s


if __name__ == "__main__":
    bolum = sys.argv[1] if len(sys.argv) > 1 else "yapay"
    if bolum in ("yapay", "hepsi"):
        bolum_yapay()
    if bolum in ("gercek", "hepsi"):
        bolum_gercek()
    if bolum in ("ud", "hepsi"):
        bolum_ud()
    if bolum in ("onisleme", "hepsi"):
        bolum_onisleme()
    if bolum in ("analiz", "hepsi"):
        bolum_analiz()
