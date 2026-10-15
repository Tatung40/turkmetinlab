"""Küçük, orta ve büyük derlemlerde süre ve bellek ölçümü.

Kullanım:
    python degerlendirme/performans_olcumu.py            # küçük + orta + büyük
    python degerlendirme/performans_olcumu.py kucuk orta # yalnızca seçilenler

Her ölçüm ayrı bir süreçte yapılır; böylece en yüksek bellek (RSS) değerleri
birbirini etkilemez. Derlemler örnek veri setindeki gerçek Türkçe cümlelerin
sabit tohumla (tekrarlanabilir) yeniden birleştirilmesiyle üretilir.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

BOYUTLAR = {
    "kucuk": (16, 60),        # (belge sayısı, belge başına cümle)  ~ 1 bin sözcük ölçeği
    "orta": (2000, 12),       # ~ 300 bin sözcük
    "buyuk": (20000, 12),     # ~ 3 milyon sözcük
    # Gerçek derlemlere yakın, geniş sözcük dağarcıklı (Zipf dağılımlı) zor senaryolar
    "orta_genis": (2000, 12),
    "buyuk_genis": (20000, 12),
}


def zipf_derlem(belge: int, cumle: int, tohum: int = 42, dagarcik: int = 200000):
    """Türkçe hecelerden üretilmiş sözde sözcüklerle, Zipf (s=1) dağılımlı derlem."""
    import bisect
    import itertools

    r = random.Random(tohum)
    heceler = [a + b for a in "bcçdfgğhjklmnprsştvyz" for b in "aeıioöuü"] + list("aeıioöuü")
    sozcukler, gorulen = [], set()
    while len(sozcukler) < dagarcik:
        w = "".join(r.choice(heceler) for _ in range(r.randint(2, 4)))
        if w not in gorulen:
            gorulen.add(w)
            sozcukler.append(w)
    agirlik = list(itertools.accumulate(1.0 / (i + 1) for i in range(dagarcik)))
    toplam = agirlik[-1]
    noktalama = [".", ",", "", "", "", "", ""]
    belgeler = []
    for _ in range(belge):
        parca = []
        for _ in range(cumle * 12):
            parca.append(sozcukler[bisect.bisect_left(agirlik, r.random() * toplam)] + r.choice(noktalama))
        belgeler.append(" ".join(parca))
    return belgeler


def cumle_havuzu():
    import csv
    import re

    yol = os.path.join(KOK, "ornek_veri", "ornek_akademik_ozetler.csv")
    with open(yol, encoding="utf-8-sig") as f:
        metinler = [r["ozet"] for r in csv.DictReader(f)]
    with open(os.path.join(KOK, "ornek_veri", "ornek_metin.txt"), encoding="utf-8") as f:
        metinler += [p for p in f.read().split("\n\n") if p.strip()]
    cumleler = []
    for m in metinler:
        cumleler += [c.strip() for c in re.split(r"(?<=[.!?])\s+", m) if len(c.split()) > 3]
    return cumleler


def sentetik_derlem(belge: int, cumle: int, tohum: int = 42):
    r = random.Random(tohum)
    havuz = cumle_havuzu()
    # Sözcük dağarcığını gerçekçi biçimde büyütmek için bazı sözcüklere ek/sayı eklenir
    ekler = ["", "", "", "ler", "in", "de", "nin", "ları", "sı"]
    belgeler = []
    for _ in range(belge):
        secilen = []
        for _ in range(cumle):
            c = r.choice(havuz).split()
            if r.random() < 0.5:
                i = r.randrange(len(c))
                c[i] = c[i].rstrip(".,") + r.choice(ekler) + str(r.randrange(50) if r.random() < 0.1 else "")
            secilen.append(" ".join(c))
        belgeler.append(" ".join(secilen))
    return belgeler


def _durum(alan: str) -> float:
    with open("/proc/self/status") as f:
        for satir in f:
            if satir.startswith(alan + ":"):
                return int(satir.split()[1]) / 1024
    return 0.0


def rss_mb() -> float:
    """Anlık bellek (RSS)."""
    return _durum("VmRSS")


def tepe_mb() -> float:
    """Bu sürecin tepe belleği. ru_maxrss yerine VmHWM kullanılır: Linux'ta ru_maxrss üst
    süreçten devralınır ve alt süreç ölçümlerini bozar."""
    return _durum("VmHWM")


def tek_olcum(ad: str, lemma: bool = False) -> dict:
    from tmlab.analysis import derlem_ozeti, ngram_sikligi, sozcuk_sikligi, tfidf_hesapla
    from tmlab.export import csv_bayt, excel_bayt
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle
    from tmlab.visualize import kelime_bulutu_png

    belge, cumle = BOYUTLAR[ad]
    metinler = zipf_derlem(belge, cumle) if ad.endswith("_genis") else sentetik_derlem(belge, cumle)
    rss0 = rss_mb()
    sure = {}

    t = time.perf_counter()
    belgeler = derlem_on_isle(metinler, OnIslemeAyarlari(lemmatizasyon=lemma))
    sure["ön işleme"] = time.perf_counter() - t
    bellek = {"veri yüklendi": round(rss0, 1)}
    bellek["ön işleme sonrası"] = round(rss_mb(), 1)
    ozet = derlem_ozeti(belgeler)

    t = time.perf_counter(); frekans = sozcuk_sikligi(belgeler); sure["sözcük sıklığı"] = time.perf_counter() - t
    t = time.perf_counter(); ngram = ngram_sikligi(belgeler, 2, 2); sure["2-gram"] = time.perf_counter() - t
    t = time.perf_counter(); tfidf = tfidf_hesapla(belgeler); sure["TF-IDF"] = time.perf_counter() - t
    t = time.perf_counter()
    kelime_bulutu_png(dict(zip(frekans["Sözcük"], frekans["Sıklık"])), en_fazla=100)
    sure["kelime bulutu"] = time.perf_counter() - t
    bellek["analizler sonrası"] = round(rss_mb(), 1)
    bellek["tepe (Excel öncesi)"] = round(tepe_mb(), 1)
    t = time.perf_counter()
    excel_bayt({"Sözcük": frekans, "N-gram": ngram, "TF-IDF": tfidf.derlem_tablosu, "Belge": tfidf.belge_tablosu})
    sure["Excel (sonuç tabloları)"] = time.perf_counter() - t
    t = time.perf_counter(); csv_bayt(frekans); sure["CSV"] = time.perf_counter() - t

    tepe = tepe_mb()
    return {
        "boyut": ad, "lemmatizasyon": lemma, "belge": belge,
        "sözcük (ön işleme öncesi)": ozet["Toplam sözcük (ön işleme öncesi)"],
        "farklı sözcük": ozet["Farklı sözcük sayısı"],
        "metin MB": round(sum(len(m.encode("utf-8")) for m in metinler) / 2**20, 2),
        "süreler_sn": {k: round(v, 2) for k, v in sure.items()},
        "toplam_sn": round(sum(sure.values()), 2),
        "bellek_MB": bellek,
        "tepe RSS MB": round(tepe, 1),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tek":
        print(json.dumps(tek_olcum(sys.argv[2], sys.argv[3] == "1"), ensure_ascii=False))
        sys.exit(0)
    secilen = sys.argv[1:] or list(BOYUTLAR)
    sonuclar = []
    for ad in secilen:
        for lemma in (False, True):
            cikti = subprocess.run([sys.executable, __file__, "--tek", ad, "1" if lemma else "0"],
                                   capture_output=True, text=True, check=True).stdout
            r = json.loads(cikti.strip().splitlines()[-1])
            sonuclar.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    with open(os.path.join(KOK, "degerlendirme", "performans_sonuclari.json"), "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=1)
