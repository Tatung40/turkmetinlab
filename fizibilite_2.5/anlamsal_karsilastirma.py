"""TürkMetinLab 2.5 fizibilite: TF-IDF ile iki çok dilli gömme (embedding) modelinin karşılaştırılması.

Yalnızca CPU. Her yöntem ayrı bir alt süreçte çalışır; böylece tepe bellek (VmHWM) ölçümleri birbirini etkilemez.

    python fizibilite_2.5/anlamsal_karsilastirma.py                       # varsayılan: n=500, bütün yöntemler
    python fizibilite_2.5/anlamsal_karsilastirma.py --n 100 --yontemler tfidf_tml,e5_ozgun
    python fizibilite_2.5/anlamsal_karsilastirma.py --veri TRAll.txt      # veri dosyası elde varsa

Görevler (etiketler insan kaynaklıdır; otomatik ya da sentetik değildir):
  A  başlık → özet: her makalenin BAŞLIĞI sorgu, aynı makalenin ÖZETİ tek ilgili belgedir (yayıncı eşleşmesi).
  B  özet → özet: ilgililik = iki makalenin YAZAR anahtar sözcüklerinden ortak olanların sayısı (derecelendirilmiş,
     dolaylı ilgililik; insan yargısıyla etiketlenmiş ilgililik değildir).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TürkMetinLab 2.1 ön işlemesinin değiştirilmemiş kopyası (depodaki uygulama sürümünden bağımsız)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

DERGIPARK_URL = "https://raw.githubusercontent.com/humakilicunlu/DergiParkEn/main/TRAll.txt"
TOHUM = 2501
MODELLER = {
    # Önekler model kartlarına göre: E5 asimetrik erişimde "query: " / "passage: ", simetrik görevde iki taraf "query: ".
    "minilm": {"ad": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "lisans": "Apache-2.0",
               "sorgu": "", "belge": "", "simetrik": ""},
    "e5": {"ad": "intfloat/multilingual-e5-small", "lisans": "MIT",
           "sorgu": "query: ", "belge": "passage: ", "simetrik": "query: "},
}
VARSAYILAN = ["tfidf_ham", "tfidf_tml", "tfidf_tml_terim", "tfidf_tml_lemma",
              "minilm_ozgun", "minilm_temiz", "e5_ozgun", "e5_temiz"]
REFERANS = "tfidf_tml"
PRATIK_ESIK = 0.03  # önceden belirlenmiş en küçük anlamlı fark (nDCG@5 / MRR)


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------

def _kucuk(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def veri_oku(yol: str | None, n: int):
    """DergiPark Türkçe özetleri: (başlıklar, özetler, anahtar sözcük kümeleri). Sabit tohumla n belge seçilir."""
    if yol:
        ham = open(yol, encoding="utf-8").read()
    else:
        with urllib.request.urlopen(DERGIPARK_URL, timeout=60) as y:
            ham = y.read().decode("utf-8")
    kayit, gorulen = [], set()
    for satir in ham.splitlines():
        a = satir.split("\t")
        if len(a) != 4:
            continue
        baslik, ozet = a[1].strip().strip('"'), a[2].strip().strip('"')
        anahtar = {_kucuk(k).strip(" .\"'") for k in a[3].replace(";", ",").split(",")}
        anahtar = {k for k in anahtar if k}
        if not baslik or len(ozet.split()) < 30 or not anahtar or ozet in gorulen:
            continue
        gorulen.add(ozet)
        kayit.append((baslik, ozet, anahtar))
    if n > len(kayit):
        raise SystemExit(f"Yeterli belge yok: {len(kayit)} < {n}")
    secim = sorted(random.Random(TOHUM).sample(range(len(kayit)), n))
    k = [kayit[i] for i in secim]
    return [x[0] for x in k], [x[1] for x in k], [x[2] for x in k], len(kayit)


def ilgililik_b(anahtar):
    n = len(anahtar)
    R = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        for j in range(n):
            if i != j:
                R[i, j] = len(anahtar[i] & anahtar[j])
    return R


# ---------------------------------------------------------------------------
# Ölçütler
# ---------------------------------------------------------------------------

def siralama_olcutleri(S: np.ndarray, R: np.ndarray, k: int = 5):
    """Her sorgu için: isabet@1, isabet@5, ters sıra (ilk ilgili), nDCG@k (derecelendirilmiş).

    Eşit skorlarda belge sırası (indeks) kullanılır; sonuçlar belirlenimcidir.
    """
    sonuc = {"isabet@1": [], "isabet@5": [], "mrr": [], "ndcg@5": []}
    indeks = np.arange(S.shape[1])
    for q in range(S.shape[0]):
        if R[q].max() <= 0:
            continue
        sira = np.lexsort((indeks, -S[q]))
        rel = R[q, sira]
        ilk = int(np.argmax(rel > 0)) + 1
        kazanc = (2.0 ** rel[:k] - 1) / np.log2(np.arange(2, k + 2))
        ideal = np.sort(R[q])[::-1][:k]
        ideal_kazanc = ((2.0 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum()
        sonuc["isabet@1"].append(float(ilk <= 1))
        sonuc["isabet@5"].append(float(ilk <= 5))
        sonuc["mrr"].append(1.0 / ilk)
        sonuc["ndcg@5"].append(float(kazanc.sum() / ideal_kazanc))
    return {a: np.array(v) for a, v in sonuc.items()}


def bootstrap_ga(x: np.ndarray, b: int = 2000, tohum: int = TOHUM):
    """Ortalama ve %95 bootstrap güven aralığı (yüzdelik yöntem)."""
    rng = np.random.default_rng(tohum)
    ort = rng.choice(x, size=(b, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(ort, 2.5)), float(np.percentile(ort, 97.5))


# ---------------------------------------------------------------------------
# Bellek
# ---------------------------------------------------------------------------

def _durum(alan):
    with open("/proc/self/status") as f:
        for s in f:
            if s.startswith(alan + ":"):
                return round(int(s.split()[1]) / 1024, 1)
    return None


def _klasor_mb(yol):
    t = 0
    for d, _, dosyalar in os.walk(yol):
        for f in dosyalar:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                t += os.path.getsize(p)
    return round(t / 2**20, 1)


# ---------------------------------------------------------------------------
# Yöntemler (alt süreçte)
# ---------------------------------------------------------------------------

def _tml_belirtecleri(metinler, ayar=None, terimci=None):
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle
    return [b.sozcukler for b in derlem_on_isle(metinler, ayar or OnIslemeAyarlari(), terimci=terimci)]


def tfidf_calistir(yontem, basliklar, ozetler):
    from sklearn.feature_extraction.text import TfidfVectorizer

    from tmlab.preprocess import OnIslemeAyarlari

    t = time.perf_counter()
    if yontem == "tfidf_ham":  # özgün metin: yalnızca Türkçe küçük harf ve sözcüklere bölme
        bol = lambda xs: [re.findall(r"\w+", _kucuk(x)) for x in xs]  # noqa: E731
    else:
        ayar, terimci = OnIslemeAyarlari(), None
        if yontem == "tfidf_tml_lemma":
            ayar = OnIslemeAyarlari(lemmatizasyon=True)
        if yontem == "tfidf_tml_terim":
            from tmlab.terimler import TerimEslestirici, cekirdek_sozluk
            terimci = TerimEslestirici(cekirdek_sozluk())
        bol = lambda xs: _tml_belirtecleri(xs, ayar, terimci)  # noqa: E731
    d = bol(ozetler)
    v = TfidfVectorizer(analyzer=lambda x: x, lowercase=False, smooth_idf=True, norm="l2")
    X = v.fit_transform(d)
    Q = v.transform(bol(basliklar))
    sure = time.perf_counter() - t
    return (Q @ X.T).toarray(), (X @ X.T).toarray(), {"vektor_boyutu": int(X.shape[1]), "kodlama_toplam_sn": round(sure, 2)}


def gomme_calistir(yontem, basliklar, ozetler, onbellek, model_yolu, eszamanli):
    import threading

    anahtar, varyant = yontem.split("_")
    m = MODELLER[anahtar]
    bilgi = {"model": m["ad"], "lisans": m["lisans"], "metin": varyant, "rss_baslangic_MB": _durum("VmRSS")}
    t = time.perf_counter()
    import torch
    from sentence_transformers import SentenceTransformer
    bilgi["ithal_sn"] = round(time.perf_counter() - t, 2)
    bilgi["torch"] = torch.__version__
    bilgi["torch_is_parcacigi"] = torch.get_num_threads()
    t = time.perf_counter()
    model = SentenceTransformer(model_yolu or m["ad"], device="cpu", cache_folder=onbellek)
    bilgi["yukleme_sn"] = round(time.perf_counter() - t, 2)
    bilgi["rss_yukleme_sonrasi_MB"] = _durum("VmRSS")
    bilgi["azami_token"] = int(model.max_seq_length)
    bilgi["vektor_boyutu"] = int(model.get_sentence_embedding_dimension())
    bilgi["parametre"] = int(sum(p.numel() for p in model.parameters()))
    bilgi["moduller"] = [type(x).__name__ for x in model]
    havuz = [x for x in model if type(x).__name__ == "Pooling"]
    if havuz:
        bilgi["havuzlama"] = {a: v for a, v in havuz[0].get_config_dict().items() if v not in (False, None)}
    bilgi["disk_MB"] = _klasor_mb(model_yolu or onbellek)
    if varyant == "temiz":
        basliklar = [" ".join(x) for x in _tml_belirtecleri(basliklar)]
        ozetler = [" ".join(x) for x in _tml_belirtecleri(ozetler)]

    def kodla(xs, onek):
        return model.encode([onek + x for x in xs], batch_size=32, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)

    tok = model.tokenizer
    uzunluk = np.array([len(tok(m["belge"] + x)["input_ids"]) for x in ozetler])
    bilgi["token_uzunlugu_ortanca"] = int(np.median(uzunluk))
    bilgi["kesilen_belge_orani"] = round(float((uzunluk > model.max_seq_length).mean()), 4)
    bilgi["kesilen_token_orani"] = round(float(np.clip(uzunluk - model.max_seq_length, 0, None).sum() / uzunluk.sum()), 4)
    kodla(ozetler[:2], m["belge"])  # ısınma
    t = time.perf_counter()
    D100 = kodla(ozetler[:100], m["belge"])
    bilgi["kodlama_100_sn"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    D = np.vstack([D100, kodla(ozetler[100:], m["belge"])]) if len(ozetler) > 100 else D100
    bilgi["kodlama_toplam_sn"] = round(bilgi["kodlama_100_sn"] + time.perf_counter() - t, 2)
    Dsim = kodla(ozetler, m["simetrik"]) if m["simetrik"] != m["belge"] else D  # E5: simetrik görev için "query: "
    Q = kodla(basliklar, m["sorgu"])
    bilgi["rss_kodlama_sonrasi_MB"] = _durum("VmRSS")
    bilgi["tepe_tek_oturum_MB"] = _durum("VmHWM")
    if eszamanli > 1:  # Streamlit'te model paylaşılır (cache_resource); her oturum kendi metnini kodlar
        hatalar = []

        def is_():
            try:
                kodla(ozetler[:100], m["belge"])
            except Exception as e:  # pragma: no cover
                hatalar.append(repr(e))
        t = time.perf_counter()
        iplikler = [threading.Thread(target=is_) for _ in range(eszamanli)]
        [x.start() for x in iplikler]
        [x.join() for x in iplikler]
        bilgi[f"eszamanli_{eszamanli}x100_sn"] = round(time.perf_counter() - t, 2)
        bilgi["eszamanli_hata"] = hatalar
    bilgi["tepe_MB"] = _durum("VmHWM")
    return Q @ D.T, Dsim @ Dsim.T, bilgi


def tek(args):
    basliklar, ozetler, anahtar, _ = veri_oku(args.veri, args.n)
    t0 = time.perf_counter()
    if args.tek.startswith("tfidf"):
        SA, SB, bilgi = tfidf_calistir(args.tek, basliklar, ozetler)
        bilgi["tepe_MB"] = _durum("VmHWM")
    else:
        model_yolu = dict(x.split("=", 1) for x in args.model_yolu.split(",") if x).get(args.tek.split("_")[0])
        SA, SB, bilgi = gomme_calistir(args.tek, basliklar, ozetler, args.onbellek, model_yolu, args.eszamanli)
    np.fill_diagonal(SB, -np.inf)
    bilgi["toplam_sn"] = round(time.perf_counter() - t0, 2)
    np.save(os.path.join(args.gecici, f"{args.tek}_A.npy"), SA.astype(np.float32))
    np.save(os.path.join(args.gecici, f"{args.tek}_B.npy"), SB.astype(np.float32))
    print("SONUC " + json.dumps(bilgi, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Ana süreç: çalıştır, ölç, karşılaştır, raporla
# ---------------------------------------------------------------------------

def _znorm(S):
    S = np.where(np.isfinite(S), S, np.nan)
    m, s = np.nanmean(S, axis=1, keepdims=True), np.nanstd(S, axis=1, keepdims=True) + 1e-9
    return np.nan_to_num((S - m) / s, nan=-1e9)


def ana():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--veri", help="DergiPark TRAll.txt yolu (yoksa GitHub'dan indirilir; diske yazılmaz)")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--yontemler", default=",".join(VARSAYILAN))
    p.add_argument("--eszamanli", type=int, default=3, help="eşzamanlı oturum benzetimi (iş parçacığı sayısı)")
    p.add_argument("--onbellek", default=os.path.expanduser("~/.cache/turkmetinlab_modeller"),
                   help="model dosyaları için önbellek (yalnızca bir kez indirilir; kullanıcı metni içermez)")
    p.add_argument("--model-yolu", default="", help="çevrimdışı/deneme: minilm=/yol,e5=/yol")
    p.add_argument("--cikti", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonuclar"))
    p.add_argument("--etiket", default="", help="sonuç dosyası adına eklenir (ör. sahte_model_testi)")
    p.add_argument("--azami-sure", type=int, default=1800, help="yöntem başına süre sınırı (sn); aşılırsa durdurulur")
    p.add_argument("--azami-rss-mb", type=int, default=6000, help="yöntem başına bellek sınırı (MB); aşılırsa durdurulur")
    p.add_argument("--kati", action="store_true",
                   help="istenen yöntemlerden biri çalışmaz ya da sınırı aşarsa 2 çıkış koduyla bitir (CI aşamaları için)")
    p.add_argument("--tek", help=argparse.SUPPRESS)
    p.add_argument("--gecici", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.tek:
        return tek(args)

    basliklar, ozetler, anahtar, havuz = veri_oku(args.veri, args.n)
    RA = np.eye(args.n, dtype=np.int16)
    RB = ilgililik_b(anahtar)
    os.makedirs(args.cikti, exist_ok=True)
    sonuc = {"veri": {"kaynak": "DergiPark Türkçe özetleri (humakilicunlu/DergiParkEn, TRAll.txt)",
                      "uygun_benzersiz_belge": havuz, "orneklem": args.n, "tohum": TOHUM,
                      "gorev_A_sorgu": int(args.n),
                      "gorev_B_sorgu (en az bir ilgili belgesi olan)": int((RB.max(axis=1) > 0).sum()),
                      "gorev_B_ortalama_ilgili_belge": round(float((RB > 0).sum(axis=1)[RB.max(axis=1) > 0].mean()), 2),
                      "ozet_sozcuk_ortanca": int(np.median([len(o.split()) for o in ozetler]))},
             "yontemler": {}}
    olcutler = {}
    with tempfile.TemporaryDirectory() as gecici:
        for y in [x for x in args.yontemler.split(",") if x]:
            komut = [sys.executable, os.path.abspath(__file__), "--tek", y, "--n", str(args.n), "--gecici", gecici,
                     "--onbellek", args.onbellek, "--eszamanli", str(args.eszamanli), "--model-yolu", args.model_yolu]
            if args.veri:
                komut += ["--veri", args.veri]
            print(f"… {y}", flush=True)
            kod, cikti, hata, sinir = _sinirli_calistir(komut, args.azami_sure, args.azami_rss_mb)
            satir = [s for s in cikti.splitlines() if s.startswith("SONUC ")]
            if sinir or kod != 0 or not satir:
                neden = sinir or (hata.strip().splitlines() or ["bilinmeyen hata"])[-1][:300]
                sonuc["yontemler"][y] = {"durum": "sınır aşıldı" if sinir else "çalıştırılamadı", "neden": neden}
                print(f"   {sonuc['yontemler'][y]['durum']}: {neden}", flush=True)
                continue
            bilgi = json.loads(satir[-1][6:])
            SA, SB = np.load(os.path.join(gecici, f"{y}_A.npy")), np.load(os.path.join(gecici, f"{y}_B.npy"))
            olcutler[y] = {"A": siralama_olcutleri(SA, RA), "B": siralama_olcutleri(SB, RB)}
            sonuc["yontemler"][y] = {"durum": "çalıştı", **bilgi}
            for g in ("A", "B"):
                sonuc["yontemler"][y][g] = {a: [round(v, 4) for v in bootstrap_ga(x)] for a, x in olcutler[y][g].items()}
        # Hibrit: TF-IDF (TürkMetinLab) + gömme, sorgu başına z-puanlarının ortalaması
        for y in list(olcutler):
            if y.startswith(("minilm", "e5")) and REFERANS in olcutler:
                h = f"hibrit_{REFERANS}+{y}"
                SA = _znorm(np.load(os.path.join(gecici, f"{REFERANS}_A.npy"))) + _znorm(np.load(os.path.join(gecici, f"{y}_A.npy")))
                SB = _znorm(np.load(os.path.join(gecici, f"{REFERANS}_B.npy"))) + _znorm(np.load(os.path.join(gecici, f"{y}_B.npy")))
                np.fill_diagonal(SB, -np.inf)
                olcutler[h] = {"A": siralama_olcutleri(SA, RA), "B": siralama_olcutleri(SB, RB)}
                sonuc["yontemler"][h] = {"durum": "çalıştı (skor birleştirme; ek kaynak ölçülmedi)"}
                for g in ("A", "B"):
                    sonuc["yontemler"][h][g] = {a: [round(v, 4) for v in bootstrap_ga(x)] for a, x in olcutler[h][g].items()}
    # Referansa göre eşleştirilmiş farklar
    if REFERANS in olcutler:
        for y in olcutler:
            if y == REFERANS:
                continue
            sonuc["yontemler"][y]["fark_referans"] = {
                g: {a: [round(v, 4) for v in bootstrap_ga(olcutler[y][g][a] - olcutler[REFERANS][g][a])]
                    for a in ("isabet@1", "isabet@5", "mrr", "ndcg@5")} for g in ("A", "B")}
    ad = f"sonuclar_n{args.n}" + (f"_{args.etiket}" if args.etiket else "")
    with open(os.path.join(args.cikti, ad + ".json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.cikti, ad + ".md"), "w", encoding="utf-8") as f:
        f.write(rapor(sonuc))
    print(rapor(sonuc))
    basarisiz = [y for y, r in sonuc["yontemler"].items() if not r["durum"].startswith("çalıştı")]
    if args.kati and basarisiz:
        print(f"DURDURULDU: {', '.join(basarisiz)} başarısız; sonraki aşama çalıştırılmamalı.", flush=True)
        sys.exit(2)


def _sinirli_calistir(komut, azami_sure, azami_rss_mb):
    """Alt süreci süre ve bellek (RSS) sınırıyla çalıştırır; yeniden deneme yapmaz."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen(komut, stdout=out, stderr=err, text=True)
        bas, sinir = time.time(), None
        while p.poll() is None:
            time.sleep(0.5)
            try:
                with open(f"/proc/{p.pid}/status") as f:
                    rss = next((int(x.split()[1]) / 1024 for x in f if x.startswith("VmRSS:")), 0)
            except OSError:
                rss = 0
            if rss > azami_rss_mb:
                sinir = f"bellek sınırı aşıldı ({rss:.0f} MB > {azami_rss_mb} MB)"
            elif time.time() - bas > azami_sure:
                sinir = f"süre sınırı aşıldı (> {azami_sure} sn)"
            if sinir:
                p.kill()
                p.wait()
                break
        out.seek(0)
        err.seek(0)
        return p.returncode, out.read(), err.read(), sinir


def rapor(s) -> str:
    v = s["veri"]
    out = [f"# Karşılaştırma sonuçları (n = {v['orneklem']})", "",
           f"Veri: {v['kaynak']}; uygun benzersiz belge {v['uygun_benzersiz_belge']}, örneklem {v['orneklem']} "
           f"(tohum {v['tohum']}); özet ortanca {v['ozet_sozcuk_ortanca']} sözcük. Görev B sorgu: "
           f"{v['gorev_B_sorgu (en az bir ilgili belgesi olan)']} (ortalama {v['gorev_B_ortalama_ilgili_belge']} ilgili belge).",
           "", "Değerler: ortalama [%95 bootstrap GA]. Fark sütunu: yöntem − tfidf_tml (eşleştirilmiş).", ""]
    for g, baslik in (("A", "Görev A — başlık → kendi özeti (tek ilgili belge)"),
                      ("B", "Görev B — özet → özet, ortak yazar anahtar sözcüğü (derecelendirilmiş)")):
        out += [f"## {baslik}", "", "| Yöntem | İsabet@1 | İsabet@5 | MRR | nDCG@5 | Fark nDCG@5 |", "|---|---|---|---|---|---|"]
        for y, r in s["yontemler"].items():
            if g not in r:
                out.append(f"| {y} | {r['durum']}: {r.get('neden', '')[:80]} | | | | |")
                continue
            h = lambda a: f"{r[g][a][0]:.3f} [{r[g][a][1]:.3f}, {r[g][a][2]:.3f}]"  # noqa: E731
            fk = r.get("fark_referans", {}).get(g, {}).get("ndcg@5")
            fs = "—" if not fk else f"{fk[0]:+.3f} [{fk[1]:+.3f}, {fk[2]:+.3f}]"
            out.append(f"| {y} | {h('isabet@1')} | {h('isabet@5')} | {h('mrr')} | {h('ndcg@5')} | {fs} |")
        out.append("")
    out += ["## Kaynak ölçümleri (bu makinede, CPU)", "",
            "| Yöntem | Yükleme sn | 100 belge sn | Tüm belgeler sn | Tepe RAM MB | Vektör boyutu | Azami token | "
            "Kesilen belge | Disk MB | Eşzamanlı |", "|---|---|---|---|---|---|---|---|---|---|"]
    for y, r in s["yontemler"].items():
        if r.get("durum") != "çalıştı":
            continue
        es = next((f"{k}: {x} sn" for k, x in r.items() if k.startswith("eszamanli_") and k.endswith("_sn")), "—")
        out.append(f"| {y} | {r.get('yukleme_sn', '—')} | {r.get('kodlama_100_sn', '—')} | {r.get('kodlama_toplam_sn', '—')} | "
                   f"{r.get('tepe_MB', '—')} | {r.get('vektor_boyutu', '—')} | {r.get('azami_token', '—')} | "
                   f"{r.get('kesilen_belge_orani', '—')} | {r.get('disk_MB', '—')} | {es} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ana()
