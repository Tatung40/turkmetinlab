"""2.5 staging — GERÇEK E5 ONNX FP32 ile entegrasyon testi (GitHub Actions'ta çalışır; model indirmez).

1. Uyum (20 belge): uygulamanın kodlayıcısı (tmlab.anlamsal) ile fizibilitede doğrulanan kodlayıcı
   (fizibilite_2.5/onnx/e5_onnx_kodlayici.py) aynı vektörleri ve sıralamaları üretmeli. Aksi halde çıkış 1.
2. Ölçek (100 ve 500 belge; her biri ayrı alt süreçte): yükleme ve kodlama süresi, tepe RAM, 3 eşzamanlı oturum,
   uygulamanın varsayılan TF-IDF'si + E5 + hibrit ile görev A/B ölçütleri (fizibilite protokolü).
3. Arayüz (AppTest): staging uygulaması gerçek modelle 100 ve 500 belgelik CSV'de uçtan uca; istisna olmamalı.

    python staging/entegrasyon_testi.py --onnx-klasor DIR --veri TRAll.txt --fizibilite fizibilite_2.5 --cikti cikti
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
from tmlab import anlamsal as an  # noqa: E402  (uygulamanın tmlab'ı, fizibilitenin vendor kopyasından önce)
from tmlab.analysis import tfidf_hesapla  # noqa: E402
from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle  # noqa: E402

REFERANS_B = an.GECERLI_SONUCLAR  # ONNX #3, n=500, görev B nDCG@5


def _tepe():
    with open("/proc/self/status") as f:
        return next(round(int(x.split()[1]) / 1024, 1) for x in f if x.startswith("VmHWM:"))


def _fizibilite(yol):
    sys.path.append(yol)
    sys.path.append(os.path.join(yol, "onnx"))
    import anlamsal_karsilastirma as ak
    from e5_onnx_kodlayici import E5Onnx
    return ak, E5Onnx


def uyum(args):
    ak, E5Onnx = _fizibilite(args.fizibilite)
    basliklar, ozetler, _, _ = ak.veri_oku(args.veri, 20)
    ref, uyg = E5Onnx(args.onnx_klasor, "fp32", 512), an.E5Kodlayici(args.onnx_klasor)
    sonuc = {}
    for ad, onek, metin in (("simetrik", "query: ", ozetler), ("belge", "passage: ", ozetler),
                            ("sorgu", "query: ", basliklar)):
        a, b = ref.encode([onek + m for m in metin]), uyg.kodla(metin, onek)
        sonuc[ad] = {"kosinus_en_kucuk": round(float((a * b).sum(1).min()), 6)}
    S1 = ref.encode(["query: " + b for b in basliklar]) @ ref.encode(["passage: " + o for o in ozetler]).T
    S2 = uyg.kodla(basliklar, "query: ") @ uyg.kodla(ozetler, "passage: ").T
    sonuc["ilk5_ayni"] = bool(all((np.argsort(-S1[i])[:5] == np.argsort(-S2[i])[:5]).all() for i in range(20)))
    sonuc["gecti"] = sonuc["ilk5_ayni"] and min(v["kosinus_en_kucuk"] for k, v in sonuc.items()
                                                if isinstance(v, dict)) >= 0.9999
    return sonuc


def olcek(args):
    """Tek bir n için (ayrı süreçte çağrılır)."""
    ak, _ = _fizibilite(args.fizibilite)
    basliklar, ozetler, anahtar, _ = ak.veri_oku(args.veri, args.n)
    b = {"n": args.n, "tepe_baslangic_MB": _tepe()}
    t = time.perf_counter()
    kod = an.E5Kodlayici(args.onnx_klasor)
    b["yukleme_sn"], b["tepe_yukleme_MB"] = round(time.perf_counter() - t, 2), _tepe()
    u = kod.belirtec_sayilari(ozetler)
    b["sinir_asan_belge"] = int((u > kod.azami_token).sum())
    t = time.perf_counter()
    sim, gec = kod.kodla(ozetler, an.SORGU_ONEKI), kod.kodla(ozetler, an.BELGE_ONEKI)
    b["kodlama_2xn_sn"] = round(time.perf_counter() - t, 2)
    q = kod.kodla(basliklar, an.SORGU_ONEKI)
    b["tepe_tek_oturum_MB"] = _tepe()
    belgeler = derlem_on_isle(ozetler, OnIslemeAyarlari())  # uygulamanın varsayılanı = fizibilite tfidf_tml
    tf = tfidf_hesapla(belgeler, [str(i) for i in range(args.n)], 1, True, False, 1)
    assert tf.analiz_edilen_belge == args.n
    M = tf.matris
    SB = {"TF-IDF": (M @ M.T).toarray(), "E5": sim @ sim.T}
    for S in SB.values():
        np.fill_diagonal(S, -np.inf)
    SB["Hibrit"] = an.hibrit(SB["TF-IDF"], SB["E5"])
    QT = np.array([an.tfidf_sorgu_vektoru(M, tf.terimler, d.sozcukler, True, False)
                   for d in derlem_on_isle(basliklar, OnIslemeAyarlari())])
    SA = {"TF-IDF": np.asarray((M @ QT.T).T), "E5": q @ gec.T}
    SA["Hibrit"] = an.hibrit(SA["TF-IDF"], SA["E5"])
    RA, RB = np.eye(args.n, dtype=np.int16), ak.ilgililik_b(anahtar)
    b["olcutler"] = {g: {y: {o: round(float(v.mean()), 4) for o, v in ak.siralama_olcutleri(S[y], R).items()}
                         for y in S} for g, S, R in (("A", SA, RA), ("B", SB, RB))}
    hatalar = []

    def oturum():
        try:
            kod.kodla(ozetler[:100], an.BELGE_ONEKI)
        except Exception as e:  # pragma: no cover
            hatalar.append(repr(e))
    t = time.perf_counter()
    ip = [threading.Thread(target=oturum) for _ in range(3)]
    [x.start() for x in ip]
    [x.join() for x in ip]
    b["eszamanli_3x100_sn"], b["eszamanli_hata"], b["tepe_MB"] = round(time.perf_counter() - t, 2), hatalar, _tepe()
    print("SONUC " + json.dumps(b, ensure_ascii=False))


def arayuz(args, n):
    import pandas as pd
    from streamlit.testing.v1 import AppTest

    ak, _ = _fizibilite(args.fizibilite)
    basliklar, ozetler, _, _ = ak.veri_oku(args.veri, n)
    os.environ.update({"TML_E5_ONNX_YEREL": args.onnx_klasor})
    at = AppTest.from_file(os.path.join(KOK, "staging", "streamlit_app.py"), default_timeout=900)
    at.run()
    at.radio(key="kaynak").set_value("Kendi dosyamı yükle").run()
    at.file_uploader[0].set_value(("dergipark.csv", pd.DataFrame({"baslik": basliklar, "ozet": ozetler})
                                   .to_csv(index=False).encode(), "text/csv")).run()
    t = time.perf_counter()
    for anahtar in ("e5_model_hazirla", "e5_hesapla"):
        d = next((x for x in at.button if x.key == anahtar), None)
        if d:
            d.click().run()
    at.text_input(key="e5_sorgu").input(basliklar[0]).run()
    sekme = next(x for x in at.tabs if x.label.startswith("🧪"))
    return {"n": n, "sn": round(time.perf_counter() - t, 1), "istisna": [str(e.value)[:300] for e in at.exception],
            "hata": [str(e.value)[:300] for e in sekme.error], "tablo": len(sekme.dataframe),
            "gecti": not at.exception and not sekme.error and len(sekme.dataframe) >= 2}


def ana():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-klasor", required=True)
    p.add_argument("--veri", required=True)
    p.add_argument("--fizibilite", default=os.path.join(os.path.dirname(KOK), "fizibilite_2.5"))
    p.add_argument("--cikti", default="cikti")
    p.add_argument("--n", type=int)
    p.add_argument("--asama", choices=["olcek"], help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.asama == "olcek":
        return olcek(args)
    os.makedirs(args.cikti, exist_ok=True)
    rapor = {"uyari": "GitHub Actions makinesinde ölçüldü; Streamlit sunucu ölçümü değildir."}
    rapor["uyum_20"] = uyum(args)
    print(json.dumps(rapor["uyum_20"], ensure_ascii=False), flush=True)
    rapor["olcek"] = []
    for n in (100, 500):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--asama", "olcek", "--n", str(n), "--onnx-klasor",
                            args.onnx_klasor, "--veri", args.veri, "--fizibilite", args.fizibilite],
                           capture_output=True, text=True, timeout=1800)
        satir = [s for s in r.stdout.splitlines() if s.startswith("SONUC ")]
        rapor["olcek"].append(json.loads(satir[-1][6:]) if satir else {"n": n, "hata": r.stderr[-800:]})
    rapor["arayuz"] = [arayuz(args, n) for n in (100, 500)]
    b500 = next((o for o in rapor["olcek"] if o.get("n") == 500 and "olcutler" in o), None)
    if b500:
        rapor["referans_farki_B500"] = {y: round(b500["olcutler"]["B"][y]["ndcg@5"] - REFERANS_B[y], 4) for y in REFERANS_B}
    # n=500 görev B: uygulama hattı ONNX #3 sonucunu yeniden üretmeli (3 ondalık yuvarlama + küçük sayısal pay)
    rapor["referans_uyumu"] = bool(b500) and all(abs(v) <= 0.002 for v in rapor.get("referans_farki_B500", {}).values())
    gecti = rapor["uyum_20"]["gecti"] and all("olcutler" in o for o in rapor["olcek"]) and all(
        a["gecti"] for a in rapor["arayuz"]) and rapor["referans_uyumu"]
    rapor["gecti"] = bool(gecti)
    with open(os.path.join(args.cikti, "staging_entegrasyon.json"), "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.cikti, "staging_entegrasyon.md"), "w", encoding="utf-8") as f:
        f.write(md(rapor))
    print(md(rapor))
    sys.exit(0 if gecti else 1)


def md(r):
    s = ["# 2.5 staging entegrasyon testi (gerçek E5 ONNX FP32)", "", r["uyari"], "",
         f"**Sonuç:** {'GEÇTİ' if r['gecti'] else 'KALDI'}", "",
         f"- Uygulama ↔ fizibilite kodlayıcısı (20 belge): {r['uyum_20']}", ""]
    s += ["| n | Yükleme sn | 2×n kodlama sn | Tepe yükleme MB | Tepe tek oturum MB | Tepe (3 eşzamanlı) MB | "
          "3×100 sn | B nDCG@5 TF-IDF / E5 / Hibrit | A nDCG@5 TF-IDF / E5 / Hibrit |", "|---|---|---|---|---|---|---|---|---|"]
    for o in r["olcek"]:
        if "olcutler" not in o:
            s.append(f"| {o['n']} | HATA | | | | | | | |")
            continue
        f = lambda g: " / ".join(f"{o['olcutler'][g][y]['ndcg@5']:.3f}" for y in ("TF-IDF", "E5", "Hibrit"))  # noqa
        s.append(f"| {o['n']} | {o['yukleme_sn']} | {o['kodlama_2xn_sn']} | {o['tepe_yukleme_MB']} | "
                 f"{o['tepe_tek_oturum_MB']} | {o['tepe_MB']} | {o['eszamanli_3x100_sn']} | {f('B')} | {f('A')} |")
    if "referans_farki_B500" in r:
        s += ["", f"- n=500 görev B, ONNX #3 referansına göre fark: {r['referans_farki_B500']}"]
    s += ["", "| Arayüz (AppTest) n | Süre sn | Geçti | Hata |", "|---|---|---|---|"]
    s += [f"| {a['n']} | {a['sn']} | {a['gecti']} | {'; '.join(a['istisna'] + a['hata'])[:200]} |" for a in r["arayuz"]]
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    ana()
