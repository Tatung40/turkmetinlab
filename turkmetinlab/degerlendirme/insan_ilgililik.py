"""İnsan ilgililik puanlarıyla puanlayıcı uyumu ve TF-IDF / E5 / hibrit sıralama doğrulaması.

Kullanım (turkmetinlab/ dizininde; çıktıyı depo DIŞINA yazın):
    python degerlendirme/insan_ilgililik.py P1.xlsx P2.xlsx P3.xlsx --cikti ~/ozel/ilgililik_raporu.xlsx

Girdi: puanlayıcı başına bir Excel; sütunlar 'Sorgu özeti', 'Aday özeti', 'Puan (0/1/2)'.
Satır sırası ve kayıt kodları puanlayıcılar arasında farklı olabilir; çiftler normalize metin özetiyle
(SHA-256) eşleştirilir. Eksik puan doldurulmaz. Gerçek veri ve rapor depoya eklenmemelidir.

Yöntem:
- Cohen kappa (ağırlıksız, doğrusal ağırlıklı; scikit-learn), çiftler için tam gözlemlerle.
- Krippendorff alfa (ordinal/nominal/aralık): `krippendorff` paketi varsa o, yoksa aynı tanımın
  çakışma matrisi uygulaması (Krippendorff 2011 örneğiyle sınanır). Eksik puanlar birim içinde atlanır.
- Sıralama: her sorgunun kendi aday listesi içinde. Kazanç = puanlayıcı ortalaması; ikili ilgililik =
  medyan >= 1 (sonuçlar görülmeden sabitlenmiş eşik). Spearman, nDCG@5/10 (scikit-learn), P@5, AP, RR
  sorgu başına hesaplanır ve sorgular üzerinden ortalanır; çiftler bağımsız gözlem sayılmaz.
- Hibrit: uygulamadaki sabit eşit ağırlık (anlamsal.hibrit); veriyle ağırlık ayarlanmaz.
- E5 modeli yapılandırılmamış/erişilemiyorsa E5 ve hibrit NOT_RUN olarak raporlanır.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import unicodedata
from itertools import combinations

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

SUTUNLAR = ("Sorgu özeti", "Aday özeti", "Puan (0/1/2)")
KATEGORILER = (0, 1, 2)
NOT_RUN = "NOT_RUN"


def anahtar(metin) -> str:
    norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(metin or ""))).strip().casefold()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def puanlayici_oku(yol: str) -> pd.DataFrame:
    t = pd.read_excel(yol, dtype=object)
    eksik = [s for s in SUTUNLAR if s not in t.columns]
    if eksik:
        raise ValueError(f"{os.path.basename(yol)}: eksik sütun {eksik}")
    t = t[t[list(SUTUNLAR)].notna().any(axis=1)]
    puan = pd.to_numeric(t["Puan (0/1/2)"], errors="coerce")
    gecersiz = puan.notna() & ~puan.isin(KATEGORILER)
    if gecersiz.any():
        raise ValueError(f"{os.path.basename(yol)}: 0/1/2 dışında puan: {sorted(puan[gecersiz].unique())}")
    return pd.DataFrame({"sorgu": t["Sorgu özeti"].map(anahtar), "aday": t["Aday özeti"].map(anahtar),
                         "puan": puan.to_numpy(dtype=float), "sorgu_metni": t["Sorgu özeti"],
                         "aday_metni": t["Aday özeti"],
                         "aday_basligi": t.get("Aday tez başlığı", pd.Series(index=t.index, dtype=object))})


def birlestir(tablolar: dict) -> tuple:
    """Puanlayıcı × çift matrisi (nan = eksik) ve bütünlük tablosu. Çift kümeleri aynı olmalıdır."""
    butunluk, kumeler = [], {}
    for ad, t in tablolar.items():
        ciftler = list(zip(t["sorgu"], t["aday"]))
        kumeler[ad] = set(ciftler)
        butunluk.append({"Puanlayıcı": ad, "Satır": len(t), "Benzersiz çift": len(kumeler[ad]),
                         "Yinelenen çift": len(ciftler) - len(kumeler[ad]), "Eksik puan": int(t["puan"].isna().sum()),
                         "Sorgu": t["sorgu"].nunique(),
                         **{f"Puan {k}": int((t["puan"] == k).sum()) for k in KATEGORILER}})
    if any(b["Yinelenen çift"] for b in butunluk):
        raise ValueError("Aynı sorgu–aday çifti bir dosyada birden çok kez puanlanmış.")
    ilk = next(iter(kumeler.values()))
    if any(k != ilk for k in kumeler.values()):
        raise ValueError("Puanlayıcıların sorgu–aday çift kümeleri farklı; eşleştirme yapılamaz.")
    ciftler = sorted(ilk)
    M = np.array([[dict(zip(zip(t["sorgu"], t["aday"]), t["puan"]))[c] for c in ciftler]
                  for t in tablolar.values()], dtype=float)
    return M, ciftler, pd.DataFrame(butunluk)


def alfa_cakisma(M: np.ndarray, duzey: str = "ordinal") -> float:
    """Krippendorff alfa, çakışma matrisiyle (Krippendorff 2011). M: puanlayıcı × birim, nan = eksik."""
    degerler = np.unique(M[~np.isnan(M)])
    ix = {v: i for i, v in enumerate(degerler)}
    k = len(degerler)
    o = np.zeros((k, k))
    for birim in M.T:
        birim = birim[~np.isnan(birim)]
        m = len(birim)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[ix[birim[a]], ix[birim[b]]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    d = np.zeros((k, k))
    for c in range(k):
        for e in range(k):
            if duzey == "nominal":
                d[c, e] = c != e
            elif duzey == "interval":
                d[c, e] = (degerler[c] - degerler[e]) ** 2
            else:
                lo, hi = min(c, e), max(c, e)
                d[c, e] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[e]) / 2) ** 2
    beklenen = (np.outer(n_c, n_c) * d).sum() / (n * (n - 1))
    return float(1 - (o * d).sum() / n / beklenen)


def alfa(M: np.ndarray, duzey: str = "ordinal") -> tuple:
    try:
        import krippendorff
        return float(krippendorff.alpha(M, level_of_measurement=duzey)), "krippendorff paketi"
    except ImportError:
        return alfa_cakisma(M, duzey), "çakışma matrisi (yerleşik)"


def uyum(M: np.ndarray, adlar: list, sorgular: np.ndarray, tekrar: int = 2000, tohum: int = 42) -> dict:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    ciftler, matrisler = [], []
    for a, b in combinations(range(len(adlar)), 2):
        m = ~np.isnan(M[a]) & ~np.isnan(M[b])
        x, y = M[a][m].astype(int), M[b][m].astype(int)
        ciftler.append({"Çift": f"{adlar[a]}–{adlar[b]}", "n (tam)": int(m.sum()), "Tam uyum": round(float(np.mean(x == y)), 4),
                        "Cohen kappa": round(cohen_kappa_score(x, y, labels=list(KATEGORILER)), 4),
                        "Doğrusal ağırlıklı kappa": round(cohen_kappa_score(x, y, labels=list(KATEGORILER), weights="linear"), 4)})
        cm = confusion_matrix(x, y, labels=list(KATEGORILER))
        matrisler.append(pd.DataFrame(cm, index=[f"{adlar[a]}={k}" for k in KATEGORILER],
                                      columns=[f"{adlar[b]}={k}" for k in KATEGORILER]))
    alfalar = []
    for duzey in ("ordinal", "nominal", "interval"):
        deger, kaynak = alfa(M, duzey)
        alfalar.append({"Ölçek": duzey, "Alfa (mevcut gözlemler)": round(deger, 4), "Kaynak": kaynak,
                        "Bağımsız çakışma matrisi": round(alfa_cakisma(M, duzey), 4)})
    tam = ~np.isnan(M).any(axis=0)
    rng = np.random.default_rng(tohum)
    us = np.unique(sorgular)
    boot = []
    for _ in range(tekrar):  # sorgu düzeyinde küme bootstrap: çiftler bağımsız sayılmaz
        sec = np.concatenate([np.flatnonzero(sorgular == q) for q in rng.choice(us, len(us), replace=True)])
        if len(np.unique(M[:, sec][~np.isnan(M[:, sec])])) > 1:
            boot.append(alfa_cakisma(M[:, sec], "ordinal"))
    farkli = [len(set(s)) for s in M[:, tam].T]
    ozet = {"Ordinal alfa (tam durum)": round(alfa(M[:, tam], "ordinal")[0], 4), "Tam durum birim": int(tam.sum()),
            "Ordinal alfa sorgu-bootstrap %2.5": round(float(np.percentile(boot, 2.5)), 4) if boot else None,
            "Ordinal alfa sorgu-bootstrap %97.5": round(float(np.percentile(boot, 97.5)), 4) if boot else None,
            "Sorgu sayısı": int(len(us)), "Üçü aynı": farkli.count(1),
            "Çoğunluk (2 aynı)": farkli.count(2), "Hepsi farklı": sum(f >= 3 for f in farkli)}
    return {"ciftler": pd.DataFrame(ciftler), "alfa": pd.DataFrame(alfalar), "ozet": ozet, "matrisler": matrisler}


def siralama_olcutleri(skor: np.ndarray, kazanc: np.ndarray, ilgili: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import ndcg_score

    sira = np.argsort(-skor, kind="stable")
    isabet = ilgili[sira]
    kesinlik = [isabet[:i + 1].mean() for i in range(len(isabet)) if isabet[i]]
    return {"Spearman": float(spearmanr(skor, kazanc).statistic), "nDCG@5": float(ndcg_score([kazanc], [skor], k=5)),
            "nDCG@10": float(ndcg_score([kazanc], [skor], k=10)), "P@5": float(isabet[:5].mean()),
            "AP": float(np.mean(kesinlik)) if kesinlik else 0.0,
            "RR": float(1 / (np.flatnonzero(isabet)[0] + 1)) if isabet.any() else 0.0}


def tfidf_skorlari(metinler: dict) -> tuple:
    """Uygulamanın varsayılan ön işleme + TF-IDF + kosinüs zinciri; IDF yalnız bu metin havuzundan kestirilir."""
    from tmlab.analysis import tfidf_hesapla
    from tmlab.benzerlik import kosinus_matrisi
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

    kimlikler = sorted(metinler)
    S = kosinus_matrisi(tfidf_hesapla(derlem_on_isle([metinler[k] for k in kimlikler], OnIslemeAyarlari())).matris)
    return S, {k: i for i, k in enumerate(kimlikler)}


def e5_skorlari(metinler: dict):
    """Yapılandırılmış ve doğrulanmış E5 ONNX modeli varsa skor matrisi; yoksa (None, gerekçe). İndirme yapmaz."""
    try:
        from tmlab import anlamsal as an
        kaynak = an.ModelKaynagi.ortamdan()
        if kaynak.eksik():
            return None, kaynak.eksik()
        klasor = kaynak.yerel or an.onbellek_klasoru(kaynak)
        if not all(os.path.exists(os.path.join(klasor, ad)) for ad in an.DOSYALAR):
            return None, f"Doğrulanmış E5 modeli yerel önbellekte yok ({klasor}); bu betik model indirmez."
        kimlikler = sorted(metinler)
        # Belge–belge benzerliği: uygulamanın "simetrik" kipiyle aynı, iki tarafta da "query: " öneki.
        V = an.E5Kodlayici(an.model_hazirla(kaynak)).kodla([metinler[k] for k in kimlikler], an.SORGU_ONEKI)
        return V @ V.T, None
    except Exception as exc:  # bağımlılık veya model hatası: sonuç uydurma, gerekçeyi raporla
        return None, f"{type(exc).__name__}: {exc}"


def degerlendir(tablolar: dict, tekrar: int = 2000) -> dict:
    adlar = list(tablolar)
    M, ciftler, butunluk = birlestir(tablolar)
    sorgular = np.array([c[0] for c in ciftler])
    sonuc = {"butunluk": butunluk, "uyum": uyum(M, adlar, sorgular, tekrar)}
    ilk = next(iter(tablolar.values()))
    metinler = {**dict(zip(ilk["sorgu"], ilk["sorgu_metni"])), **dict(zip(ilk["aday"], ilk["aday_metni"]))}
    baslik = dict(zip(ilk["aday"], ilk["aday_basligi"]))
    S_tf, pos = tfidf_skorlari(metinler)
    S_e5, e5_gerekce = e5_skorlari(metinler)
    yontemler = {"TF-IDF": S_tf}
    if S_e5 is not None:
        from tmlab import anlamsal as an
        yontemler["E5"] = S_e5
        tf_inf, e5_inf = S_tf.copy(), S_e5.copy()
        np.fill_diagonal(tf_inf, -np.inf)  # uygulamadaki gibi: belgenin kendisi z-puanına katılmaz
        np.fill_diagonal(e5_inf, -np.inf)
        yontemler["Hibrit"] = an.hibrit(tf_inf, e5_inf)  # uygulamadaki sabit eşit ağırlık
    sorgu_satir, cift_satir = [], []
    for qi, q in enumerate(sorted(set(sorgular)), start=1):
        idx = np.flatnonzero(sorgular == q)
        adaylar = [ciftler[i][1] for i in idx]
        H = M[:, idx]
        kazanc = np.nanmean(H, axis=0)
        ilgili = np.nanmedian(H, axis=0) >= 1
        for ad, Sm in yontemler.items():
            skor = np.array([Sm[pos[q], pos[c]] for c in adaylar])
            sorgu_satir.append({"Sorgu": f"S{qi}", "Yöntem": ad, "İlgili aday (medyan≥1)": int(ilgili.sum()),
                                **{k: round(v, 4) for k, v in siralama_olcutleri(skor, kazanc, ilgili).items()}})
        for j, c in zip(idx, adaylar):
            cift_satir.append({"Sorgu": f"S{qi}", "Aday başlığı": baslik.get(c),
                               **{a: (None if np.isnan(M[r, j]) else int(M[r, j])) for r, a in enumerate(adlar)},
                               "Ortalama": round(float(np.nanmean(M[:, j])), 4),
                               "Farklı puan sayısı": int(len(set(M[:, j][~np.isnan(M[:, j])]))),
                               "Eksik var": bool(np.isnan(M[:, j]).any()),
                               **{f"{ad} skoru": round(float(Sm[pos[q], pos[c]]), 6) for ad, Sm in yontemler.items()}})
    sorgu_tablo = pd.DataFrame(sorgu_satir)
    karsilastirma = (sorgu_tablo.drop(columns=["Sorgu", "İlgili aday (medyan≥1)"]).groupby("Yöntem", sort=False)
                     .agg(["mean", "min", "max"]).round(4))
    karsilastirma.columns = [f"{a} {b}" for a, b in karsilastirma.columns]
    sonuc.update(sorgu_olcutleri=sorgu_tablo, karsilastirma=karsilastirma.reset_index(),
                 ciftler=pd.DataFrame(cift_satir), e5_gerekce=e5_gerekce, metin_sayisi=len(metinler))
    return sonuc


def excel_yaz(sonuc: dict, yol: str) -> None:
    u = sonuc["uyum"]
    notlar = pd.DataFrame([
        ("Uyum", "Krippendorff (2004) için alfa ≥ 0,667 geçici, ≥ 0,800 güvenilir sonuç eşiğidir; altında insan puanları "
                 "altın standart sayılmamalı, uyuşmazlıklar araştırmacı uzlaşmasıyla çözülmelidir."),
        ("Kapsam", "Model ölçütleri yalnız her sorgunun verilen aday listesi içindeki sıralamayı değerlendirir; tüm tez "
                   "evrenindeki erişim/recall başarısı değildir."),
        ("Belirsizlik", "Sorgu sayısı az; ölçütler sorgu başına hesaplanıp ortalandı, çiftler bağımsız sayılmadı. "
                        "Sorgu aralıkları raporlandı."),
        ("TF-IDF", f"IDF yalnız değerlendirme havuzundaki {sonuc['metin_sayisi']} özetten kestirildi; tam derlemde değerler değişebilir."),
        ("E5 / Hibrit", "Çalıştırıldı." if sonuc["e5_gerekce"] is None else f"{NOT_RUN}: {sonuc['e5_gerekce']}"),
    ], columns=["Konu", "Not"])
    with pd.ExcelWriter(yol, engine="openpyxl") as w:
        notlar.to_excel(w, sheet_name="Notlar", index=False)
        sonuc["butunluk"].to_excel(w, sheet_name="Veri_Butunlugu", index=False)
        u["ciftler"].to_excel(w, sheet_name="Uyum_Kappa", index=False)
        pd.concat([u["alfa"], pd.DataFrame([u["ozet"]])], axis=0).to_excel(w, sheet_name="Uyum_Alfa", index=False)
        satir = 0
        for m in u["matrisler"]:
            m.to_excel(w, sheet_name="Anlasmazlik_Matrisleri", startrow=satir)
            satir += len(m) + 3
        c = sonuc["ciftler"]
        c[(c["Farklı puan sayısı"] > 1) | c["Eksik var"]].to_excel(w, sheet_name="Uzlasma_Gerekli", index=False)
        sonuc["karsilastirma"].to_excel(w, sheet_name="Model_Karsilastirma", index=False)
        sonuc["sorgu_olcutleri"].to_excel(w, sheet_name="Model_Sorgu_Bazinda", index=False)
        c.to_excel(w, sheet_name="Cift_Skorlari", index=False)
        if sonuc["e5_gerekce"] is not None:
            for ad in ("Model_E5", "Model_Hibrit"):
                pd.DataFrame({"Durum": [NOT_RUN], "Gerekçe": [sonuc["e5_gerekce"]]}).to_excel(w, sheet_name=ad, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dosyalar", nargs="+", help="Puanlayıcı Excel dosyaları (en az 2)")
    ap.add_argument("--cikti", required=True, help="Rapor .xlsx yolu (depo dışında tutun)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    if len(a.dosyalar) < 2:
        ap.error("En az iki puanlayıcı dosyası gerekir.")
    tablolar = {f"P{i}": puanlayici_oku(y) for i, y in enumerate(a.dosyalar, start=1)}
    sonuc = degerlendir(tablolar, a.bootstrap)
    excel_yaz(sonuc, a.cikti)
    print(sonuc["butunluk"].to_string(index=False))
    print(sonuc["uyum"]["alfa"].to_string(index=False))
    print(sonuc["karsilastirma"].to_string(index=False))
    print("E5/Hibrit:", "çalıştırıldı" if sonuc["e5_gerekce"] is None else f"{NOT_RUN} — {sonuc['e5_gerekce']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
