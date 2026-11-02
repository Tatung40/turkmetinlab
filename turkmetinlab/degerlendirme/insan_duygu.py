"""İnsan duygu etiketleriyle puanlayıcı uyumu ve TürkMetinLab duygu yöntemlerinin doğrulanması.

Kullanım (turkmetinlab/ dizininde; çıktıyı depo DIŞINA yazın):
    python degerlendirme/insan_duygu.py D1.xlsx ... D6.xlsx --cikti ~/ozel/duygu_raporu.xlsx [--haric P5]

Girdi: puanlayıcı başına bir Excel; sütunlar 'Yorum ID', 'Ürün yorumu', 'Puanlayıcı'. Etiketler DEĞİŞTİRİLMEZ:
0 = NÖTR, 1 = OLUMLU, 2 = OLUMSUZ. Kayıtlar kimlik + normalize metinle eşleştirilir; eksik puan doldurulmaz.

Yöntem:
- Uyum: Krippendorff alfa (nominal), Fleiss kappa (tam kayıtlar; statsmodels varsa o, yoksa aynı formül),
  çiftler için Cohen kappa (scikit-learn). Etiketler nominaldir; sıralı kabul edilmez.
- Referans etiket (model tahmini kullanılmaz): oy birliği; mutlak çoğunluk (mevcut oyların yarısından fazlası);
  güçlü çoğunluk (en az 5/6 oranında). Mutlak çoğunluğu olmayan yorumlar uzlaşmaya gönderilir.
- Yöntemler: SWNetTR++ ve TES-Lex sözlükleri, önceden sabitlenmiş kural: olumlu eşleşme > olumsuz → 1,
  olumsuz > olumlu → 2, eşit ya da kutuplu eşleşme yok → 0. Kapsam = en az bir kutuplu eşleşmesi olan yorum oranı.
  ELECTRA yalnız model doğrulanıp yüklenebilirse çalışır; aksi hâlde NOT_RUN. Eşik/sözlük test verisine göre ayarlanmaz.
- Başarı: Accuracy, Macro-F1, sınıf bazında P/R/F1, karışıklık matrisi (scikit-learn), her referans stratejisi için.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for yol in (KOK, os.path.dirname(os.path.abspath(__file__))):
    if yol not in sys.path:
        sys.path.insert(0, yol)

from insan_ilgililik import alfa_cakisma  # noqa: E402  (Krippendorff 2011 örneğiyle sınanmış)

ETIKETLER = {0: "NÖTR", 1: "OLUMLU", 2: "OLUMSUZ"}
SUTUNLAR = ("Yorum ID", "Ürün yorumu", "Puanlayıcı")
NOT_RUN = "NOT_RUN"

# Hata analizi işaretleri: yalnız metinde görülebilen yüzey kanıtı; ironi/yazım hatası otomatik saptanmaz.
OLUMSUZLAMA = re.compile(r"(?<!\w)(değil|yok|hiç|asla)(?!\w)|\w(m[ıiuü]yor|ma[zd]|me[zd]|mad[ıi]|med[ıi])\w*", re.I)
KARSITLIK = re.compile(r"(?<!\w)(ama|fakat|ancak|lakin|yalnız|ne var ki|rağmen)(?!\w)", re.I)


def _norm(s) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s if s is not None else ""))).strip()


def _sutun(sutunlar, kosul, ad):
    bulunan = [c for c in sutunlar if kosul(c.casefold())]
    if len(bulunan) != 1:
        raise ValueError(f"'{ad}' sütunu belirsiz ya da yok: {list(sutunlar)}")
    return bulunan[0]


def puanlayici_oku(yol: str, esleme: dict | None = None) -> pd.DataFrame:
    """esleme: yalnız analizde uygulanan kod dönüşümü (ör. {3: 0}); dosya değiştirilmez.

    Geçerli kodlar dışındaki dolu hücreler (ör. metinsel not) durdurmaz: eksik sayılır, ham değeri
    'gecersiz' sütununda uzlaşma listesine taşınır.
    """
    t = pd.read_excel(yol, dtype=object)
    t.columns = [_norm(c) for c in t.columns]
    kimlik = _sutun(t.columns, lambda c: c.startswith("yorum id"), "Yorum ID")
    metin = _sutun(t.columns, lambda c: "yorum" in c and not c.startswith("yorum id"), "yorum metni")
    puan = _sutun(t.columns, lambda c: c.startswith("puanlayıcı"), "Puanlayıcı")
    t = t[t[[kimlik, metin]].notna().any(axis=1) & (t[kimlik].map(_norm) != "")]
    ham = t[puan]
    sayi = pd.to_numeric(ham, errors="coerce")
    gecerli = set(esleme or {}) | set(ETIKETLER)
    gecersiz = ham.notna() & (ham.map(_norm) != "") & (sayi.isna() | ~sayi.isin(gecerli))
    if esleme:
        sayi = sayi.map(lambda v: esleme.get(v, v) if pd.notna(v) else v)
    sayi = sayi.where(~gecersiz)
    return pd.DataFrame({"id": t[kimlik].map(_norm), "metin": t[metin].map(_norm), "etiket": sayi.to_numpy(dtype=float),
                         "gecersiz": ham.where(gecersiz).map(lambda v: None if pd.isna(v) else str(v).strip()).to_numpy()})


def birlestir(tablolar: dict) -> tuple:
    butunluk, kumeler = [], {}
    for ad, t in tablolar.items():
        anahtar = list(zip(t["id"], t["metin"]))
        kumeler[ad] = set(anahtar)
        butunluk.append({"Puanlayıcı": ad, "Satır": len(t), "Benzersiz kayıt": len(kumeler[ad]),
                         "Yinelenen kimlik": int(t["id"].duplicated().sum()), "Eksik etiket": int(t["etiket"].isna().sum()),
                         "Geçersiz hücre (eksik sayıldı)": int(t["gecersiz"].notna().sum()),
                         **{f"{k}={v}": int((t["etiket"] == k).sum()) for k, v in ETIKETLER.items()}})
    if any(b["Yinelenen kimlik"] for b in butunluk):
        raise ValueError("Aynı yorum kimliği bir dosyada birden çok kez geçiyor.")
    ilk = next(iter(kumeler.values()))
    if any(k != ilk for k in kumeler.values()):
        raise ValueError("Puanlayıcıların (kimlik, yorum) kümeleri farklı; eşleştirme yapılamaz.")
    kayitlar = sorted(ilk)
    M = np.array([[dict(zip(zip(t["id"], t["metin"]), t["etiket"]))[k] for k in kayitlar] for t in tablolar.values()])
    gecersiz = {}
    for ad, t in tablolar.items():
        for k, g in zip(zip(t["id"], t["metin"]), t["gecersiz"]):
            if g is not None:
                gecersiz.setdefault(k, []).append(f"{ad}: {g}")
    return M, kayitlar, pd.DataFrame(butunluk), gecersiz


def fleiss_kappa(C: np.ndarray) -> tuple:
    """C: puanlayıcı × birim, eksiksiz tamsayı etiketler."""
    sayim = np.array([[np.sum(C[:, j] == k) for k in ETIKETLER] for j in range(C.shape[1])])
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa as fk
        return float(fk(sayim, method="fleiss")), "statsmodels"
    except ImportError:
        m = C.shape[0]
        P = ((sayim ** 2).sum(1) - m) / (m * (m - 1))
        p = sayim.sum(0) / sayim.sum()
        return float((P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())), "formül (Fleiss 1971)"


def uyum(M: np.ndarray, adlar: list) -> dict:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    try:
        import krippendorff
        alfa, kaynak = float(krippendorff.alpha(M, level_of_measurement="nominal")), "krippendorff paketi"
    except ImportError:
        alfa, kaynak = alfa_cakisma(M, "nominal"), "çakışma matrisi (yerleşik)"
    tam = ~np.isnan(M).any(axis=0)
    fk, fk_kaynak = fleiss_kappa(M[:, tam].astype(int)) if tam.sum() else (float("nan"), "-")
    ciftler, matrisler = [], []
    for a, b in combinations(range(len(adlar)), 2):
        m = ~np.isnan(M[a]) & ~np.isnan(M[b])
        x, y = M[a][m].astype(int), M[b][m].astype(int)
        ciftler.append({"Çift": f"{adlar[a]}–{adlar[b]}", "n": int(m.sum()), "Tam uyum": round(float(np.mean(x == y)), 4),
                        "Cohen kappa": round(cohen_kappa_score(x, y, labels=list(ETIKETLER)), 4)})
        matrisler.append(pd.DataFrame(confusion_matrix(x, y, labels=list(ETIKETLER)),
                                      index=[f"{adlar[a]}={v}" for v in ETIKETLER.values()],
                                      columns=[f"{adlar[b]}={v}" for v in ETIKETLER.values()]))
    ozet = pd.DataFrame([
        {"Ölçüt": "Krippendorff alfa (nominal, mevcut gözlemler)", "Değer": round(alfa, 4), "Kaynak": kaynak,
         "Bağımsız": round(alfa_cakisma(M, "nominal"), 4)},
        {"Ölçüt": f"Fleiss kappa (tam kayıt n={int(tam.sum())})", "Değer": round(fk, 4), "Kaynak": fk_kaynak, "Bağımsız": None},
        {"Ölçüt": "Cohen kappa çift medyanı", "Değer": round(float(np.median([c["Cohen kappa"] for c in ciftler])), 4),
         "Kaynak": "scikit-learn", "Bağımsız": None}])
    return {"ozet": ozet, "ciftler": pd.DataFrame(ciftler), "matrisler": matrisler}


ASGARI_OY = 2  # referans için en az iki insan yargısı (iki kişilik veride yalnız uzlaşılan yorumlar)


def oy_tablosu(M: np.ndarray, adlar: list, kayitlar: list, gecersiz: dict | None = None) -> pd.DataFrame:
    satirlar = []
    for j, (kid, metin) in enumerate(kayitlar):
        v = M[:, j][~np.isnan(M[:, j])].astype(int)
        say = Counter(v)
        en = say.most_common() or [(None, 0)]
        n, tepe = len(v), en[0][1]
        yeterli = n >= ASGARI_OY
        esit = len(en) > 1 and en[1][1] == tepe
        satirlar.append({"Yorum ID": kid, "Yorum": metin, **{a: (None if np.isnan(M[i, j]) else int(M[i, j])) for i, a in enumerate(adlar)},
                         "Oy sayısı": n, **{f"{ETIKETLER[k]} oyu": say.get(k, 0) for k in ETIKETLER},
                         "En çok oy": tepe, "Eşitlik": esit,
                         "Oy birliği": int(en[0][0]) if yeterli and tepe == n else None,
                         "Mutlak çoğunluk": int(en[0][0]) if yeterli and tepe > n / 2 else None,
                         "Güçlü çoğunluk (oran ≥ 5/6)": int(en[0][0]) if yeterli and tepe / n >= 5 / 6 else None,
                         "Geçersiz hücre": "; ".join((gecersiz or {}).get((kid, metin), [])) or None})
    t = pd.DataFrame(satirlar)
    t["Durum"] = np.select([t["Oy birliği"].notna(), t["Güçlü çoğunluk (oran ≥ 5/6)"].notna(), t["Mutlak çoğunluk"].notna(),
                            t["Oy sayısı"] < ASGARI_OY, t["Eşitlik"]],
                           ["oy birliği", "güçlü çoğunluk", "mutlak çoğunluk", "yetersiz oy — uzlaşma gerekli",
                            "eşitlik — uzlaşma gerekli"], "çoğunluk yok — uzlaşma gerekli")
    return t


def _kutup_say(eslesmeler, olumlu, olumsuz) -> tuple:
    p = sum(1 for e in eslesmeler if olumlu(e))
    n = sum(1 for e in eslesmeler if olumsuz(e))
    return p, n


def karar(p: int, n: int) -> int:
    """Önceden sabitlenmiş kural; veriyle ayarlanmaz."""
    return 1 if p > n else 2 if n > p else 0


def yontem_tahminleri(metinler: list, sw_sozluk=None, tes_cerceve=None, electra_durum=None) -> dict:
    from tmlab import duygu, teslex

    sonuc, notlar = {}, {}
    if sw_sozluk is None:
        try:
            sw_sozluk = duygu.load_lexicon()
        except Exception as exc:
            notlar["SWNetTR++"] = f"{NOT_RUN}: sözlük yüklenemedi ({type(exc).__name__})"
    if sw_sozluk is not None:
        tah, kap, ayr = [], [], []
        for m in metinler:
            r = duygu.analyze(m, sw_sozluk)
            p, n = r["Olumlu eşleşme"], r["Olumsuz eşleşme"]
            tah.append(karar(p, n)); kap.append(p + n > 0); ayr.append((p, n))
        sonuc["SWNetTR++"] = {"tahmin": np.array(tah), "kapsam": np.array(kap), "sayim": ayr}
    if tes_cerceve is None:
        try:
            tes_cerceve = teslex.load_source()
        except Exception as exc:
            notlar["TES-Lex"] = f"{NOT_RUN}: sözlük yüklenemedi ({type(exc).__name__})"
    if tes_cerceve is not None:
        kelime = next(c for c in tes_cerceve.columns if c.lower().startswith("kelime"))
        kutup = next(c for c in tes_cerceve.columns if c.startswith("Pozitif/Negatif/Nötr\n"))
        sozluk = teslex.compile_lexicon(tes_cerceve, kelime, kutup)
        tah, kap, ayr = [], [], []
        for m in metinler:
            es = teslex.analyze(m, sozluk)["eşleşmeler"]
            p, n = _kutup_say(es, lambda e: teslex.normalize(e["Polarite"]) == "pozitif",
                              lambda e: teslex.normalize(e["Polarite"]) == "negatif")
            tah.append(karar(p, n)); kap.append(p + n > 0); ayr.append((p, n))
        sonuc["TES-Lex"] = {"tahmin": np.array(tah), "kapsam": np.array(kap), "sayim": ayr}
    if electra_durum is None:
        try:
            from tmlab import electra_duygu as el
            durum = el.load_model()
        except Exception as exc:
            notlar["ELECTRA"] = (f"{NOT_RUN}: model yüklenemedi ({type(exc).__name__}); sürüm commit'e sabitli değil "
                                 "ve lisans doğrulanmadı")
        else:
            esle = {"negative": 2, "neutral": 0, "notr": 0, "nötr": 0, "positive": 1}  # config.json: "Notr"
            etiketler = {k: esle.get(str(v).lower()) for k, v in durum[2].items()}
            if None in etiketler.values():
                notlar["ELECTRA"] = f"{NOT_RUN}: beklenmeyen model etiketleri {sorted(durum[2].values())}"
            else:
                tah = [esle[str(el.infer(m, durum)["Etiket"]).lower()] for m in metinler]
                sonuc["ELECTRA"] = {"tahmin": np.array(tah), "kapsam": np.ones(len(metinler), bool), "sayim": None}
    return {"sonuc": sonuc, "notlar": notlar}


def basari(y: np.ndarray, p: np.ndarray) -> tuple:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    pr, rc, f, s = precision_recall_fscore_support(y, p, labels=list(ETIKETLER), zero_division=0)
    sinif = {f"{ETIKETLER[k]} {a}": round(float(v[i]), 4) for i, k in enumerate(ETIKETLER)
             for a, v in (("P", pr), ("R", rc), ("F1", f))}
    return ({"n": int(len(y)), "Accuracy": round(float(accuracy_score(y, p)), 4),
             "Macro-F1": round(float(f1_score(y, p, labels=list(ETIKETLER), average="macro", zero_division=0)), 4), **sinif},
            confusion_matrix(y, p, labels=list(ETIKETLER)))


STRATEJILER = ("Oy birliği", "Güçlü çoğunluk (oran ≥ 5/6)", "Mutlak çoğunluk")


def degerlendir(tablolar: dict, haric: tuple = (), yontemler: dict | None = None) -> dict:
    adlar = list(tablolar)
    M, kayitlar, butunluk, gecersiz = birlestir(tablolar)
    oy = oy_tablosu(M, adlar, kayitlar, gecersiz)
    yontemler = yontemler or yontem_tahminleri([m for _, m in kayitlar])
    sonuc = {"butunluk": butunluk, "uyum": uyum(M, adlar), "oy": oy, "notlar": yontemler["notlar"]}
    duyarlilik = []
    if haric:
        kalan = [i for i, a in enumerate(adlar) if a not in haric]
        sonuc["uyum_haric"] = uyum(M[kalan], [adlar[i] for i in kalan])
        oy_h = oy_tablosu(M[kalan], [adlar[i] for i in kalan], kayitlar, gecersiz)
    satir, matrisler = [], []
    for ad, r in yontemler["sonuc"].items():
        for etiket_tablosu, kume in [(oy, "tüm puanlayıcılar")] + ([(oy_h, "hariç: " + ",".join(haric))] if haric else []):
            for strateji in STRATEJILER:
                ref = etiket_tablosu[strateji].to_numpy(dtype=float)
                m = ~np.isnan(ref)
                if not m.any():
                    continue
                olcu, cm = basari(ref[m].astype(int), r["tahmin"][m])
                kayit = {"Yöntem": ad, "Puanlayıcı kümesi": kume, "Referans": strateji,
                         "Referans kapsamı (%)": round(100 * m.mean(), 2),
                         "Yöntem kapsamı (%)": round(100 * float(r["kapsam"][m].mean()), 2), **olcu}
                (satir if kume == "tüm puanlayıcılar" else duyarlilik).append(kayit)
                if kume == "tüm puanlayıcılar":
                    matrisler.append((f"{ad} · {strateji}", pd.DataFrame(
                        cm, index=[f"insan={v}" for v in ETIKETLER.values()], columns=[f"model={v}" for v in ETIKETLER.values()])))
    sonuc.update(basari=pd.DataFrame(satir), duyarlilik=pd.DataFrame(duyarlilik), matrisler=matrisler,
                 hata=hata_analizi(oy, yontemler["sonuc"]))
    return sonuc


def hata_analizi(oy: pd.DataFrame, sonuc: dict) -> tuple:
    """Mutlak çoğunluk referansıyla; işaretler yalnız metindeki yüzey kanıtına dayanır."""
    ref = oy["Mutlak çoğunluk"].to_numpy(dtype=float)
    m = ~np.isnan(ref)
    metin = oy["Yorum"].tolist()
    isaret = {"Olumsuzlama ifadesi": np.array([bool(OLUMSUZLAMA.search(t)) for t in metin]),
              "Karşıtlık bağlacı": np.array([bool(KARSITLIK.search(t)) for t in metin]),
              "Kısa yorum (≤3 sözcük)": np.array([len(t.split()) <= 3 for t in metin])}
    ozet, ornek = [], []
    for ad, r in sonuc.items():
        hatali = m & (r["tahmin"] != np.nan_to_num(ref, nan=-1))
        bayrak = dict(isaret)
        bayrak["Sözlük eşleşmesi yok"] = ~r["kapsam"]
        if r.get("sayim"):
            bayrak["Hem olumlu hem olumsuz eşleşme"] = np.array([p > 0 and n > 0 for p, n in r["sayim"]])
        for tur, b in bayrak.items():
            k = m & b
            if k.sum():
                ozet.append({"Yöntem": ad, "İşaret": tur, "Yorum (referanslı)": int(k.sum()),
                             "Hata oranı (%)": round(100 * float(hatali[k].mean()), 2),
                             "İşaretsiz yorumlarda hata (%)": round(100 * float(hatali[m & ~b].mean()), 2) if (m & ~b).any() else None})
        for i in np.flatnonzero(hatali):
            ornek.append({"Yöntem": ad, "Yorum ID": oy["Yorum ID"].iat[i], "Yorum": metin[i],
                          "İnsan (mutlak çoğunluk)": ETIKETLER[int(ref[i])], "Model": ETIKETLER[int(r["tahmin"][i])],
                          "Oy dağılımı": "/".join(str(oy[f"{v} oyu"].iat[i]) for v in ETIKETLER.values()),
                          "İşaretler": ", ".join(t for t, b in bayrak.items() if b[i])})
    return pd.DataFrame(ozet), pd.DataFrame(ornek)


def excel_yaz(sonuc: dict, yol: str, haric: tuple = ()) -> None:
    oy = sonuc["oy"]
    notlar = [("Etiketler", "0 = NÖTR, 1 = OLUMLU, 2 = OLUMSUZ; hiçbir etiket değiştirilmedi, eksikler doldurulmadı."),
              ("Uyum", "Krippendorff alfa ≥ 0,667 geçici, ≥ 0,800 güvenilir sonuç eşiğidir (Krippendorff 2004). Altında insan "
                       "etiketleri tek başına altın standart sayılmamalıdır."),
              ("Referans", "Model tahmini referans üretmekte kullanılmadı. Mutlak çoğunluğu olmayan yorumlar uzlaşmaya bırakıldı."),
              ("Karar kuralı", "Sözlük yöntemleri: olumlu eşleşme > olumsuz → OLUMLU, tersi → OLUMSUZ, eşit/eşleşme yok → NÖTR. "
                               "Kural test verisi görülmeden sabitlendi; sözlük ve eşik ayarlanmadı."),
              ("Genelleme", "Tek bir ürünün yorumları; genel Türkçe duygu analizi başarısı iddia edilemez."),
              ("Hata analizi", "İşaretler yüzey kalıplarıdır (olumsuzlama, karşıtlık, kısa yorum, sözlük kapsamı). İroni ve "
                               "yazım hatası otomatik saptanmadı; bu türler için iddia yoktur."),
              ("Olumsuzlama", "TürkMetinLab'de olumsuzlamayı ele alan bir duygu sınıflandırıcısı yoktur; sözlük yöntemleri "
                              "olumsuzlamayı dikkate almaz.")]
    notlar += [(ad, metin) for ad, metin in sonuc["notlar"].items()]
    if haric:
        notlar.append(("Duyarlılık", f"{', '.join(haric)} hariç uyum ve başarı ayrı sayfalarda; etiketleri değiştirilmedi."))
    with pd.ExcelWriter(yol, engine="openpyxl") as w:
        sonuc["butunluk"].to_excel(w, sheet_name="Veri_Butunlugu", index=False)
        u = sonuc["uyum"]
        u["ozet"].to_excel(w, sheet_name="Puanlayici_Uyumu", index=False)
        u["ciftler"].to_excel(w, sheet_name="Puanlayici_Uyumu", index=False, startrow=len(u["ozet"]) + 2)
        if "uyum_haric" in sonuc:
            uh = sonuc["uyum_haric"]
            uh["ozet"].to_excel(w, sheet_name="Uyum_Duyarlilik", index=False)
            uh["ciftler"].to_excel(w, sheet_name="Uyum_Duyarlilik", index=False, startrow=len(uh["ozet"]) + 2)
        dag = sonuc["butunluk"].set_index("Puanlayıcı")[[f"{k}={v}" for k, v in ETIKETLER.items()] + ["Eksik etiket"]]
        dag.to_excel(w, sheet_name="Sinif_Dagilimlari")
        oy["Durum"].value_counts().rename_axis("Oy durumu").to_frame("Yorum").to_excel(
            w, sheet_name="Sinif_Dagilimlari", startrow=len(dag) + 3)
        oy.to_excel(w, sheet_name="Oylar", index=False)
        oy[oy["Mutlak çoğunluk"].isna() | oy["Geçersiz hücre"].notna()].to_excel(w, sheet_name="Uzlasma_Gerekli", index=False)
        sonuc["basari"].to_excel(w, sheet_name="Model_Basari", index=False)
        if not sonuc["duyarlilik"].empty:
            sonuc["duyarlilik"].to_excel(w, sheet_name="Model_Duyarlilik", index=False)
        satir = 0
        for baslik, cm in sonuc["matrisler"]:
            pd.DataFrame({baslik: []}).to_excel(w, sheet_name="Karmaşıklık_Matrisleri", startrow=satir, index=False)
            cm.to_excel(w, sheet_name="Karmaşıklık_Matrisleri", startrow=satir + 1)
            satir += len(cm) + 4
        for i, cm in enumerate(u["matrisler"]):
            cm.to_excel(w, sheet_name="Puanlayici_Matrisleri", startrow=5 * i)
        ozet, ornek = sonuc["hata"]
        ozet.to_excel(w, sheet_name="Hata_Analizi", index=False)
        ornek.to_excel(w, sheet_name="Hata_Analizi", index=False, startrow=len(ozet) + 3)
        pd.DataFrame(notlar, columns=["Konu", "Not"]).to_excel(w, sheet_name="Yontem_Sinirliliklari", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dosyalar", nargs="+")
    ap.add_argument("--cikti", required=True, help="Rapor .xlsx yolu (depo dışında tutun)")
    ap.add_argument("--haric", nargs="*", default=[], help="Duyarlılık analizinde dışarıda bırakılacak puanlayıcılar (ör. P5)")
    ap.add_argument("--esleme", nargs="*", default=[], help="Yalnız analizde kod dönüşümü, ör. 3=0 (dosya değişmez)")
    a = ap.parse_args(argv)
    esleme = {float(k): float(v) for k, v in (x.split("=") for x in a.esleme)}
    if len(a.dosyalar) < 2:
        ap.error("En az iki puanlayıcı dosyası gerekir.")
    tablolar = {f"P{i}": puanlayici_oku(y, esleme) for i, y in enumerate(a.dosyalar, start=1)}
    sonuc = degerlendir(tablolar, tuple(a.haric))
    excel_yaz(sonuc, a.cikti, tuple(a.haric))
    print(sonuc["uyum"]["ozet"].to_string(index=False))
    print(sonuc["oy"]["Durum"].value_counts().to_string())
    print(sonuc["basari"].to_string(index=False))
    for ad, metin in sonuc["notlar"].items():
        print(ad, metin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
