"""TürkMetinLab 2.0 — gerçek akademik verilerle yöntemsel validasyon.

Veri
----
1. DergiParkEn / TRAll.txt (Kılıç Ünlü & Çetin, 2023, Neural Computing and Applications 35:3413–3422):
   DergiPark'ta yayımlanmış 1.417 Türkçe makalenin başlığı, özeti ve YAZARLARIN VERDİĞİ anahtar
   sözcükleri. Lisansı belirtilmediği için depoya EKLENMEZ; betik çalışırken GitHub'dan belleğe indirir.
   https://github.com/humakilicunlu/DergiParkEn
2. UD Turkish-BOUN örneği (tests/veri, CC BY-SA 4.0): türü (biyografi, deneme, bilgilendirici,
   haber, popüler) bilinen gerçek Türkçe metinler.

Kullanım:  python degerlendirme/v2_dogrulama.py [benzerlik ag konu grup]
Sonuçlar ekrana ve degerlendirme/v2_dogrulama_sonuclari.json dosyasına yazılır.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.join(KOK, "tests"))

DERGIPARK_URL = "https://raw.githubusercontent.com/humakilicunlu/DergiParkEn/main/TRAll.txt"
SONUC_YOLU = os.path.join(KOK, "degerlendirme", "v2_dogrulama_sonuclari.json")


def dergipark():
    """[{kimlik, baslik, ozet, anahtar: set}] — yalnızca bellekte."""
    from tmlab.preprocess import turkce_kucuk_harf

    yerel = os.environ.get("TML_DERGIPARK_YOLU")
    if yerel:
        metin = open(yerel, encoding="utf-8").read()
    else:
        with urllib.request.urlopen(DERGIPARK_URL, timeout=60) as y:
            metin = y.read().decode("utf-8")
    kayitlar = []
    for satir in metin.splitlines():
        a = satir.split("\t")
        if len(a) != 4 or not a[2].strip():
            continue
        anahtar = {turkce_kucuk_harf(k).strip(" .\"'") for k in a[3].replace(";", ",").split(",")}
        kayitlar.append({"kimlik": a[0], "baslik": a[1].strip('"'), "ozet": a[2].strip('"'),
                         "anahtar": {k for k in anahtar if k}})
    return kayitlar


def _auc(pozitif: np.ndarray, negatif: np.ndarray) -> float:
    """Mann–Whitney U ile ROC AUC (eşitlikler yarım sayılır)."""
    from scipy.stats import mannwhitneyu

    u = mannwhitneyu(pozitif, negatif, alternative="two-sided").statistic
    return float(u / (len(pozitif) * len(negatif)))


# ---------------------------------------------------------------------------
# Modül 1 — Belge benzerliği
# ---------------------------------------------------------------------------

def benzerlik_dogrulamasi(kayitlar) -> dict:
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer

    from tmlab.analysis import tfidf_hesapla
    from tmlab.benzerlik import en_yakin_komsular, kosinus_matrisi
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

    sonuc = {}
    t = time.perf_counter()
    belgeler = derlem_on_isle([k["ozet"] for k in kayitlar], OnIslemeAyarlari())
    s = tfidf_hesapla(belgeler, [k["kimlik"] for k in kayitlar])
    n = s.matris.shape[0]
    secim = list(range(min(n, 1417)))
    S = kosinus_matrisi(s.matris[secim]) if len(secim) <= 2000 else None
    sonuc["belge"] = n
    sonuc["sure_sn_matris"] = round(time.perf_counter() - t, 2)

    # (1) Yazar anahtar sözcüğü paylaşan çiftler daha mı benzer?
    anahtar = [kayitlar[i]["anahtar"] for i in secim]
    i, j = np.triu_indices(len(secim), k=1)
    ortak = np.array([len(anahtar[a] & anahtar[b]) for a, b in zip(i, j)])
    jaccard = np.array([len(anahtar[a] & anahtar[b]) / max(1, len(anahtar[a] | anahtar[b])) for a, b in zip(i, j)])
    benz = S[i, j]
    sonuc["cift"] = int(len(benz))
    sonuc["ortak_anahtarli_cift"] = int((ortak > 0).sum())
    sonuc["ort_benzerlik_ortak_anahtar_var"] = round(float(benz[ortak > 0].mean()), 4)
    sonuc["ort_benzerlik_ortak_anahtar_yok"] = round(float(benz[ortak == 0].mean()), 4)
    sonuc["auc_ortak_anahtar"] = round(_auc(benz[ortak > 0], benz[ortak == 0]), 4)
    rho, p = spearmanr(benz, jaccard)
    sonuc["spearman_kosinus_anahtar_jaccard"] = round(float(rho), 4)
    sonuc["spearman_p"] = float(p)

    # (2) Başlıkla arama: her makalenin başlığı, aynı TF-IDF uzayında hangi özete en yakın?
    #     (Özetler üzerinde öğrenilmiş IDF; başlıklar aynı ön işlemeyle sorgu olarak gömülür.)
    terimler = {t_: k_ for k_, t_ in enumerate(s.terimler)}
    vek = TfidfVectorizer(analyzer=lambda x: x, lowercase=False, vocabulary=terimler)
    vek.fit([b.sozcukler for b in belgeler if b.sozcukler])
    basliklar = derlem_on_isle([kayitlar[i]["baslik"] for i in secim], OnIslemeAyarlari())
    Q = vek.transform([b.sozcukler for b in basliklar])
    Y = (Q @ s.matris[secim].T).toarray()
    dogru_sira = []
    for r in range(len(secim)):
        if Q[r].nnz == 0:
            continue
        dogru_sira.append(int((Y[r] > Y[r, r]).sum()) + 1)
    dogru_sira = np.array(dogru_sira)
    sonuc["baslik_sorgu_sayisi"] = int(len(dogru_sira))
    sonuc["baslik_ilk1_isabet"] = round(float((dogru_sira == 1).mean()), 4)
    sonuc["baslik_ilk5_isabet"] = round(float((dogru_sira <= 5).mean()), 4)
    sonuc["baslik_ortalama_ters_sira"] = round(float((1 / dogru_sira).mean()), 4)
    sonuc["rastgele_ilk1_beklenen"] = round(1 / len(secim), 5)

    # (3) Parçalı en yakın komşu = tam matris (gerçek veride)
    I, V = en_yakin_komsular(s.matris[secim], k=10, parca=97)
    tam = np.argsort(-np.where(np.eye(len(secim), dtype=bool), -1, S), axis=1, kind="stable")[:, :10]
    sonuc["knn_tam_matrisle_ayni_deger"] = bool(np.allclose(V, np.take_along_axis(S, tam, 1)))
    return sonuc


def ud_kumeleme_dogrulamasi() -> dict:
    """UD BOUN: tür etiketleri ile kümeler örtüşüyor mu? (rastgele etiket karşılaştırmasıyla)"""
    from collections import defaultdict

    from sklearn.metrics import adjusted_rand_score

    from conftest import ud_cumleleri
    from tmlab.analysis import tfidf_hesapla
    from tmlab.benzerlik import hiyerarsik_kumeleme, kosinus_matrisi, kume_etiketleri
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

    tur = defaultdict(list)
    for c in ud_cumleleri():
        tur[c["tur"]].append(c["metin"])
    metin, etiket = [], []
    for t_, cs in sorted(tur.items()):
        for k in range(0, len(cs) - 4, 5):
            metin.append(" ".join(cs[k:k + 5]))
            etiket.append(t_)
    s = tfidf_hesapla(derlem_on_isle(metin, OnIslemeAyarlari()))
    S = kosinus_matrisi(s.matris)
    sonuc = {"belge": len(metin), "tur_sayisi": len(tur)}
    rng = np.random.default_rng(0)
    for y in ("average", "ward"):
        ari = adjusted_rand_score(etiket, kume_etiketleri(hiyerarsik_kumeleme(S, y), len(tur)))
        sonuc[f"ari_{y}"] = round(float(ari), 4)
    sonuc["ari_rastgele_ortalama"] = round(float(np.mean([adjusted_rand_score(etiket, rng.permutation(etiket))
                                                          for _ in range(200)])), 4)
    # Tür içi / türler arası ortalama benzerlik
    e = np.array(etiket)
    ayni = e[:, None] == e[None, :]
    ust = np.triu(np.ones_like(S, dtype=bool), 1)
    sonuc["ort_benzerlik_ayni_tur"] = round(float(S[ust & ayni].mean()), 4)
    sonuc["ort_benzerlik_farkli_tur"] = round(float(S[ust & ~ayni].mean()), 4)
    return sonuc


# ---------------------------------------------------------------------------
# Modül 2 — Sözcük birliktelik ağları
# ---------------------------------------------------------------------------

def ag_dogrulamasi(kayitlar) -> dict:
    """Dış ölçüt: yazarların verdiği ÇOK SÖZCÜKLÜ anahtar terimler (ör. "yapay sinir ağları").

    Aynı anahtar terim içinde geçen sözcük çiftleri 'olumlu' kabul edilir. Her ölçütün bu çiftleri
    diğer çiftlerden ne kadar iyi ayırdığı ROC AUC ile ölçülür. Anahtar sözcükler özetten ayrı bir
    alandır; analize girmez.
    """
    from itertools import combinations

    from tmlab.birliktelik import AgAyarlari, birliktelik_hesapla, kenar_tablosu
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

    ayar = OnIslemeAyarlari()
    belgeler = derlem_on_isle([k["ozet"] for k in kayitlar], ayar)
    olumlu = set()
    for k in kayitlar:
        for terim in k["anahtar"]:
            sozc = sorted(set(derlem_on_isle([terim], ayar)[0].sozcukler))
            olumlu.update(combinations(sozc, 2))
    sonuc = {"olumlu_cift_havuzu": len(olumlu)}
    for ad, ag in [("pencere5_cumle", AgAyarlari("pencere", 5, True, 2000, 3)),
                   ("pencere2_cumle", AgAyarlari("pencere", 2, True, 2000, 3)),
                   ("cumle", AgAyarlari("bolut", 5, True, 2000, 3)),
                   ("belge", AgAyarlari("belge", 5, True, 2000, 3))]:
        t = time.perf_counter()
        r = birliktelik_hesapla(belgeler, ag)
        k = kenar_tablosu(r)
        etiket = np.array([tuple(sorted((a, b))) in olumlu for a, b in zip(k["Kaynak"], k["Hedef"])])
        d = {"kenar": int(len(k)), "olumlu_kenar": int(etiket.sum()), "sure_sn": round(time.perf_counter() - t, 2)}
        for sutun in ("PPMI", "Jaccard", "Birlikte görülme"):
            d[f"auc_{sutun}"] = round(_auc(k[sutun].to_numpy()[etiket], k[sutun].to_numpy()[~etiket]), 4)
        ust = k.sort_values("PPMI", ascending=False).head(100)
        d["ilk100_ppmi_olumlu_orani"] = round(float(np.mean([tuple(sorted((a, b))) in olumlu
                                                              for a, b in zip(ust["Kaynak"], ust["Hedef"])])), 3)
        ust = k.sort_values("Birlikte görülme", ascending=False).head(100)
        d["ilk100_ham_olumlu_orani"] = round(float(np.mean([tuple(sorted((a, b))) in olumlu
                                                            for a, b in zip(ust["Kaynak"], ust["Hedef"])])), 3)
        d["olumlu_oran_tum_kenarlar"] = round(float(etiket.mean()), 4)
        sonuc[ad] = d
    r = birliktelik_hesapla(belgeler, AgAyarlari("pencere", 5, True, 2000, 5))
    sonuc["ornek_ilk15_ppmi"] = [f"{a} — {b} ({c})" for a, b, c in
                                 kenar_tablosu(r).sort_values("PPMI", ascending=False).head(15)[
                                     ["Kaynak", "Hedef", "Birlikte görülme"]].itertuples(index=False)]
    return sonuc


# ---------------------------------------------------------------------------
# Modül 3 — Konu modellemesi
# ---------------------------------------------------------------------------

def konu_dogrulamasi(kayitlar) -> dict:
    """Dış ölçüt: baskın konunun ilk 10 sözcüğü, belgenin YAZAR anahtar sözcükleriyle örtüşüyor mu?

    Taban çizgisi: aynı konular belgelere rastgele (permütasyonla) atanırsa beklenen örtüşme.
    Ayrıca NPMI/UMass tutarlılığı, çeşitlilik ve alt örneklem kararlılığı raporlanır.
    """
    from tmlab.konu import KonuAyarlari, konu_modeli
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

    ayar = OnIslemeAyarlari(durak_sozcuk_ingilizce=True)
    belgeler = derlem_on_isle([k["ozet"] for k in kayitlar], ayar)
    anahtar = [set(t for terim in k["anahtar"] for t in derlem_on_isle([terim], ayar)[0].sozcukler)
               for k in kayitlar]
    rng = np.random.default_rng(0)
    sonuc = {}
    for yontem in ("lda", "nmf"):
        for k in (5, 10, 15, 20):
            t = time.perf_counter()
            r = konu_modeli(belgeler, KonuAyarlari(yontem, k), kararlilik_tekrari=3)
            sure = round(time.perf_counter() - t, 1)
            ust = [set(u) for u in r.ust_sozcukler(10)]
            baskin = np.argmax(r.belge_konu, axis=1)
            idx = r.belge_indisleri
            ortusme = np.mean([len(ust[baskin[m]] & anahtar[i]) > 0 for m, i in enumerate(idx)])
            taban = np.mean([np.mean([len(ust[b] & anahtar[i]) > 0 for b, i in zip(rng.permutation(baskin), idx)])
                             for _ in range(50)])
            sonuc[f"{yontem}_k{k}"] = {
                **{a: v for a, v in r.olcutler.items() if a not in ("Sözlük büyüklüğü",)},
                "anahtar_ortusme": round(float(ortusme), 4), "anahtar_ortusme_rastgele": round(float(taban), 4),
                "sure_sn (kararlılık dahil)": sure,
                "ilk3_konu": [" ".join(u) for u in r.ust_sozcukler(8)[:3]],
            }
            print(yontem, k, sonuc[f"{yontem}_k{k}"], flush=True)
    return sonuc


# ---------------------------------------------------------------------------
# Modül 4 — Grup karşılaştırması
# ---------------------------------------------------------------------------

def grup_dogrulamasi(kayitlar) -> dict:
    """(1) Kalibrasyon: anlamsız (rastgele) gruplamalarda yanlış pozitif oranı.
    (2) Pozitif kontrol: yazar anahtar sözcüklerine göre ayrılmış gerçek konu grupları."""
    from tmlab.grup import GrupAyarlari, ayirt_edici_sozcukler, grup_verisi_hazirla
    from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle

    belgeler = derlem_on_isle([k["ozet"] for k in kayitlar], OnIslemeAyarlari(durak_sozcuk_ingilizce=True))
    sonuc = {}
    rng = np.random.default_rng(123)
    for onsel in ("bilgilendirici", "zayif"):
        nominal, asan_bolme, asan_sozcuk = [], [], []
        for t in range(30):
            etiket = np.where(rng.permutation(len(belgeler)) < len(belgeler) // 2, "A", "B")
            veri, _ = grup_verisi_hazirla(belgeler, list(etiket))
            tablo, perm = ayirt_edici_sozcukler(veri, GrupAyarlari(onsel=onsel, min_siklik=5, permutasyon=100,
                                                                   tohum=1000 + t))
            a = tablo[tablo["Grup"] == "A"]
            nominal.append(float((a["z (ağırlıklı log-odds)"].abs() > 1.96).mean()))
            asan = int((tablo["Permütasyon eşiğini aşıyor"] == "evet").sum())
            asan_sozcuk.append(asan)
            asan_bolme.append(asan > 0)
        sonuc[f"rastgele_bolme_{onsel}"] = {
            "bolme_sayisi": 30,
            "nominal_|z|>1.96_orani_ortalama": round(float(np.mean(nominal)), 4),
            "kuramsal_beklenen": 0.05,
            "permutasyon_esigini_asan_sozcuk_ortalama": round(float(np.mean(asan_sozcuk)), 2),
            "en_az_bir_sozcuk_asan_bolme_orani": round(float(np.mean(asan_bolme)), 3),
            "not": "Her bölmede 2 grup sınanır; aile bazında beklenen oran ~%5 × 2 grup ≈ %10",
        }
    # Pozitif kontrol: deprem/sismik vs enerji/güneş/rüzgâr (yazar anahtar sözcükleri)
    grup = []
    for k in kayitlar:
        ana = " ".join(k["anahtar"])
        d = any(x in ana for x in ("deprem", "sismik"))
        e = any(x in ana for x in ("enerji", "güneş", "rüzgar", "rüzgâr", "fotovoltaik"))
        grup.append("deprem" if d and not e else ("enerji" if e and not d else None))
    veri, bos = grup_verisi_hazirla(belgeler, grup)
    tablo, perm = ayirt_edici_sozcukler(veri, GrupAyarlari(min_siklik=3, permutasyon=200))
    sonuc["pozitif_kontrol"] = {
        "belge": {g: int((veri.etiket == i).sum()) for i, g in enumerate(veri.gruplar)},
        "permutasyon_esikleri": {g: round(float(e), 3) for g, e in zip(veri.gruplar, perm["esik"])},
        **{f"{g}_ilk12": [f"{w} ({z:.1f}{'*' if a == 'evet' else ''})" for w, z, a in
                          tablo[tablo["Grup"] == g].head(12)[["Sözcük", "z (ağırlıklı log-odds)",
                                                               "Permütasyon eşiğini aşıyor"]].itertuples(index=False)]
           for g in veri.gruplar},
        **{f"{g}_esigi_asan_sozcuk": int(((tablo["Grup"] == g) & (tablo["Permütasyon eşiğini aşıyor"] == "evet")).sum())
           for g in veri.gruplar},
    }
    return sonuc


MODULLER = {
    "benzerlik": lambda k: {"dergipark": benzerlik_dogrulamasi(k), "ud_boun_tur": ud_kumeleme_dogrulamasi()},
    "ag": lambda k: {"dergipark": ag_dogrulamasi(k)},
    "konu": lambda k: {"dergipark": konu_dogrulamasi(k)},
    "grup": lambda k: {"dergipark": grup_dogrulamasi(k)},
}


if __name__ == "__main__":
    secilen = sys.argv[1:] or list(MODULLER)
    kayitlar = dergipark()
    print(f"DergiPark: {len(kayitlar)} özet", flush=True)
    sonuclar = json.load(open(SONUC_YOLU, encoding="utf-8")) if os.path.exists(SONUC_YOLU) else {}
    for ad in secilen:
        t = time.perf_counter()
        sonuclar[ad] = MODULLER[ad](kayitlar)
        sonuclar[ad]["toplam_sure_sn"] = round(time.perf_counter() - t, 1)
        print(ad, json.dumps(sonuclar[ad], ensure_ascii=False, indent=1), flush=True)
    with open(SONUC_YOLU, "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=1)
