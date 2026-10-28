"""50 sorguluk uzman değerlendirme havuzunu ve iki kör uzman dosyasını üretir (GitHub Actions'ta, gerçek E5 ile).

Kaynak: DergiPark Türkçe özetleri (humakilicunlu/DergiParkEn, TRAll.txt; çoğu mühendislik; lisans belirtilmemiş).
Görev: belge → belge benzerliği (uygulamanın "Belge benzerliği" işlevi). Sorgu = bir makale (başlık + özet), aday =
derlemdeki başka bir makale.

Seçim ve dışlama kuralları (tohum 2510):
1. Uygun belge: fizibilitedeki kurallar (özet ≥ 30 sözcük, anahtar sözcük var, yinelenen özet bir kez).
2. Sorgu adayı olamaz:
   - fizibilitenin n=500 örnekleminde (tohum 2501) yer almış belge (önceki analizlerle örtüşme olmasın);
   - belirgin İngilizce içeren özet;
   - E5 sınırını (512 belirteç) aşan özet.
3. Konu yayılımı: Derlemde alan etiketi yoktur. Yazar anahtar sözcüklerinden (iki yöntemin de kullanmadığı bilgi)
   10 konu kümesi çıkarılır (k-ortalama). Her kümeden 5 sorgu çekilir; küme yetmezse kalan sorgular rastgele
   tamamlanır.
4. Adaylar: tüm uygun derlem. Sorgunun kendisi ve yakın kopyaları dışlanır: aynı normalleştirilmiş başlık ya da
   özetin ilk 200 karakteri aynı.
5. Yöntemler uygulamadakiyle aynıdır:
   - TF-IDF: varsayılan ön işleme, tek sözcük, yumuşatılmış IDF.
   - E5: ONNX FP32, iki tarafta "query: ".
   - Hibrit: satır başına z-puanı toplamı.
   Her yöntemin ilk 10 sonucu birleştirilir ve tekilleştirilir.
6. Pilot (--asama pilot): asıl 50 sorgu aynı kurallarla çekildikten SONRA, kalan uygun belgelerden farklı tohumla
   (2511) ve 5 farklı konu kümesinden 5 sorgu seçilir. Pilot sorgular asıl çalışmada yer almaz. Pilot dosyalarında
   ayrı kimlik öneki (P-), ayrı havuz kimliği ve "Aşama: PİLOT" bulunur; analiz, aşaması uymayan dosyayı reddeder.
7. Körleme:
   - Uzman dosyalarında yöntem adı, sıra ve puan yoktur. Çift kimlikleri rastgeledir.
   - Adaylar her sorgu içinde ve her uzman için ayrı tohumla karıştırılır.
   - Eşleştirme anahtarı (anahtar.json) yalnızca araştırmacıda kalır.

    python uzman_dogrulama/havuz_olustur.py --onnx-klasor DIR --veri TRAll.txt --fizibilite fizibilite_2.5 \
        --cikti GIZLI_KLASOR --ozet cikti
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
from tmlab import anlamsal as an  # noqa: E402
from tmlab.analysis import tfidf_hesapla  # noqa: E402
from tmlab.kalite import ingilizce_iceren  # noqa: E402
from tmlab.preprocess import OnIslemeAyarlari, derlem_on_isle, turkce_kucuk_harf  # noqa: E402
from uzman_dogrulama import ortak as o  # noqa: E402

TEMEL36 = "0123456789ABCDFGHJKLMNPQRSTVWXYZ"  # I, O (karışır) ve E (ör. "E5" yöntem adını andırmasın) yok


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", turkce_kucuk_harf(s)).strip()


def kumele(anahtarlar, k: int, tohum: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    sozcuk = [[w for a in sorted(ks) for w in _norm(a).split()] or ["_"] for ks in anahtarlar]
    X = TfidfVectorizer(analyzer=lambda x: x, lowercase=False).fit_transform(sozcuk)
    return KMeans(n_clusters=k, n_init=10, random_state=tohum).fit_predict(X)


def sorgulari_sec(uygun: np.ndarray, kume: np.ndarray, n: int, tohum: int) -> list:
    rng = np.random.default_rng(tohum)
    k = int(kume.max()) + 1
    pay, secilen = n // k, []
    for c in range(k):
        aday = np.flatnonzero(uygun & (kume == c))
        secilen += [int(x) for x in rng.choice(aday, min(pay, len(aday)), replace=False)]
    kalan = np.setdiff1d(np.flatnonzero(uygun), secilen)
    if len(secilen) < n:
        secilen += [int(x) for x in rng.choice(kalan, n - len(secilen), replace=False)]
    return sorted(secilen)


def siralamalar(ozetler, basliklar, sorgular, kod, k: int = o.ILK_K):
    """Her yöntem için {sorgu: [(belge, skor), …ilk k]}; sorgunun kendisi ve yakın kopyaları dışlanır."""
    N = len(ozetler)
    tf = tfidf_hesapla(derlem_on_isle(ozetler, OnIslemeAyarlari()), [str(i) for i in range(N)], 1, True, False, 1)
    if tf.analiz_edilen_belge != N:
        raise o.DogrulamaHatasi("Ön işleme sonrası boş kalan belge var; derlem hizalanamıyor.")
    V = kod.kodla(ozetler, an.SORGU_ONEKI)
    q = np.array(sorgular)
    S = {"TF-IDF": (tf.matris[q] @ tf.matris.T).toarray(), "E5": V[q] @ V.T}
    for M in S.values():
        M[np.arange(len(q)), q] = -np.inf
    S["Hibrit"] = an.hibrit(S["TF-IDF"], S["E5"])
    bas, oz = [_norm(b) for b in basliklar], [_norm(x)[:200] for x in ozetler]
    sonuc, kopya = {y: {} for y in o.YONTEMLER}, {}
    for r, s in enumerate(sorgular):
        haric = [j for j in range(N) if j == s or bas[j] == bas[s] or oz[j] == oz[s]]
        kopya[s] = len(haric) - 1
        for y in o.YONTEMLER:
            satir = S[y][r].copy()
            satir[haric] = -np.inf
            ust = np.argsort(-satir, kind="stable")[:k]
            sonuc[y][s] = [(int(j), float(satir[j])) for j in ust]
    return sonuc, kopya


def pilot_sec(uygun: np.ndarray, kume: np.ndarray, asil: list, n: int, tohum: int) -> list:
    rng = np.random.default_rng(tohum)
    kalan = uygun.copy()
    kalan[asil] = False
    kumeler = [c for c in range(int(kume.max()) + 1) if (kalan & (kume == c)).any()]
    secilen = [int(rng.choice(np.flatnonzero(kalan & (kume == c)))) for c in rng.permutation(kumeler)[:n]]
    if len(secilen) < n:
        secilen += [int(x) for x in rng.choice(np.setdiff1d(np.flatnonzero(kalan), secilen), n - len(secilen), replace=False)]
    return sorted(secilen)


def havuz_kur(sorgular, sira, tohum: int, onek: str = "Ç-"):
    """Tekilleştirilmiş sorgu–aday çiftleri, rastgele kimlikler ve rastgele sorgu numaraları."""
    rng = np.random.default_rng(tohum + 1)
    sorgu_no = {s: i + 1 for i, s in enumerate(rng.permutation(sorgular))}
    ciftler = {}
    for s in sorgular:
        for y in o.YONTEMLER:
            for r, (d, puan) in enumerate(sira[y][s], 1):
                c = ciftler.setdefault((s, d), {"sorgu_no": sorgu_no[s], "sorgu_belge": int(s), "aday_belge": d,
                                                "siralar": {}, "puanlar": {}})
                c["siralar"][y], c["puanlar"][y] = r, round(puan, 6)
    kodlar = rng.choice(len(TEMEL36) ** 5, len(ciftler), replace=False)
    for c, kod in zip(ciftler.values(), kodlar):  # kodlar rastgele; yöntem ya da sırayla ilişkisiz
        k, s = int(kod), ""
        for _ in range(5):
            k, r = divmod(k, len(TEMEL36))
            s = TEMEL36[r] + s
        c["cift"] = onek + s
    return list(ciftler.values()), sorgu_no


def uzman_satirlari(ciftler, basliklar, ozetler, uzman: int, tohum: int):
    rng = np.random.default_rng(tohum + 100 * uzman)
    satir = []
    for no in sorted({c["sorgu_no"] for c in ciftler}):
        grup = [c for c in ciftler if c["sorgu_no"] == no]
        for i in rng.permutation(len(grup)):
            c = grup[i]
            s, d = c["sorgu_belge"], c["aday_belge"]
            satir.append([c["cift"], no, basliklar[s], ozetler[s], basliklar[d], ozetler[d], None, None,
                          o.denetim_kodu(c["cift"], ozetler[s], ozetler[d])])
    return satir


def excel_yaz(yol, satirlar, uzman: int, havuz_kimligi: str, yonerge: str, asama: str = "asil"):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Protection
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ys = wb.active
    ys.title = o.YONERGE_SAYFASI
    bilgi = [f"Uzman kodu: {uzman}", f"Havuz kimliği: {havuz_kimligi}", f"Aşama: {o.ASAMALAR[asama]}",
             f"Değerlendirilecek çift: {len(satirlar)}",
             f"Sorgu sayısı: {len({r[1] for r in satirlar})}", ""]
    for satir in bilgi + [re.sub(r"[*`#]", "", x) for x in yonerge.splitlines()]:
        ys.append([satir])
    ys.column_dimensions["A"].width = 120
    for h in ys["A"]:
        h.alignment = Alignment(wrap_text=True, vertical="top")
    ys["A1"].font = ys["A2"].font = Font(bold=True)
    ws = wb.create_sheet(o.SAYFA)
    ws.append(o.SUTUNLAR)
    for r in satirlar:
        ws.append(r)
    for i, g in zip("ABCDEFGHI", (11, 8, 30, 60, 30, 60, 12, 28, 12)):
        ws.column_dimensions[i].width = g
    sari = PatternFill("solid", fgColor="FFF2CC")
    for satir in ws.iter_rows(min_row=2):
        for h in satir:
            h.alignment = Alignment(wrap_text=True, vertical="top")
        for h in satir[6:8]:
            h.protection = Protection(locked=False)
            h.fill = sari
    for h in ws[1]:
        h.font = Font(bold=True)
    dv = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Geçersiz puan", error="Yalnızca 0, 1 ya da 2 girin.")
    ws.add_data_validation(dv)
    dv.add(f"G2:G{len(satirlar) + 1}")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:I{len(satirlar) + 1}"
    ws.protection.sheet = True
    ws.protection.autoFilter = False  # süzmeye izin ver
    wb.active = 1
    wb.save(yol)


def ana():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-klasor", required=True)
    p.add_argument("--veri", required=True)
    p.add_argument("--fizibilite", required=True)
    p.add_argument("--cikti", required=True, help="GİZLİ klasör (metin içerir; şifrelenmeden paylaşılmaz)")
    p.add_argument("--ozet", required=True, help="metin içermeyen özet (artifact'e konabilir)")
    p.add_argument("--n", type=int, default=o.SORGU_SAYISI)
    p.add_argument("--kume", type=int, default=o.KUME_SAYISI)
    p.add_argument("--asama", choices=list(o.ASAMALAR), required=True)
    a = p.parse_args()
    gizli, acik = os.path.realpath(a.cikti), os.path.realpath(a.ozet)
    if gizli == acik or gizli.startswith(acik + os.sep) or acik.startswith(gizli + os.sep):
        raise SystemExit("--cikti (metin içerir) ile --ozet (paylaşılabilir) ayrı klasörler olmalı.")
    sys.path.append(a.fizibilite)
    import anlamsal_karsilastirma as ak  # yalnızca veri okuma (fizibilite ile aynı kurallar)

    havuz_n = ak.veri_oku(a.veri, 2)[3]
    basliklar, ozetler, anahtarlar, _ = ak.veri_oku(a.veri, havuz_n)  # tüm uygun belgeler, özgün sırada
    onceki = set(ak.veri_oku(a.veri, 500)[1])
    kod = an.E5Kodlayici(a.onnx_klasor)
    ing = set(ingilizce_iceren(ozetler))
    uzun = kod.belirtec_sayilari(ozetler, an.SORGU_ONEKI) > kod.azami_token
    uygun = np.array([(x not in onceki) and (i not in ing) and not uzun[i] for i, x in enumerate(ozetler)])
    kume = kumele(anahtarlar, a.kume, o.SORGU_TOHUMU)
    asil = sorgulari_sec(uygun, kume, a.n, o.SORGU_TOHUMU)  # pilotta da önce asıl sorgular çekilir ve dışlanır
    pilot = a.asama == "pilot"
    sorgular = pilot_sec(uygun, kume, asil, o.PILOT_SAYISI, o.PILOT_TOHUMU) if pilot else asil
    assert not pilot or not set(sorgular) & set(asil)
    tohum = o.PILOT_TOHUMU if pilot else o.SORGU_TOHUMU
    sira, kopya = siralamalar(ozetler, basliklar, sorgular, kod)
    ciftler, sorgu_no = havuz_kur(sorgular, sira, tohum, o.KIMLIK_ONEKI[a.asama])
    kimlik = o.havuz_ozeti((c["cift"] for c in ciftler), a.asama)
    on = "pilot_" if pilot else ""
    os.makedirs(os.path.join(a.cikti, "uzmanlara_gonderin"), exist_ok=True)
    os.makedirs(os.path.join(a.cikti, "yalnizca_arastirmaci"), exist_ok=True)
    yonerge = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "YONERGE.md"), encoding="utf-8").read()
    for u in (1, 2):
        excel_yaz(os.path.join(a.cikti, "uzmanlara_gonderin", f"{on}uzman_{u}.xlsx"),
                  uzman_satirlari(ciftler, basliklar, ozetler, u, tohum), u, kimlik, yonerge, a.asama)
    model_sha = kod.kunye.get("dosyalar", {}).get("model_fp32.onnx", {}).get("sha256")
    olusturma = {"asama": a.asama, "kaynak": "DergiPark TRAll.txt (humakilicunlu/DergiParkEn)",
                 "kaynak_lisans_durumu": o.KAYNAK_LISANS, "pilot_asil_sorgu_ortusmesi": 0 if pilot else None,
                 "veri_sha256": hashlib.sha256(open(a.veri, "rb").read()).hexdigest(),
                 "uygun_belge": havuz_n, "sorgu_disi_onceki_orneklem": int(sum(x in onceki for x in ozetler)),
                 "sorgu_disi_ingilizce": len(ing), "sorgu_disi_512_ustu": int(uzun.sum()),
                 "sorgu_olabilir": int(uygun.sum()), "sorgu": len(sorgular), "kume_sayisi": a.kume,
                 "sorgu_kume_dagilimi": {int(c): int((kume[sorgular] == c).sum()) for c in range(a.kume)},
                 "yakin_kopya_dislanan": {int(sorgu_no[s]): kopya[s] for s in sorgular if kopya[s]},
                 "ilk_k": o.ILK_K, "tohum": tohum, "model_sha256": model_sha,
                 "tfidf": "varsayılan ön işleme, tek sözcük, yumuşatılmış IDF, L2", "e5": "ONNX FP32, iki taraf 'query: '",
                 "hibrit": "satır başına z-puanı toplamı (sorgunun kendisi hariç)", "cift_sayisi": len(ciftler)}
    siralama = {str(sorgu_no[s]): {y: [next(c["cift"] for c in ciftler if c["sorgu_belge"] == s and c["aday_belge"] == d)
                                       for d, _ in sira[y][s]] for y in o.YONTEMLER} for s in sorgular}
    o.json_yaz(os.path.join(a.cikti, "yalnizca_arastirmaci", f"{on}anahtar.json"),
               {"surum": 1, "asama": a.asama, "havuz_kimligi": kimlik, "olusturma": olusturma,
                "ciftler": [{**c, "denetim": o.denetim_kodu(c["cift"], ozetler[c["sorgu_belge"]], ozetler[c["aday_belge"]])}
                            for c in ciftler], "siralamalar": siralama})
    havuz_boyu = [sum(c["sorgu_no"] == n for c in ciftler) for n in range(1, len(sorgular) + 1)]
    ortusme = {f"{x}–{y}": round(float(np.mean([len(set(siralama[q][x]) & set(siralama[q][y])) / o.ILK_K
                                                 for q in siralama])), 3)
               for i, x in enumerate(o.YONTEMLER) for y in o.YONTEMLER[i + 1:]}
    ozet = {**{k: v for k, v in olusturma.items() if k != "yakin_kopya_dislanan"}, "havuz_kimligi": kimlik,
            "yalnizca_tek_yontemden_gelen_aday": {y: sum(set(c["siralar"]) == {y} for c in ciftler) for y in o.YONTEMLER},
            "sorgu_basina_aday": {"ortalama": round(float(np.mean(havuz_boyu)), 1), "en_az": min(havuz_boyu),
                                  "en_cok": max(havuz_boyu)}, "ilk10_ortusmesi": ortusme}
    os.makedirs(a.ozet, exist_ok=True)
    o.json_yaz(os.path.join(a.ozet, f"uzman_havuz_ozeti_{a.asama}.json"), ozet)
    print(ozet)


if __name__ == "__main__":
    ana()