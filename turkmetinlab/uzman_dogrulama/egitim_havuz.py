"""Eğitim Programları ve Öğretim derleminden üç insan uzmanlı kör değerlendirme havuzu (pilot ya da asıl).

Girdi: egitim_derlem.py çıktısı (gizli .jsonl). Hesaplama, DergiPark YZ pilotuyla AYNI işlevlerle yapılır
(havuz_olustur.siralamalar / havuz_kur / uzman_satirlari / excel_yaz). Değişmeyenler:
- yöntemler (TF-IDF, E5 ONNX FP32, hibrit);
- ilk 10 sonuç;
- sorgunun kendisinin ve yakın kopyalarının dışlanması;
- körleme;
- +0,030 karar eşiği.

Sorgu seçimi:
1. **Sorgu olabilir:** Türkçe ve en çok 512 E5 belirteci.
2. **Asıl çalışma:** 50 sorgu, birincil alt alanlara eşit paylı tabakalı seçimle önce çekilir (tohum 2520).
3. **Pilot:** Kalan kayıtlardan, mümkün olduğunca farklı alt alanlardan 5 sorgu seçilir (tohum 2521). Pilot
   sorgular asıl çalışmaya girmez.

**Farklı zamanda çalıştırma:** Asıl çalışma pilottan sonra yeni bir derlem toplamasıyla çalışabilir. Bu durumda
pilot sorgularının kaynak kimlikleri (açık özetteki `pilot_sorgu_kaynak_kimlikleri`; metin içermez) `--pilot-ozeti`
ile verilir ve bu kayıtlar sorgu olamaz.

**Yeterlilik:** egitim_derlem.yeterlilik koşulları sağlanmazsa hiçbir uzman dosyası üretilmez; çıkış kodu 3.

Çıktılar:
- `--cikti` (gizli): `uzmanlara_gonderin/{pilot_}insan_uzman_{1,2,3}.xlsx` ve
  `yalnizca_arastirmaci/{pilot_}insan_anahtar.json`.
- `--ozet` (açık): `egitim_havuz_ozeti_<aşama>.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
from tmlab import anlamsal as an  # noqa: E402
from tmlab.kalite import ingilizce_iceren  # noqa: E402
from uzman_dogrulama import egitim_derlem as ed  # noqa: E402
from uzman_dogrulama import havuz_olustur as ho  # noqa: E402
from uzman_dogrulama import ortak as o  # noqa: E402

UZMAN_SAYISI = 3


def tabakali_sec(uygun: np.ndarray, tabaka: list, n: int, tohum: int, dislanan=()) -> list:
    """Tabakalara (alt alan) eşit pay; yetmezse kalanlardan rastgele tamamlar."""
    rng = np.random.default_rng(tohum)
    uygun = uygun.copy()
    uygun[list(dislanan)] = False
    adlar = sorted({tabaka[i] for i in np.flatnonzero(uygun)})
    pay, secilen = max(1, n // max(1, len(adlar))), []
    for t in rng.permutation(adlar):
        aday = [i for i in np.flatnonzero(uygun) if tabaka[i] == t and i not in secilen]
        secilen += [int(x) for x in rng.choice(aday, min(pay, len(aday), n - len(secilen)), replace=False)]
    kalan = np.setdiff1d(np.flatnonzero(uygun), secilen)
    if len(secilen) < n:
        secilen += [int(x) for x in rng.choice(kalan, n - len(secilen), replace=False)]
    return sorted(secilen)


def ana(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--derlem", required=True)
    p.add_argument("--onnx-klasor", required=True)
    p.add_argument("--asama", choices=["egitim_pilot", "egitim_asil"], required=True)
    p.add_argument("--cikti", required=True)
    p.add_argument("--ozet", required=True)
    p.add_argument("--n", type=int, default=o.SORGU_SAYISI)
    p.add_argument("--pilot-ozeti", help="asıl çalışmada: pilotun açık özet dosyası (pilot sorguları dışlanır)")
    a = p.parse_args(argv)
    gizli, acik = os.path.realpath(a.cikti), os.path.realpath(a.ozet)
    if gizli == acik or gizli.startswith(acik + os.sep) or acik.startswith(gizli + os.sep):
        raise SystemExit("--cikti (metin içerir) ile --ozet (paylaşılabilir) ayrı klasörler olmalı.")
    kayit = [json.loads(x) for x in open(a.derlem, encoding="utf-8") if x.strip()]
    basliklar, ozetler = [k["baslik"] for k in kayit], [k["ozet"] for k in kayit]
    tabaka = [k["birincil_alt_alan"] for k in kayit]
    kod = an.E5Kodlayici(a.onnx_klasor)
    ing = set(ingilizce_iceren(ozetler))
    uzun = kod.belirtec_sayilari(ozetler, an.SORGU_ONEKI) > kod.azami_token if kayit else np.array([], bool)
    pilot_kimlik = set(json.load(open(a.pilot_ozeti, encoding="utf-8")).get("pilot_sorgu_kaynak_kimlikleri", [])) \
        if a.pilot_ozeti else set()
    uygun = np.array([i not in ing and not uzun[i] and kayit[i]["kaynak_kimligi"] not in pilot_kimlik
                      for i in range(len(kayit))], bool)
    yeter = ed.yeterlilik(kayit, int(uygun.sum()))
    ozet = {"asama": a.asama, "derlem_sha256": hashlib.sha256(open(a.derlem, "rb").read()).hexdigest(),
            "uygun_kayit": len(kayit), "sorgu_olabilir": int(uygun.sum()), "yeterlilik": yeter,
            "pilot_nedeniyle_sorgu_disi": len(pilot_kimlik & {k["kaynak_kimligi"] for k in kayit}),
            "kaynak_lisans_durumu": "Özetlerin yeniden dağıtım koşulları BİLİNMİYOR; yalnızca şifreli paket."}
    os.makedirs(a.ozet, exist_ok=True)
    yol = os.path.join(a.ozet, f"egitim_havuz_ozeti_{a.asama}.json")
    if not yeter["yeterli"]:
        ozet["durum"] = "YETERSİZ VERİ: uzman dosyaları üretilmedi."
        o.json_yaz(yol, ozet)
        print(json.dumps(ozet, ensure_ascii=False))
        return 3
    asil = tabakali_sec(uygun, tabaka, a.n, o.EGITIM_TOHUMU)
    pilot = a.asama == "egitim_pilot"
    sorgular = tabakali_sec(uygun, tabaka, o.PILOT_SAYISI, o.EGITIM_PILOT_TOHUMU, dislanan=asil) if pilot else asil
    assert not pilot or not set(sorgular) & set(asil)
    tohum = o.EGITIM_PILOT_TOHUMU if pilot else o.EGITIM_TOHUMU
    sira, kopya = ho.siralamalar(ozetler, basliklar, sorgular, kod)
    ciftler, sorgu_no = ho.havuz_kur(sorgular, sira, tohum, o.KIMLIK_ONEKI[a.asama])
    kimlik = o.havuz_ozeti((c["cift"] for c in ciftler), a.asama)
    on = "pilot_" if pilot else ""
    os.makedirs(os.path.join(a.cikti, "uzmanlara_gonderin"), exist_ok=True)
    os.makedirs(os.path.join(a.cikti, "yalnizca_arastirmaci"), exist_ok=True)
    yonerge = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "YONERGE_EGITIM.md"), encoding="utf-8").read()
    for u in range(1, UZMAN_SAYISI + 1):
        ho.excel_yaz(os.path.join(a.cikti, "uzmanlara_gonderin", f"{on}insan_uzman_{u}.xlsx"),
                     ho.uzman_satirlari(ciftler, basliklar, ozetler, u, tohum), u, kimlik, yonerge, a.asama)
    sorgu_bilgi = {str(sorgu_no[s]): {"kaynak_kimligi": kayit[s]["kaynak_kimligi"], "alt_alan": tabaka[s]} for s in sorgular}
    for c in ciftler:
        c["aday_kaynak_kimligi"] = kayit[c["aday_belge"]]["kaynak_kimligi"]
    siralama = {str(sorgu_no[s]): {y: [next(c["cift"] for c in ciftler if c["sorgu_belge"] == s and c["aday_belge"] == d)
                                       for d, _ in sira[y][s]] for y in o.YONTEMLER} for s in sorgular}
    olusturma = {"asama": a.asama, "uzman_sayisi": UZMAN_SAYISI, "derlem": "egitim_derlem.py (OpenAlex)",
                 "derlem_sha256": ozet["derlem_sha256"], "tohum": tohum, "ilk_k": o.ILK_K,
                 "model_sha256": kod.kunye.get("dosyalar", {}).get("model_fp32.onnx", {}).get("sha256"),
                 "sorgular": sorgu_bilgi, "yakin_kopya_dislanan": {str(sorgu_no[s]): kopya[s] for s in sorgular if kopya[s]},
                 "pilot_asil_sorgu_ortusmesi": 0 if pilot else None, "cift_sayisi": len(ciftler),
                 "kaynak_lisans_durumu": ozet["kaynak_lisans_durumu"]}
    o.json_yaz(os.path.join(a.cikti, "yalnizca_arastirmaci", f"{on}insan_anahtar.json"),
               {"surum": 1, "asama": a.asama, "havuz_kimligi": kimlik, "olusturma": olusturma,
                "ciftler": [{**c, "denetim": o.denetim_kodu(c["cift"], ozetler[c["sorgu_belge"]], ozetler[c["aday_belge"]])}
                            for c in ciftler], "siralamalar": siralama})
    boy = [sum(c["sorgu_no"] == n for c in ciftler) for n in range(1, len(sorgular) + 1)]
    ozet.update({"durum": "üretildi", "havuz_kimligi": kimlik, "sorgu": len(sorgular), "cift_sayisi": len(ciftler),
                 "sorgu_alt_alanlari": dict(Counter(tabaka[s] for s in sorgular)),
                 **({"pilot_sorgu_kaynak_kimlikleri": sorted(kayit[s]["kaynak_kimligi"] for s in sorgular)} if pilot else {}),
                 "sorgu_basina_aday": {"ortalama": round(float(np.mean(boy)), 1), "en_az": min(boy), "en_cok": max(boy)},
                 "yalnizca_tek_yontemden_gelen_aday": {y: sum(set(c["siralar"]) == {y} for c in ciftler) for y in o.YONTEMLER}})
    o.json_yaz(yol, ozet)
    print(json.dumps(ozet, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(ana())