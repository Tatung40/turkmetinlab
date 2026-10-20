"""Eğitim Programları ve Öğretim alanında doğrulanabilir Türkçe makale özetleri derlemi (OpenAlex; GitHub Actions'ta).

Kaynak: OpenAlex Works API (https://api.openalex.org, ücretsiz, anahtarsız).
- OpenAlex üst verileri CC0 lisanslıdır. Özetler ise yayıncılardan gelir ve **yeniden dağıtım koşulları
  bilinmiyor.** Bu nedenle derlem yalnızca şifreli pakette tutulur; depoya ve açık artifact'e konmaz.
- DergiPark kayıtları, erişim adresinden (dergipark.org.tr) ayrıca işaretlenir.

Arama: dil = Türkçe, özeti olan, tür = makale, geri çekilmemiş. Her alan ifadesi için ayrı tam ifade araması
yapılır ve sonuçlar OpenAlex kimliğiyle birleştirilir (ARAMA_IFADELERI).

Dahil etme (önceden belirlendi):
1. **Doğrulanabilir kayıt:** OpenAlex kimliği, başlık, özet ve erişim adresi (DOI ya da yayıncı sayfası) var.
2. **Özet:** Türkçe, en az 50 sözcük, belirgin İngilizce içermiyor.
3. **Alan ölçütü:** Başlık + özette en az iki farklı alan ifadesi geçiyor (ALT_ALANLAR). Bunlardan en az biri
   "çekirdek" ifade olmalı. Başlıkta yalnızca "eğitim" sözcüğünün geçmesi yetmez; "eğitim" tek başına alan
   ifadesi sayılmaz.

Dışlama:
- yinelenen kayıt (DOI, OpenAlex kimliği ya da normalleştirilmiş başlık);
- özeti başlığın tekrarı olan kayıt;
- 1.500 sözcüğü aşan "özet" (tam metin parçası olabilir).

**Alt alan:** İfade eşleşmesine dayalı, kural tabanlı bir tahmindir; uzman sınıflaması değildir. Bir kaydın birincil
alt alanı, en çok ifade eşleşen alt alandır.

**Yeterlilik** (önceden belirlendi):
- en az 300 uygun kayıt;
- sorgu olabilecek (≤ 512 belirteç) en az 55 kayıt (50 asıl + 5 pilot);
- en az 4 alt alanda en az 10'ar kayıt.

Bu koşullar sağlanmazsa çıkış kodu 3 olur ve pilot üretilmez.

    python uzman_dogrulama/egitim_derlem.py --cikti GIZLI/derlem.jsonl --ozet cikti
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
from tmlab.kalite import ingilizce_iceren  # noqa: E402
from tmlab.preprocess import turkce_kucuk_harf  # noqa: E402

ADRES = os.environ.get("TML_OPENALEX_URL") or "https://api.openalex.org"
EN_AZ_KAYIT, EN_AZ_SORGU_ADAYI, EN_AZ_ALT_ALAN, ALT_ALAN_EN_AZ = 300, 55, 4, 10
EN_AZ_SOZCUK, EN_COK_SOZCUK = 50, 1500
SAYFA_SINIRI = 25  # ifade başına en çok 25 × 200 kayıt

ALT_ALANLAR = {  # (çekirdek ifadeler, destekleyici ifadeler) — küçük harf, Türkçe
    "Program geliştirme ve değerlendirme": (
        ["öğretim programı", "öğretim programları", "eğitim programı", "eğitim programları", "program geliştirme",
         "program değerlendirme", "programın değerlendirilmesi", "müfredat", "ders programı"],
        ["kazanım", "program öğeleri", "hedef", "içerik", "sınama durumları", "stufflebeam", "cipp"]),
    "Öğretim tasarımı": (
        ["öğretim tasarımı", "öğretimsel tasarım", "ders tasarımı", "tasarım tabanlı araştırma"],
        ["addie", "öğretim materyali", "materyal geliştirme", "tasarım ilkeleri"]),
    "Öğrenme-öğretme süreçleri": (
        ["öğrenme-öğretme süreci", "öğretme-öğrenme süreci", "öğrenme öğretme süreci", "öğretme öğrenme süreci",
         "öğretim süreci", "sınıf içi öğretim"],
        ["öğrenme ortamı", "ders işleniş", "sınıf içi", "öğretmen davranışları"]),
    "Öğretim stratejileri ve yöntemleri": (
        ["öğretim stratejisi", "öğretim stratejileri", "öğretim yöntemi", "öğretim yöntemleri", "öğretim modeli",
         "öğretim teknikleri", "ters yüz", "probleme dayalı öğrenme", "proje tabanlı öğrenme", "işbirlikli öğrenme",
         "harmanlanmış öğrenme", "farklılaştırılmış öğretim", "5e öğrenme modeli", "argümantasyon"],
        ["akademik başarı", "deney grubu", "kontrol grubu", "tutum"]),
    "Program uygulamaları": (
        ["programın uygulanması", "program uygulamaları", "programın uygulanmasına", "uygulanan program",
         "programa ilişkin öğretmen görüşleri", "programa yönelik öğretmen görüşleri"],
        ["öğretmen görüşleri", "uygulama sürecinde", "karşılaşılan sorunlar"]),
}
ARAMA_IFADELERI = sorted({i for c, _ in ALT_ALANLAR.values() for i in c})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", turkce_kucuk_harf(s or "")).strip()


def ozet_kur(ters_dizin: dict | None) -> str:
    """OpenAlex abstract_inverted_index → düz metin."""
    if not ters_dizin:
        return ""
    konum = {i: sozcuk for sozcuk, yerler in ters_dizin.items() for i in yerler}
    return " ".join(konum[i] for i in sorted(konum))


def alan_esles(baslik: str, ozet: str) -> dict:
    """{alt_alan: (çekirdek eşleşme sayısı, toplam eşleşme sayısı)}; 'eğitim' tek başına sayılmaz."""
    metin = _norm(baslik) + " " + _norm(ozet)
    sonuc = {}
    for alan, (cekirdek, destek) in ALT_ALANLAR.items():
        c = sum(i in metin for i in cekirdek)
        d = sum(i in metin for i in destek)
        if c or d:
            sonuc[alan] = (c, c + d)
    return sonuc


def dahil_mi(kayit: dict) -> tuple[bool, str]:
    if not (kayit.get("kaynak_kimligi") and kayit.get("baslik") and kayit.get("ozet") and kayit.get("erisim_adresi")):
        return False, "doğrulanamayan kayıt (kimlik/başlık/özet/erişim adresi eksik)"
    n = len(kayit["ozet"].split())
    if n < EN_AZ_SOZCUK:
        return False, "özet 50 sözcükten kısa"
    if n > EN_COK_SOZCUK:
        return False, "özet 1.500 sözcükten uzun"
    if _norm(kayit["ozet"]) == _norm(kayit["baslik"]):
        return False, "özet başlığın tekrarı"
    if ingilizce_iceren([kayit["ozet"]]):
        return False, "özet belirgin İngilizce içeriyor"
    es = alan_esles(kayit["baslik"], kayit["ozet"])
    ifade = sum(t for _, t in es.values())
    if ifade < 2 or not any(c for c, _ in es.values()):
        return False, "alan ölçütü sağlanmadı (en az 2 ifade, en az 1 çekirdek)"
    return True, ""


def openalex_kayit(w: dict) -> dict:
    yer = w.get("primary_location") or {}
    kaynak = (yer.get("source") or {})
    adres = yer.get("landing_page_url") or w.get("doi") or ""
    return {"kaynak_kimligi": (w.get("id") or "").rsplit("/", 1)[-1], "doi": w.get("doi"),
            "baslik": (w.get("display_name") or w.get("title") or "").strip(),
            "ozet": ozet_kur(w.get("abstract_inverted_index")).strip(), "yil": w.get("publication_year"),
            "dergi": kaynak.get("display_name"), "erisim_adresi": adres,
            "platform": "DergiPark" if "dergipark" in adres.lower() else "diğer",
            "alan_bilgisi": [t.get("display_name") for t in (w.get("topics") or [])][:3]}


def _getir(url: str) -> dict:
    """Tek deneme; 429/5xx'te bir kez 10 sn bekleyip yeniden dener, sonra durur."""
    for deneme in (1, 2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "TurkMetinLab-dogrulama"}),
                                        timeout=60) as y:
                return json.load(y)
        except urllib.error.HTTPError as e:
            if deneme == 2 or e.code not in (429, 500, 502, 503):
                raise
            time.sleep(10)


def topla(ifadeler=ARAMA_IFADELERI, bekle: float = 0.2) -> tuple[list, dict]:
    ham, sayac = {}, {}
    for ifade in ifadeler:
        imlec, sayfa = "*", 0
        while imlec and sayfa < SAYFA_SINIRI:
            q = urllib.parse.urlencode({"filter": "language:tr,has_abstract:true,type:article,is_retracted:false",
                                        "search": f'"{ifade}"', "per-page": 100, "cursor": imlec,
                                        "select": "id,doi,display_name,title,abstract_inverted_index,publication_year,"
                                                  "primary_location,topics"})
            anahtar = os.environ.get("OPENALEX_API_KEY", "").strip()
            if not anahtar:
                raise RuntimeError("OPENALEX_API_KEY GitHub Actions secret tanımlanmamış.")
            q += "&" + urllib.parse.urlencode({"api_key": anahtar})
            veri = _getir(f"{ADRES}/works?{q}")
            for w in veri.get("results", []):
                ham.setdefault(w.get("id"), w)
            imlec = (veri.get("meta") or {}).get("next_cursor")
            sayfa += 1
            sayac[ifade] = sayac.get(ifade, 0) + len(veri.get("results", []))
            if not veri.get("results"):
                break
            time.sleep(bekle)
    return [openalex_kayit(w) for w in ham.values()], sayac


def suz(kayitlar: list) -> tuple[list, Counter]:
    neden, secilen, gorulen = Counter(), [], set()
    for k in kayitlar:
        anahtar = {("kimlik", k.get("kaynak_kimligi")), ("doi", (k.get("doi") or "").lower()),
                   ("baslik", re.sub(r"\W+", "", turkce_kucuk_harf(k.get("baslik") or "")))} - {("doi", ""), ("kimlik", None)}
        if anahtar & gorulen:
            neden["yinelenen kayıt"] += 1
            continue
        tamam, n = dahil_mi(k)
        if not tamam:
            neden[n] += 1
            continue
        gorulen |= anahtar
        es = alan_esles(k["baslik"], k["ozet"])
        k["alt_alanlar"] = sorted(es)
        k["birincil_alt_alan"] = max(es, key=lambda a: (es[a][1], es[a][0], a))
        secilen.append(k)
    secilen.sort(key=lambda k: k["kaynak_kimligi"])  # belirlenimci sıra
    return secilen, neden


def yeterlilik(kayitlar: list, sorgu_olabilir: int) -> dict:
    alt = Counter(k["birincil_alt_alan"] for k in kayitlar)
    yeterli_alt = [a for a, n in alt.items() if n >= ALT_ALAN_EN_AZ]
    kosul = {"uygun_kayit >= 300": len(kayitlar) >= EN_AZ_KAYIT, "sorgu_olabilir >= 55": sorgu_olabilir >= EN_AZ_SORGU_ADAYI,
             "en_az_10_kayitli_alt_alan >= 4": len(yeterli_alt) >= EN_AZ_ALT_ALAN}
    return {"yeterli": all(kosul.values()), "kosullar": kosul, "alt_alan_dagilimi": dict(alt)}


def ana(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cikti", required=True, help="GİZLİ derlem dosyası (.jsonl; metin içerir)")
    p.add_argument("--ozet", required=True, help="metin içermeyen özet klasörü")
    a = p.parse_args(argv)
    ham, sayac = topla()
    kayitlar, neden = suz(ham)
    os.makedirs(os.path.dirname(os.path.abspath(a.cikti)), exist_ok=True)
    with open(a.cikti, "w", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    ozet = {"kaynak": "OpenAlex Works API", "kaynak_adresi": ADRES, "tarih": time.strftime("%Y-%m-%d"),
            "lisans": "OpenAlex üst verisi CC0; özetlerin yeniden dağıtım koşulları BİLİNMİYOR (yalnızca şifreli tutulur).",
            "arama_ifadeleri": sayac, "ham_tekil_kayit": len(ham), "uygun_kayit": len(kayitlar),
            "dislama_nedenleri": dict(neden), "platform": dict(Counter(k["platform"] for k in kayitlar)),
            "yil_araligi": [min((k["yil"] for k in kayitlar if k["yil"]), default=None),
                            max((k["yil"] for k in kayitlar if k["yil"]), default=None)],
            "dergi_sayisi": len({k["dergi"] for k in kayitlar}),
            "alt_alan_kapsami (birincil, kural tabanlı)": dict(Counter(k["birincil_alt_alan"] for k in kayitlar))}
    os.makedirs(a.ozet, exist_ok=True)
    with open(os.path.join(a.ozet, "egitim_derlem_ozeti.json"), "w", encoding="utf-8") as f:
        json.dump(ozet, f, ensure_ascii=False, indent=1)
    print(json.dumps(ozet, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(ana())