"""TürkMetinLab 2.5 — uzman doğrulaması: ortak sabitler, Excel biçimi, uyum ve bilgi erişim ölçütleri.

Bu modül yalnızca numpy, pandas ve openpyxl kullanır (uygulamanın mevcut bağımlılıkları).

Önceden belirlenen analiz planı (sonuçlar görülmeden yazıldı; değiştirilmez):
* Ölçek: 0 = ilgisiz, 1 = kısmen ilgili, 2 = ilgili.
* Uyum:
  - Birincil: doğrusal ağırlıklı Cohen κ ve doğrusal ağırlıklı Gwet AC2.
  - İkincil: karesel ağırlıklar, ağırlıksız κ ve AC1, ham uyum.
  - Güven aralığı: sorgular üzerinden küme bootstrap.
* Erişim ölçütleri: uzlaşı etiketleriyle nDCG@5 ve nDCG@10; kazanç 2^ilgililik − 1. İdeal sıralama, o sorgunun
  yargılanmış havuzundan hesaplanır.
  - Precision@5/10 iki eşikle verilir: katı (ilgililik = 2) ve gevşek (ilgililik ≥ 1).
  - Havuz, karşılaştırılan üç yöntemin ilk 10 sonucunun birleşimidir. Bu yüzden k ≤ 10 için her sonuç
    yargılanmıştır; yargısız belge ilgisiz sayılmaz, böyle bir belge varsa hata verilir.
* Karşılaştırma: sorgu bazında eşleştirilmiş bootstrap (10.000 yeniden örnekleme, tohum 2501), %95 yüzdelik GA.
  - Pratik önem eşiği: hibrit − TF-IDF nDCG@5 ≥ +0,030 ve GA alt sınırı > 0 (fizibiliteden; değiştirilmez).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Sequence

import numpy as np

OLCEK = (0, 1, 2)
OLCEK_ADLARI = {0: "İlgisiz", 1: "Kısmen ilgili", 2: "İlgili"}
YONTEMLER = ("TF-IDF", "E5", "Hibrit")
REFERANS = "TF-IDF"
PRATIK_ESIK = 0.030
BOOTSTRAP_B = 10_000
BOOTSTRAP_TOHUM = 2501
SORGU_TOHUMU = 2510  # sorgu seçimi ve aday sıralaması (fizibilite örneklem tohumu 2501'den farklı)
SORGU_SAYISI = 50
PILOT_SAYISI = 5
PILOT_TOHUMU = 2511  # pilot sorgular, asıl 50 sorgu çekildikten SONRA kalanlardan seçilir (örtüşme yok)
ASAMALAR = {"pilot": "PİLOT (yönerge denemesi; asıl analize katılmaz)", "asil": "ASIL ÇALIŞMA"}
KIMLIK_ONEKI = {"pilot": "P-", "asil": "Ç-"}
# Eğitim Programları ve Öğretim — üç insan uzmanlı ayrı çalışma (YZ pilotundan bağımsız; ayrı havuz, önek, dosya adı)
ASAMALAR.update({"egitim_pilot": "EĞİTİM İNSAN PİLOTU (yönerge denemesi; asıl analize katılmaz)",
                 "egitim_asil": "EĞİTİM İNSAN ASIL ÇALIŞMA"})
KIMLIK_ONEKI.update({"egitim_pilot": "IP-", "egitim_asil": "IA-"})
EGITIM_TOHUMU, EGITIM_PILOT_TOHUMU = 2520, 2521
KAYNAK_LISANS = ("DergiParkEn deposunda (humakilicunlu/DergiParkEn) lisans dosyası ya da yeniden dağıtım izni "
                 "bulunmuyor; README yalnızca atıf istiyor (Kılıç Ünlü ve Çetin, 2023, Neural Comput & Applic "
                 "35:3413–3422). Özetlerin telif hakları yazarlara/dergilere aittir ve yeniden dağıtım koşulları "
                 "BİLİNMİYOR. Metinler depoya konmaz, yalnızca şifreli olarak ve iç uzman değerlendirmesi için "
                 "kullanılır; yayımlanacak çıktılarda metin değil, kaynak, veri SHA-256 ve sıra numaraları paylaşılmalıdır.")
KUME_SAYISI = 10
ILK_K = 10

SAYFA = "Değerlendirme"
YONERGE_SAYFASI = "Yönerge"
SUTUNLAR = ["Çift kimliği", "Sorgu no", "Sorgu başlığı", "Sorgu özeti", "Aday başlığı", "Aday özeti",
            "Puan (0/1/2)", "Açıklama (isteğe bağlı)", "Denetim kodu"]
YASAK_IFADELER = ("TF-IDF", "TFIDF", "E5", "Hibrit", "hibrit", "kosinüs", "benzerlik puanı", "sıra:")


class DogrulamaHatasi(ValueError):
    """Girdi dosyaları güvenle eşleştirilemiyor ya da eksik."""


def denetim_kodu(cift: str, sorgu: str, aday: str) -> str:
    """Satırın kimliğini metinlerine bağlar (yanlış eşleşen satır ya da dosya bulunur)."""
    return hashlib.sha256(f"{cift}\x1f{sorgu}\x1f{aday}".encode("utf-8")).hexdigest()[:10]


def havuz_ozeti(ciftler: Iterable[str], asama: str = "asil") -> str:
    return hashlib.sha256((asama + "\n" + "\n".join(sorted(ciftler))).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Uyum ölçütleri (iki puanlayıcı, sıralı 3 kategori)
# ---------------------------------------------------------------------------

def agirliklar(tur: str, q: int = 3) -> np.ndarray:
    i, j = np.meshgrid(range(q), range(q), indexing="ij")
    if tur == "yok":
        return (i == j).astype(float)
    if tur == "dogrusal":
        return 1 - np.abs(i - j) / (q - 1)
    if tur == "karesel":
        return 1 - ((i - j) / (q - 1)) ** 2
    raise ValueError(tur)


def capraz_tablo(a: Sequence[int], b: Sequence[int], q: int = 3) -> np.ndarray:
    T = np.zeros((q, q))
    for x, y in zip(a, b):
        T[int(x), int(y)] += 1
    return T


def cohen_kappa(a, b, tur: str = "dogrusal") -> float:
    """Ağırlıklı Cohen κ = (po − pe) / (1 − pe); po, pe uyum ağırlıklarıyla (Cohen 1968)."""
    P = capraz_tablo(a, b) / len(a)
    W = agirliklar(tur)
    po = (W * P).sum()
    pe = (W * np.outer(P.sum(1), P.sum(0))).sum()
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def gwet_ac(a, b, tur: str = "dogrusal") -> float:
    """Gwet AC1 (tur="yok") / AC2 (ağırlıklı), iki puanlayıcı (Gwet 2014, Handbook of Inter-Rater Reliability):
    pa = Σ w_kl p_kl;  pe = T_w / (q(q−1)) · Σ_k π_k (1 − π_k);  π_k = (p_k+ + p_+k) / 2;  T_w = Σ w_kl."""
    q = 3
    P = capraz_tablo(a, b) / len(a)
    W = agirliklar(tur)
    pa = (W * P).sum()
    pi = (P.sum(1) + P.sum(0)) / 2
    pe = W.sum() / (q * (q - 1)) * (pi * (1 - pi)).sum()
    return float((pa - pe) / (1 - pe))


# ---------------------------------------------------------------------------
# Bilgi erişim ölçütleri
# ---------------------------------------------------------------------------

def ndcg(siralama: Sequence[str], etiket: Dict[str, float], k: int) -> float:
    """siralama: yöntemin ilk sonuçları; etiket: sorgunun yargılanmış havuzu (belge → ilgililik)."""
    eksik = [d for d in siralama[:k] if d not in etiket]
    if eksik:
        raise DogrulamaHatasi(f"Yargılanmamış belge ilk {k} içinde: {eksik[:3]} (havuz kapsamı eksik).")
    kazanc = lambda r: 2.0 ** np.asarray(r, float) - 1  # noqa: E731
    indirim = 1 / np.log2(np.arange(2, k + 2))
    dcg = (kazanc([etiket[d] for d in siralama[:k]]) * indirim[:len(siralama[:k])]).sum()
    ideal = np.sort(np.asarray(list(etiket.values()), float))[::-1][:k]
    idcg = (kazanc(ideal) * indirim[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else float("nan")  # ilgili belgesi olmayan sorgu: tanımsız


def precision(siralama: Sequence[str], etiket: Dict[str, float], k: int, esik: float) -> float:
    ust = list(siralama[:k])
    if any(d not in etiket for d in ust):
        raise DogrulamaHatasi("Yargılanmamış belge (havuz kapsamı eksik).")
    return float(np.mean([etiket[d] >= esik for d in ust])) if ust else float("nan")


def eslestirilmis_bootstrap(x: np.ndarray, y: np.ndarray, b: int = BOOTSTRAP_B, tohum: int = BOOTSTRAP_TOHUM) -> dict:
    """Sorgu bazında eşleştirilmiş fark (x − y): ortalama ve %95 yüzdelik GA. NaN sorgular birlikte dışlanır."""
    gecerli = ~(np.isnan(x) | np.isnan(y))
    d = (x - y)[gecerli]
    if len(d) == 0:
        return {"fark": float("nan"), "ga": [float("nan")] * 2, "n_sorgu": 0}
    rng = np.random.default_rng(tohum)
    ort = d[rng.integers(0, len(d), (b, len(d)))].mean(1)
    return {"fark": float(d.mean()), "ga": [float(np.percentile(ort, 2.5)), float(np.percentile(ort, 97.5))],
            "n_sorgu": int(len(d))}


def kume_bootstrap_uyum(sorgu: np.ndarray, a: np.ndarray, b_: np.ndarray, fn, b: int = 2000,
                        tohum: int = BOOTSTRAP_TOHUM) -> List[float]:
    """Uyum katsayısı için sorgular (küme) üzerinden bootstrap %95 GA (çiftler sorgu içinde bağımlıdır)."""
    rng = np.random.default_rng(tohum)
    sorgular = np.unique(sorgu)
    gruplar = {s: np.flatnonzero(sorgu == s) for s in sorgular}
    degerler = []
    for _ in range(b):
        idx = np.concatenate([gruplar[s] for s in rng.choice(sorgular, len(sorgular))])
        v = fn(a[idx], b_[idx])
        if np.isfinite(v):
            degerler.append(v)
    return [float(np.percentile(degerler, 2.5)), float(np.percentile(degerler, 97.5))] if degerler else [float("nan")] * 2


def json_yaz(yol: str, veri) -> None:
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=1)