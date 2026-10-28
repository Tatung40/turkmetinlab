"""Konu modellemesi: LDA ve NMF; tutarlılık, çeşitlilik ve kararlılık ölçütleri.

Yöntemler (scikit-learn)
------------------------
* LDA (Blei, Ng & Jordan, 2003): CountVectorizer ile üretilen ham sözcük sayıları; toplu (batch)
  varyasyonel çıkarım; doc_topic_prior = topic_word_prior = 1/k (scikit-learn varsayılanı).
  Konu–sözcük değerleri olasılıktır (her konu satırı 1'e toplanır); belge–konu değerleri belgenin
  konu olasılıklarıdır.
* NMF (Lee & Seung, 1999): L2 normalleştirilmiş TF-IDF matrisi; NNDSVDa başlangıcı; Frobenius kaybı.
  Konu–sözcük ve belge–konu değerleri negatif olmayan ağırlıklardır (olasılık DEĞİL); belge–konu
  ağırlıkları karşılaştırma kolaylığı için her belgede 1'e toplanacak biçimde oranlanır.
Her iki yöntemde random_state sabittir; aynı veri ve ayarlar aynı sonucu verir.

Değerlendirme ölçütleri (aynı derlem üzerinde, belge düzeyinde birlikte görülme)
-------------------------------------------------------------------------------
* NPMI tutarlılığı (Bouma, 2009; Aletras & Stevenson, 2013; Lau vd., 2014): konunun ilk N sözcüğünün
  tüm çiftleri için NPMI = log[P(a,b) / (P(a)P(b))] / −log P(a,b); hiç birlikte geçmeyen çift −1.
  Ortalama alınır; −1 … 1, yüksek = sözcükler aynı belgelerde birlikte geçme eğiliminde.
* UMass tutarlılığı (Mimno vd., 2011): ilk N sözcük konu ağırlığına göre sıralı; çiftler için
  log[(D(w_m, w_l) + 1) / D(w_l)] (l < m) ortalaması; 0'a yakın = daha tutarlı.
* Konu çeşitliliği (Dieng, Ruiz & Blei, 2020): tüm konuların ilk 25 sözcüğü içinde benzersiz
  sözcüklerin oranı; 1 = konular sözcük paylaşmıyor.
* Kararlılık (Greene, O'Callaghan & Cunningham, 2014): model, belgelerin rastgele %80'i üzerinde
  yeniden kurulur; konular, ilk N sözcük listelerinin Ortalama Jaccard (AJ) benzerliğine göre
  Macar algoritmasıyla eşleştirilir. 1 = alt örneklemlerde aynı konular yeniden bulunuyor.
* LDA eğitim (örneklem içi) şaşkınlığı ve NMF yeniden oluşturma hatası raporlanır; şaşkınlık insan
  yorumlanabilirliğiyle zayıf ilişkilidir (Chang vd., 2009), tek başına konu sayısı seçmek için
  kullanılmamalıdır.

Algoritmik konular, araştırmacı tarafından belgeler okunarak yorumlanmadıkça doğrulanmış nitel
temalar DEĞİLDİR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .preprocess import IslenmisBelge

import os
import warnings

# Varsayılan NMF: DergiPark akademik özetlerinde (1.417 belge) NMF, LDA'ya göre daha yüksek NPMI tutarlılığı,
# kararlılık ve yazar anahtar sözcükleriyle örtüşme verdi ve çok daha hızlıydı (docs/bilimsel_degerlendirme_2.0.md).
YONTEMLER = {"nmf": "NMF (matris ayrıştırma; TF-IDF) — önerilen", "lda": "LDA (olasılıksal model; sözcük sayıları)"}

# LDA'nın ücretsiz sunucuda çalıştırılacağı en büyük derlem (analiz edilen sözcük). Ölçüm: 288 bin sözcük,
# 10 konu ≈ 21 sn; 1,4 milyon sözcük ≈ 160 sn (2 çekirdek). Yerel kurulumda TML_LDA_SOZCUK ile değiştirilebilir.
try:
    LDA_SOZCUK_SINIRI = max(1, int(os.environ.get("TML_LDA_SOZCUK", 500_000)))
except ValueError:
    LDA_SOZCUK_SINIRI = 500_000


def sure_tahmini(yontem: str, sozcuk: int, konu_sayisi: int, model_sayisi: int = 1) -> float:
    """Ölçüme dayalı kaba süre tahmini (saniye, ölçüm makinesinde). Bulutta 1–3 kat uzun sürebilir."""
    birim = 21.0 if yontem == "lda" else 2.7  # 288 bin sözcük, 10 konu
    return birim * (sozcuk / 288_000) * (konu_sayisi / 10) * model_sayisi
CESITLILIK_DERINLIGI = 25
EN_FAZLA_KONU = 30

UYARI_METNI = (
    "Konu modelleri, sözcüklerin belgelerde **birlikte kullanılma örüntülerini** istatistiksel olarak özetler. "
    "Bulunan konular **doğrulanmış nitel temalar değildir**: anlamlı olup olmadıklarına, her konunun sözcüklerini "
    "ve en tipik belgelerini okuyarak siz karar vermelisiniz. Konu adlarını siz verirsiniz; sonuçlar konu sayısına, "
    "ön işleme ayarlarına ve rastgele başlangıca duyarlıdır (kararlılık ölçümü bunu sınar)."
)


class KonuHatasi(ValueError):
    pass


@dataclass(frozen=True)
class KonuAyarlari:
    yontem: str = "nmf"
    konu_sayisi: int = 5
    en_az_belge: int = 2
    en_fazla_belge_orani: float = 0.95
    en_fazla_sozcuk: int = 5000
    ust_sozcuk: int = 10
    tohum: int = 42
    lda_yineleme: int = 30
    nmf_yineleme: int = 500

    def __post_init__(self):
        if self.yontem not in YONTEMLER:
            raise ValueError(f"Geçersiz yöntem: {self.yontem}")
        if not 2 <= self.konu_sayisi <= EN_FAZLA_KONU:
            raise ValueError(f"Konu sayısı 2 ile {EN_FAZLA_KONU} arasında olmalıdır.")
        if not 0 < self.en_fazla_belge_orani <= 1:
            raise ValueError("En fazla belge oranı 0 ile 1 arasında olmalıdır.")
        if self.ust_sozcuk < 3:
            raise ValueError("Konu başına en az 3 sözcük gösterilmelidir.")

    def sozluk(self) -> dict:
        return {"yöntem": YONTEMLER[self.yontem], "konu sayısı": self.konu_sayisi,
                "sözcük en az kaç belgede": self.en_az_belge, "sözcük en fazla belge oranı": self.en_fazla_belge_orani,
                "en fazla sözlük büyüklüğü": self.en_fazla_sozcuk, "konu başına sözcük": self.ust_sozcuk,
                "rastgele tohum (random_state)": self.tohum,
                "LDA yineleme": self.lda_yineleme if self.yontem == "lda" else "—",
                "NMF en fazla yineleme": self.nmf_yineleme if self.yontem == "nmf" else "—",
                "LDA önselleri": "doc_topic_prior = topic_word_prior = 1/k (scikit-learn varsayılanı)"
                if self.yontem == "lda" else "—",
                "NMF başlangıcı": "NNDSVDa, Frobenius kaybı" if self.yontem == "nmf" else "—"}


@dataclass
class KonuSonucu:
    ayarlar: KonuAyarlari
    sozcukler: np.ndarray
    konu_sozcuk: np.ndarray      # k × V
    belge_konu: np.ndarray       # n_model × k (LDA: olasılık; NMF: satırı 1'e oranlanmış ağırlık)
    belge_indisleri: np.ndarray  # modele giren belgelerin özgün sıradaki indisleri
    olcutler: Dict[str, float]
    konu_olcutleri: pd.DataFrame
    kararlilik: Optional[Dict] = None

    def ust_sozcukler(self, n: Optional[int] = None) -> List[List[str]]:
        n = n or self.ayarlar.ust_sozcuk
        return [list(self.sozcukler[np.argsort(-satir, kind="stable")[:n]]) for satir in self.konu_sozcuk]


# ---------------------------------------------------------------------------
# Girdi ve uygunluk denetimi
# ---------------------------------------------------------------------------

def _vektorlestir(belgeler: Sequence[IslenmisBelge], ayar: KonuAyarlari, tfidf: bool):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    sinif = TfidfVectorizer if tfidf else CountVectorizer
    v = sinif(analyzer=lambda x: x, lowercase=False, min_df=ayar.en_az_belge, max_df=ayar.en_fazla_belge_orani,
              max_features=ayar.en_fazla_sozcuk)
    terimler = [b.sozcukler for b in belgeler]
    try:
        X = v.fit_transform(terimler)
    except ValueError as e:
        raise KonuHatasi("Seçilen ayarlarla konu modeli için yeterli sözcük kalmadı. 'Bir sözcük en az kaç "
                         "belgede geçsin' değerini düşürmeyi deneyin.") from e
    return X.tocsr(), np.asarray(v.get_feature_names_out(), dtype=object)


def uygunluk_denetimi(belgeler: Sequence[IslenmisBelge], konu_sayisi: int) -> List[str]:
    """Derlemin konu modellemesine uygunluğu için uyarılar (boş liste: belirgin sorun yok)."""
    dolu = [b for b in belgeler if b.sozcukler]
    n = len(dolu)
    uzunluk = np.array([len(b.sozcukler) for b in dolu]) if dolu else np.array([0])
    uyarilar = []
    if n < 3:
        raise KonuHatasi("Konu modellemesi için ön işleme sonrası en az 3 dolu belge gerekir.")
    if konu_sayisi >= n:
        raise KonuHatasi(f"Konu sayısı ({konu_sayisi}) belge sayısından ({n}) küçük olmalıdır.")
    if n < 50:
        uyarilar.append(f"Derlem çok küçük ({n} belge). Konu modelleri genellikle yüzlerce belge gerektirir; bu "
                        "sonuçları yalnızca keşif amaçlı kullanın ve kararlılık değerlerine mutlaka bakın.")
    if uzunluk.sum() < 5000:
        uyarilar.append(f"Toplam sözcük sayısı düşük ({int(uzunluk.sum())}). Konu–sözcük dağılımları güvenilir "
                        "biçimde kestirilemeyebilir.")
    if np.median(uzunluk) < 20:
        uyarilar.append(f"Belgeler çok kısa (ortanca {int(np.median(uzunluk))} sözcük). LDA kısa metinlerde zorlanır; "
                        "NMF'yi de deneyin ve sonuçları karşılaştırın.")
    if konu_sayisi > max(2, n // 5):
        uyarilar.append(f"Konu sayısı belge sayısına göre yüksek ({konu_sayisi} konu / {n} belge). Konular tek tek "
                        "belgeleri yansıtabilir.")
    return uyarilar


# ---------------------------------------------------------------------------
# Ölçütler
# ---------------------------------------------------------------------------

def _ikili(X):
    B = X.copy().tocsc()
    B.data = np.ones_like(B.data)
    return B


def tutarlilik(ust_indisler: Sequence[Sequence[int]], B) -> pd.DataFrame:
    """Her konu için NPMI ve UMass (belge düzeyinde). B: belge × sözcük ikili (csc) matris."""
    D = B.shape[0]
    satirlar = []
    for liste in ust_indisler:
        liste = list(liste)
        alt = B[:, liste]
        ortak = np.asarray((alt.T @ alt).toarray(), dtype=np.float64)
        df = np.diag(ortak).copy()
        npmi, umass = [], []
        for m in range(1, len(liste)):
            for l in range(m):
                p_ab = ortak[l, m] / D
                if p_ab == 0:
                    npmi.append(-1.0)
                else:
                    p_a, p_b = df[l] / D, df[m] / D
                    npmi.append(1.0 if p_ab == 1 else np.log(p_ab / (p_a * p_b)) / -np.log(p_ab))
                umass.append(np.log((ortak[l, m] + 1.0) / df[l]))
        satirlar.append((float(np.mean(npmi)), float(np.mean(umass))))
    return pd.DataFrame(satirlar, columns=["NPMI tutarlılığı", "UMass tutarlılığı"])


def cesitlilik(konu_sozcuk: np.ndarray, derinlik: int = CESITLILIK_DERINLIGI) -> float:
    derinlik = min(derinlik, konu_sozcuk.shape[1])
    ust = np.argsort(-konu_sozcuk, axis=1, kind="stable")[:, :derinlik]
    return float(len(np.unique(ust)) / ust.size)


def ortalama_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Sıralı listeler için Ortalama Jaccard (Greene vd., 2014): derinlik 1..t için Jaccard ortalaması."""
    t = min(len(a), len(b))
    return float(np.mean([len(set(a[:d]) & set(b[:d])) / len(set(a[:d]) | set(b[:d])) for d in range(1, t + 1)]))


def konu_eslestir(referans: Sequence[Sequence[str]], diger: Sequence[Sequence[str]]):
    """Macar algoritmasıyla en iyi eşleşme; (referans konu başına AJ, eşleşen indisler)."""
    from scipy.optimize import linear_sum_assignment

    M = np.array([[ortalama_jaccard(r, d) for d in diger] for r in referans])
    satir, sutun = linear_sum_assignment(-M)
    sonuc = np.zeros(len(referans))
    sonuc[satir] = M[satir, sutun]
    eslesen = np.full(len(referans), -1)
    eslesen[satir] = sutun
    return sonuc, eslesen


# ---------------------------------------------------------------------------
# Model kurma
# ---------------------------------------------------------------------------

def _model_kur(X, ayar: KonuAyarlari, tohum: int):
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        # Yakınsama, uyarı yakalanarak değil modelin n_iter_ özniteliğinden okunur (aşağıda).
        warnings.simplefilter("ignore", ConvergenceWarning)
        if ayar.yontem == "lda":
            from sklearn.decomposition import LatentDirichletAllocation

            m = LatentDirichletAllocation(n_components=ayar.konu_sayisi, learning_method="batch",
                                          max_iter=ayar.lda_yineleme, random_state=tohum)
            W = m.fit_transform(X)
            H = m.components_ / m.components_.sum(axis=1, keepdims=True)
            # evaluate_every=-1 (varsayılan): yakınsama sınanmaz, sabit sayıda yineleme yapılır. Şaşkınlık
            # modelin eğitildiği belgeler üzerinde hesaplanır (örneklem içi); dış geçerlik ölçüsü değildir.
            ek = {"Eğitim şaşkınlığı (perplexity, örneklem içi)": float(m.perplexity(X))}
        else:
            from sklearn.decomposition import NMF

            m = NMF(n_components=ayar.konu_sayisi, init="nndsvda", random_state=tohum, max_iter=ayar.nmf_yineleme)
            W = m.fit_transform(X)
            H = m.components_
            # scikit-learn ConvergenceWarning koşulunun aynısı: n_iter_ == max_iter ve tol > 0.
            ek = {"Yeniden oluşturma hatası": float(m.reconstruction_err_),
                  "NMF yineleme sayısı": int(m.n_iter_),
                  "NMF yakınsadı mı": "hayır (yineleme sınırına ulaşıldı)"
                  if m.n_iter_ >= m.max_iter and m.tol > 0 else "evet"}
    return W, H, ek


def konu_modeli(belgeler: Sequence[IslenmisBelge], ayar: KonuAyarlari, kararlilik_tekrari: int = 0,
                alt_orneklem: float = 0.8) -> KonuSonucu:
    uygunluk_denetimi(belgeler, ayar.konu_sayisi)
    X, sozcukler = _vektorlestir(belgeler, ayar, tfidf=ayar.yontem == "nmf")
    dolu = np.flatnonzero(np.asarray(X.sum(axis=1)).ravel() > 0)
    if len(dolu) <= ayar.konu_sayisi:
        raise KonuHatasi("Sözlük süzgecinden sonra konu sayısından fazla dolu belge kalmadı.")
    if len(sozcukler) < ayar.konu_sayisi * 2:
        raise KonuHatasi("Sözlük çok küçük; konu sayısını azaltın ya da süzgeçleri gevşetin.")
    Xd = X[dolu]
    W, H, ek = _model_kur(Xd, ayar, ayar.tohum)
    toplam = W.sum(axis=1, keepdims=True)
    belge_konu = np.divide(W, toplam, out=np.zeros_like(W), where=toplam > 0)
    n_ust = ayar.ust_sozcuk
    ust_ind = np.argsort(-H, axis=1, kind="stable")[:, :n_ust]
    B = _ikili(Xd)
    konu_olc = tutarlilik(ust_ind, B)
    konu_olc.insert(0, "Konu", np.arange(1, ayar.konu_sayisi + 1))
    baskin = np.argmax(belge_konu, axis=1)
    konu_olc["Baskın olduğu belge"] = [int((baskin == k).sum()) for k in range(ayar.konu_sayisi)]
    konu_olc["Ortalama ağırlık"] = np.round(belge_konu.mean(axis=0), 4)
    olcutler = {
        "Modele giren belge": int(len(dolu)),
        "Modeldışı kalan (boş) belge": int(X.shape[0] - len(dolu)),
        "Sözlük büyüklüğü": int(len(sozcukler)),
        "Ortalama NPMI tutarlılığı": round(float(konu_olc["NPMI tutarlılığı"].mean()), 4),
        "Ortalama UMass tutarlılığı": round(float(konu_olc["UMass tutarlılığı"].mean()), 4),
        f"Konu çeşitliliği (ilk {min(CESITLILIK_DERINLIGI, len(sozcukler))} sözcük)": round(cesitlilik(H), 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ek.items()},
    }
    sonuc = KonuSonucu(ayar, sozcukler, H, belge_konu, dolu, olcutler, konu_olc)
    if kararlilik_tekrari > 0:
        sonuc.kararlilik = kararlilik_olc(belgeler, ayar, sonuc.ust_sozcukler(), kararlilik_tekrari, alt_orneklem)
        sonuc.olcutler["Ortalama kararlılık (AJ)"] = round(sonuc.kararlilik["ortalama"], 4)
        sonuc.konu_olcutleri["Kararlılık (AJ)"] = np.round(sonuc.kararlilik["konu_basina"], 4)
    return sonuc


def kararlilik_olc(belgeler: Sequence[IslenmisBelge], ayar: KonuAyarlari, referans: List[List[str]],
                   tekrar: int = 5, oran: float = 0.8) -> Dict:
    """Alt örneklem kararlılığı: her tekrarda belgelerin %oran'ı rastgele (sabit tohumlu) seçilir."""
    rng = np.random.default_rng(ayar.tohum)
    dolu = [i for i, b in enumerate(belgeler) if b.sozcukler]
    puanlar = []
    for t in range(tekrar):
        secim = np.sort(rng.choice(dolu, size=max(ayar.konu_sayisi + 1, int(round(len(dolu) * oran))), replace=False))
        alt = [belgeler[i] for i in secim]
        try:
            X, sozcukler = _vektorlestir(alt, ayar, tfidf=ayar.yontem == "nmf")
            X = X[np.asarray(X.sum(axis=1)).ravel() > 0]
            _, H, _ = _model_kur(X, ayar, ayar.tohum + t + 1)
        except (KonuHatasi, ValueError):
            continue
        ust = [list(sozcukler[np.argsort(-s, kind="stable")[:ayar.ust_sozcuk]]) for s in H]
        puan, _ = konu_eslestir(referans, ust)
        puanlar.append(puan)
    if not puanlar:
        return {"ortalama": float("nan"), "konu_basina": np.full(len(referans), np.nan), "tekrar": 0}
    P = np.array(puanlar)
    return {"ortalama": float(P.mean()), "konu_basina": P.mean(axis=0), "tekrar": len(puanlar),
            "tekrar_ortalamalari": P.mean(axis=1).round(4).tolist()}


def konu_sayisi_karsilastir(belgeler: Sequence[IslenmisBelge], ayar: KonuAyarlari, konu_sayilari: Sequence[int],
                            kararlilik_tekrari: int = 3, ilerleme=None) -> pd.DataFrame:
    satirlar = []
    for s, k in enumerate(konu_sayilari):
        if ilerleme:
            ilerleme(s / len(konu_sayilari), k)
        try:
            r = konu_modeli(belgeler, replace(ayar, konu_sayisi=int(k)), kararlilik_tekrari)
        except KonuHatasi as e:
            satirlar.append({"Konu sayısı": int(k), "Not": str(e)})
            continue
        d = {"Konu sayısı": int(k)}
        d.update({a: v for a, v in r.olcutler.items() if a not in ("Modele giren belge", "Modeldışı kalan (boş) belge",
                                                                    "Sözlük büyüklüğü")})
        satirlar.append(d)
    if ilerleme:
        ilerleme(1.0, None)
    return pd.DataFrame(satirlar)


# ---------------------------------------------------------------------------
# Tablolar ve grafikler
# ---------------------------------------------------------------------------

def konu_adlari(sonuc: KonuSonucu, ozel: Optional[Dict[int, str]] = None) -> List[str]:
    ozel = ozel or {}
    return [(ozel.get(k) or "").strip() or f"Konu {k + 1}" for k in range(sonuc.ayarlar.konu_sayisi)]


def konu_sozcuk_tablosu(sonuc: KonuSonucu, adlar: Sequence[str]) -> pd.DataFrame:
    deger_adi = "Olasılık" if sonuc.ayarlar.yontem == "lda" else "Ağırlık"
    satirlar = []
    for k, satir in enumerate(sonuc.konu_sozcuk):
        ust = np.argsort(-satir, kind="stable")[:sonuc.ayarlar.ust_sozcuk]
        for r, j in enumerate(ust, start=1):
            satirlar.append((k + 1, adlar[k], r, sonuc.sozcukler[j], round(float(satir[j]), 6)))
    return pd.DataFrame(satirlar, columns=["Konu", "Konu adı", "Sıra", "Sözcük", deger_adi])


def belge_konu_tablosu(sonuc: KonuSonucu, belge_adlari: Sequence[str], adlar: Sequence[str]) -> pd.DataFrame:
    bk = sonuc.belge_konu
    baskin = np.argmax(bk, axis=1)
    t = pd.DataFrame({"Belge": [belge_adlari[i] for i in sonuc.belge_indisleri],
                      "Baskın konu": [adlar[k] for k in baskin],
                      "Baskın konu ağırlığı": np.round(bk.max(axis=1), 4)})
    for k, ad in enumerate(adlar):
        t[f"{k + 1}: {ad}"] = np.round(bk[:, k], 4)
    return t


def tipik_belgeler(sonuc: KonuSonucu, konu: int, n: int = 3) -> List[int]:
    """Konunun ağırlığının en yüksek olduğu n belgenin özgün indisleri."""
    sira = np.lexsort((np.arange(sonuc.belge_konu.shape[0]), -sonuc.belge_konu[:, konu]))[:n]
    return [int(sonuc.belge_indisleri[i]) for i in sira]


def konu_grafigi(sonuc: KonuSonucu, adlar: Sequence[str], sutun_sayisi: int = 3):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = sonuc.ayarlar.konu_sayisi
    satir = int(np.ceil(k / sutun_sayisi))
    fig = make_subplots(rows=satir, cols=sutun_sayisi, subplot_titles=[f"{i + 1}. {a}" for i, a in enumerate(adlar)],
                        horizontal_spacing=0.12, vertical_spacing=max(0.02, 0.3 / satir))
    for i, s in enumerate(sonuc.konu_sozcuk):
        ust = np.argsort(-s, kind="stable")[:sonuc.ayarlar.ust_sozcuk][::-1]
        fig.add_trace(go.Bar(x=s[ust], y=list(sonuc.sozcukler[ust]), orientation="h", marker_color="#1f5f8b",
                             showlegend=False, hovertemplate="%{y}: %{x:.4f}<extra></extra>"),
                      row=i // sutun_sayisi + 1, col=i % sutun_sayisi + 1)
    fig.update_layout(height=max(320, 26 * sonuc.ayarlar.ust_sozcuk * satir + 60 * satir),
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(type="category")
    return fig


def karsilastirma_grafigi(tablo: pd.DataFrame):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sutunlar = [c for c in tablo.columns if c not in ("Konu sayısı", "Not")]
    fig = make_subplots(rows=1, cols=len(sutunlar), subplot_titles=sutunlar)
    for i, c in enumerate(sutunlar, start=1):
        fig.add_trace(go.Scatter(x=tablo["Konu sayısı"], y=tablo[c], mode="lines+markers", line=dict(color="#1f5f8b"),
                                 showlegend=False), row=1, col=i)
        fig.update_xaxes(title_text="Konu sayısı", dtick=1, row=1, col=i)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return fig
