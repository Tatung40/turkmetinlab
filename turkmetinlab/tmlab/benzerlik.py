"""Belge benzerliği: TF-IDF kosinüs benzerliği, en yakın komşular ve hiyerarşik kümeleme.

Tanımlar
--------
* Girdi, tfidf_hesapla() ile üretilen L2 normalleştirilmiş TF-IDF matrisidir. Her belge vektörünün
  uzunluğu 1 olduğu için kosinüs benzerliği iç çarpıma eşittir: cos(a, b) = a · b.
  TF-IDF değerleri negatif olmadığından benzerlik 0 (ortak terim yok) ile 1 (aynı terim dağılımı)
  arasındadır.
* Kosinüs benzerliği yalnızca TERİM DAĞILIMLARININ örtüşmesini ölçer. Anlamsal eşdeğerlik, intihal
  veya metin kökeni (birinin diğerinden türediği) tespiti DEĞİLDİR: aynı konuyu farklı sözcüklerle
  anlatan metinler düşük, aynı kalıp sözcükleri kullanan ilgisiz metinler yüksek benzerlik alabilir.
* Hiyerarşik kümeleme (scipy):
  - "average": ortalama bağlantı (UPGMA), uzaklık = 1 − kosinüs benzerliği.
  - "ward": Ward yöntemi; L2 normalleştirilmiş vektörlerde Öklid uzaklığı
    ||a − b|| = sqrt(2 − 2·cos(a, b)) olduğundan kosinüs benzerliğinden hesaplanır.
* Büyük derlemlerde tam n × n matris yerine, satırlar parça parça işlenerek her belgenin en yakın
  k komşusu bulunur (bellek: parça × n). Sonuç tam matristen hesaplananla birebir aynıdır.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Tam benzerlik matrisi (ısı haritası, kümeleme) için en fazla belge sayısı:
# 2000 belge → 2000×2000×8 bayt ≈ 32 MB.
YOGUN_MATRIS_SINIRI = 2000
ISI_HARITASI_SINIRI = 150
DENDROGRAM_SINIRI = 200
# Parçalı en yakın komşu hesabında bir parçanın en fazla kaç hücre (satır × belge) tutacağı (~64 MB)
PARCA_HUCRE = 8_000_000

UYARI_METNI = (
    "Kosinüs benzerliği, iki belgedeki **terimlerin ve TF-IDF ağırlıklarının örtüşmesini** ölçer (0: ortak terim "
    "yok, 1: aynı terim dağılımı). Anlamsal eşdeğerlik, intihal ya da bir metnin diğerinden türediği anlamına "
    "**gelmez**: aynı konuyu farklı sözcüklerle anlatan metinler düşük, aynı kalıp ifadeleri kullanan ilgisiz "
    "metinler yüksek benzerlik alabilir. Sonuçlar ön işleme ve TF-IDF ayarlarına bağlıdır."
)


def _normalize_mi(X) -> bool:
    normlar = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return bool(np.all((np.abs(normlar - 1) < 1e-6) | (normlar == 0)))


def kosinus_matrisi(X) -> np.ndarray:
    """L2 normalleştirilmiş seyrek matristen tam kosinüs benzerlik matrisi (n × n)."""
    if X.shape[0] > YOGUN_MATRIS_SINIRI:
        raise ValueError(f"Tam benzerlik matrisi en fazla {YOGUN_MATRIS_SINIRI} belge için hesaplanır.")
    if not _normalize_mi(X):
        raise ValueError("Girdi satırları L2 normalleştirilmiş olmalıdır.")
    S = np.asarray((X @ X.T).toarray(), dtype=np.float64)
    np.clip(S, 0.0, 1.0, out=S)  # kayan nokta taşmalarını (1.0000000002) düzelt
    np.fill_diagonal(S, 1.0)
    return S


def en_benzer_ciftler(S: np.ndarray, adlar: Sequence[str], n: int = 50) -> pd.DataFrame:
    """Tam matristen en yüksek benzerlikli n belge çifti (her çift bir kez)."""
    i, j = np.triu_indices(S.shape[0], k=1)
    degerler = S[i, j]
    if len(degerler) == 0:
        return pd.DataFrame(columns=["Sıra", "Belge A", "Belge B", "Kosinüs benzerliği"])
    # Eşitlikte belge sırasına göre (belirlenimci)
    sira = np.lexsort((j, i, -degerler))[:n]
    return pd.DataFrame({
        "Sıra": np.arange(1, len(sira) + 1),
        "Belge A": [adlar[k] for k in i[sira]],
        "Belge B": [adlar[k] for k in j[sira]],
        "Kosinüs benzerliği": np.round(degerler[sira], 6),
    })


def en_yakin_komsular(X, k: int = 10, parca: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Her belgenin en benzer k belgesi (kendisi hariç), parça parça ve bellek dostu.

    Döndürür: (indisler [n, k], benzerlikler [n, k]); eşitlikte küçük indis önce gelir.
    """
    n = X.shape[0]
    k = max(0, min(k, n - 1))
    if parca is None:
        parca = max(1, PARCA_HUCRE // max(n, 1))
    X = X.tocsr()
    XT = X.T.tocsc()
    indisler = np.zeros((n, k), dtype=np.int64)
    degerler = np.zeros((n, k), dtype=np.float64)
    for bas in range(0, n, parca):
        son = min(n, bas + parca)
        blok = np.asarray((X[bas:son] @ XT).toarray(), dtype=np.float64)
        np.clip(blok, 0.0, 1.0, out=blok)
        blok[np.arange(son - bas), np.arange(bas, son)] = -1.0  # kendisi hariç
        if k == 0:
            continue
        # Her satırın k'ıncı en büyük değeri (vektörel); yalnızca ona eşit/büyük adaylar sıralanır
        esik = -np.partition(-blok, k - 1, axis=1)[:, k - 1]
        for r in range(son - bas):
            satir = blok[r]
            buyuk = np.flatnonzero(satir > esik[r])
            esit = np.flatnonzero(satir == esik[r])[: k - len(buyuk)]  # eşitlikte küçük indis önce
            secilen = np.concatenate([buyuk, esit])
            sira = secilen[np.lexsort((secilen, -satir[secilen]))]
            indisler[bas + r] = sira
            degerler[bas + r] = satir[sira]
    return indisler, degerler


def komsu_tablosu(indisler: np.ndarray, degerler: np.ndarray, adlar: Sequence[str]) -> pd.DataFrame:
    satirlar = []
    for i in range(indisler.shape[0]):
        for r in range(indisler.shape[1]):
            satirlar.append((adlar[i], r + 1, adlar[indisler[i, r]], round(float(degerler[i, r]), 6)))
    return pd.DataFrame(satirlar, columns=["Belge", "Sıra", "Benzer belge", "Kosinüs benzerliği"])


def ciftler_komsulardan(indisler: np.ndarray, degerler: np.ndarray, adlar: Sequence[str], n: int = 50) -> pd.DataFrame:
    """En yakın komşu listelerinden en benzer n çift.

    Genel sıralamada ilk n'e giren bir (i, j) çiftinde j, i'nin en yakın n komşusu arasında olmak
    zorundadır; bu yüzden k ≥ n ise sonuç tam matristen hesaplananla aynıdır.
    """
    gorulen = {}
    for i in range(indisler.shape[0]):
        for r in range(indisler.shape[1]):
            j = int(indisler[i, r])
            a, b = (i, j) if i < j else (j, i)
            gorulen[(a, b)] = float(degerler[i, r])
    ogeler = sorted(gorulen.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:n]
    return pd.DataFrame({
        "Sıra": range(1, len(ogeler) + 1),
        "Belge A": [adlar[a] for (a, _), _ in ogeler],
        "Belge B": [adlar[b] for (_, b), _ in ogeler],
        "Kosinüs benzerliği": [round(v, 6) for _, v in ogeler],
    })


def belge_siralamasi(S: np.ndarray, adlar: Sequence[str], indis: int, n: int = 10) -> pd.DataFrame:
    """Seçilen belgeye en benzer n belge."""
    satir = S[indis].copy()
    satir[indis] = -1
    sira = np.lexsort((np.arange(len(satir)), -satir))
    sira = sira[sira != indis][:n]  # belgenin kendisi listeye girmez
    return pd.DataFrame({"Sıra": range(1, len(sira) + 1), "Benzer belge": [adlar[k] for k in sira],
                         "Kosinüs benzerliği": np.round(satir[sira], 6)})


def benzerlik_ozeti(S: np.ndarray) -> dict:
    ust = S[np.triu_indices(S.shape[0], k=1)]
    if len(ust) == 0:
        return {}
    return {
        "Belge sayısı": S.shape[0],
        "Belge çifti sayısı": len(ust),
        "Ortalama benzerlik": round(float(ust.mean()), 4),
        "Ortanca benzerlik": round(float(np.median(ust)), 4),
        "En yüksek benzerlik": round(float(ust.max()), 4),
        "Hiç ortak terimi olmayan çiftler (%)": round(float((ust == 0).mean() * 100), 2),
    }


# ---------------------------------------------------------------------------
# Hiyerarşik kümeleme
# ---------------------------------------------------------------------------

KUMELEME_YONTEMLERI = {
    "average": "Ortalama bağlantı (kosinüs uzaklığı) — önerilen",
    "ward": "Ward (normalleştirilmiş vektörlerde Öklid uzaklığı)",
}


def _uzaklik_vektoru(S: np.ndarray, yontem: str) -> np.ndarray:
    from scipy.spatial.distance import squareform

    D = 1.0 - S if yontem == "average" else np.sqrt(np.maximum(2.0 - 2.0 * S, 0.0))
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)
    D = (D + D.T) / 2  # sayısal simetri
    return squareform(D, checks=False)


def hiyerarsik_kumeleme(S: np.ndarray, yontem: str = "average") -> np.ndarray:
    """scipy bağlantı (linkage) matrisi döndürür."""
    from scipy.cluster.hierarchy import linkage

    if yontem not in KUMELEME_YONTEMLERI:
        raise ValueError(f"Geçersiz kümeleme yöntemi: {yontem}")
    if S.shape[0] < 3:
        raise ValueError("Kümeleme için en az 3 belge gerekir.")
    return linkage(_uzaklik_vektoru(S, yontem), method=yontem)


def kume_etiketleri(Z: np.ndarray, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster

    return fcluster(Z, t=k, criterion="maxclust")


def siluet_tablosu(S: np.ndarray, Z: np.ndarray, yontem: str, en_fazla_k: int = 10) -> pd.DataFrame:
    """Farklı küme sayıları için ortalama siluet değeri (−1…1; yüksek = daha belirgin ayrım)."""
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    D = squareform(_uzaklik_vektoru(S, yontem))
    satirlar = []
    for k in range(2, min(en_fazla_k, S.shape[0] - 1) + 1):
        etiket = kume_etiketleri(Z, k)
        if len(set(etiket)) < 2:
            continue
        satirlar.append((k, len(set(etiket)), round(float(silhouette_score(D, etiket, metric="precomputed")), 4)))
    return pd.DataFrame(satirlar, columns=["İstenen küme sayısı", "Oluşan küme sayısı", "Ortalama siluet"])


def kume_tablosu(etiketler: np.ndarray, adlar: Sequence[str]) -> pd.DataFrame:
    t = pd.DataFrame({"Belge": list(adlar), "Küme": etiketler.astype(int)})
    boyut = t["Küme"].value_counts()
    t["Kümedeki belge sayısı"] = t["Küme"].map(boyut).astype(int)
    return t.sort_values(["Küme", "Belge"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Görselleştirme
# ---------------------------------------------------------------------------

def isi_haritasi(S: np.ndarray, adlar: Sequence[str], sira: Optional[List[int]] = None):
    import plotly.graph_objects as go

    if sira is None:
        sira = list(range(S.shape[0]))
    A = S[np.ix_(sira, sira)]
    etiket = [adlar[i] for i in sira]
    fig = go.Figure(go.Heatmap(z=A, x=etiket, y=etiket, colorscale="Blues", zmin=0, zmax=1,
                               hovertemplate="%{y}<br>%{x}<br>Kosinüs: %{z:.3f}<extra></extra>",
                               colorbar=dict(title="Kosinüs")))
    boyut = max(420, min(900, 16 * len(sira) + 200))
    fig.update_layout(height=boyut, margin=dict(l=10, r=10, t=40, b=10), title="Belge × belge kosinüs benzerliği")
    fig.update_xaxes(type="category", tickangle=-45, showticklabels=len(sira) <= 60)
    fig.update_yaxes(type="category", autorange="reversed", showticklabels=len(sira) <= 60)
    return fig


def dendrogram_figuru(Z: np.ndarray, adlar: Sequence[str], yontem: str):
    """scipy dendrogram koordinatlarından Plotly şekli (yatay, yaprak etiketleri solda)."""
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import dendrogram

    d = dendrogram(Z, no_plot=True, labels=list(adlar), orientation="right")
    fig = go.Figure()
    for xs, ys in zip(d["dcoord"], d["icoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#1f5f8b", width=1.4),
                                 hoverinfo="skip", showlegend=False))
    yapraklar = d["ivl"]
    fig.update_layout(
        height=max(400, 18 * len(yapraklar) + 80), margin=dict(l=10, r=10, t=40, b=10),
        title="Dendrogram", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Uzaklık (1 − kosinüs)" if yontem == "average" else "Ward birleşme uzaklığı",
                   gridcolor="rgba(128,128,128,0.25)"),
        yaxis=dict(tickvals=[5 + 10 * i for i in range(len(yapraklar))], ticktext=yapraklar, showgrid=False),
    )
    return fig, d["leaves"]
