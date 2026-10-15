"""Grup karşılaştırması: gruplara göre sıklıklar ve ayırt edici sözcükler.

Ayırt edici sözcükler — Dirichlet önselli log-odds oranı (Monroe, Colaresi & Quinn, 2008,
"Fightin' Words", Political Analysis 16(4), denklem 15–22)
------------------------------------------------------------------------------------------
Her grup i, geri kalan TÜM gruplar (j) ile karşılaştırılır:

    δ_w = log[(y_wi + α_w) / (n_i + α_0 − y_wi − α_w)] − log[(y_wj + α_w) / (n_j + α_0 − y_wj − α_w)]
    σ²_w ≈ 1/(y_wi + α_w) + 1/(y_wj + α_w)          z_w = δ_w / σ_w

y_wi: w sözcüğünün i grubundaki sıklığı; n_i: i grubundaki toplam sözcük; α_0 = Σ α_w.
Önsel (α_w):
* "bilgilendirici" (varsayılan): α_w = w'nin tüm derlemdeki sıklığı (deneysel Bayes; seyrek
  sözcüklerin aşırı uç değerler almasını önler; jmhessel/FightingWords ve tidylo bu yaklaşımı kullanır).
* "zayif": α_w = 0,01 (neredeyse yumuşatmasız; FightingWords varsayılanı).
z > 0: sözcük i grubunda diğer gruplara göre daha sık; z < 0: daha seyrek.

ÖNEMLİ — z değerlerinin yorumu
-----------------------------
z formülü sözcükleri birbirinden bağımsız sayar. Gerçekte aynı belgedeki sözcükler bağımlıdır
(bir belge bir sözcüğü çok kez kullanabilir), bu yüzden |z| > 1,96 'anlamlı' sayılırsa yanlış
pozitifler çoğalır; ayrıca binlerce sözcük aynı anda sınanır. Bu modül bu nedenle BELGE DÜZEYİNDE
PERMÜTASYON uygular: grup etiketleri belgeler arasında (grup büyüklükleri korunarak) rastgele
karıştırılır, her karıştırmada her grubun en büyük z değeri kaydedilir; bu dağılımın %95'lik
dilimi, o veri için "tesadüfen aşılmayan" eşik olarak raporlanır (aile bazında hata kontrolü).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .preprocess import IslenmisBelge

ONSELLER = {
    "bilgilendirici": "Bilgilendirici önsel (tüm derlemin sıklıkları) — önerilen",
    "zayif": "Zayıf önsel (α = 0,01)",
}
EN_FAZLA_GRUP = 20
KUCUK_GRUP_BELGE = 5
KUCUK_GRUP_SOZCUK = 1000
DENGESIZLIK_ORANI = 5.0

UYARI_METNI = (
    "Ayırt edici sözcükler, bir grupta diğer gruplara göre **orantılı olarak daha sık** kullanılan sözcüklerdir. "
    "Grupların neden farklı olduğunu açıklamaz; farklar grup büyüklüğü, belge uzunluğu ve konu dağılımından da "
    "kaynaklanabilir. z değerleri sözcükleri bağımsız varsaydığı için anlamlılığı abartır; bu yüzden tabloda "
    "belge düzeyinde **permütasyonla** belirlenen, bu veriye özgü eşik de gösterilir."
)


class GrupHatasi(ValueError):
    pass


@dataclass(frozen=True)
class GrupAyarlari:
    onsel: str = "bilgilendirici"
    min_siklik: int = 5
    permutasyon: int = 50
    tohum: int = 42

    def __post_init__(self):
        if self.onsel not in ONSELLER:
            raise ValueError(f"Geçersiz önsel: {self.onsel}")
        if self.min_siklik < 1:
            raise ValueError("En az sıklık 1 veya daha büyük olmalıdır.")
        if not 0 <= self.permutasyon <= 1000:
            raise ValueError("Permütasyon sayısı 0 ile 1000 arasında olmalıdır.")

    def sozluk(self) -> dict:
        return {"önsel": ONSELLER[self.onsel], "sözcük en az toplam sıklık": self.min_siklik,
                "permütasyon sayısı": self.permutasyon, "permütasyon tohumu": self.tohum}


@dataclass
class GrupVerisi:
    gruplar: List[str]            # grup adları (sıralı)
    etiket: np.ndarray            # her belgenin grup indisi
    X: sp.csr_matrix              # belge × sözcük sayım matrisi
    sozcukler: np.ndarray
    belge_indisleri: np.ndarray   # analize giren belgelerin özgün indisleri


def grup_verisi_hazirla(belgeler: Sequence[IslenmisBelge], grup_degerleri: Sequence) -> tuple:
    """Boş grup değerli ve sözcüksüz belgeleri ayıklar. (GrupVerisi, dışarıda kalan özgün indisler)."""
    from sklearn.feature_extraction.text import CountVectorizer

    from .analysis import turkce_sirala_anahtari

    bos_grup, gecerli = [], []
    for i, (b, g) in enumerate(zip(belgeler, grup_degerleri)):
        if g is None or (isinstance(g, float) and g != g) or not str(g).strip():
            bos_grup.append(i)
        elif b.sozcukler:
            gecerli.append(i)
    adlar = sorted({str(grup_degerleri[i]).strip() for i in gecerli}, key=turkce_sirala_anahtari)
    if len(adlar) < 2:
        raise GrupHatasi("Karşılaştırma için metni olan en az 2 grup gerekir.")
    if len(adlar) > EN_FAZLA_GRUP:
        raise GrupHatasi(f"Seçilen sütunda {len(adlar)} farklı değer var; en fazla {EN_FAZLA_GRUP} grup "
                         "karşılaştırılabilir. Kategorik bir sütun seçtiğinizden emin olun (ör. bölüm, yıl, cinsiyet).")
    indeks = {g: k for k, g in enumerate(adlar)}
    etiket = np.array([indeks[str(grup_degerleri[i]).strip()] for i in gecerli])
    v = CountVectorizer(analyzer=lambda x: x, lowercase=False)
    X = v.fit_transform([belgeler[i].sozcukler for i in gecerli]).tocsr()
    return GrupVerisi(adlar, etiket, X, np.asarray(v.get_feature_names_out(), dtype=object),
                      np.array(gecerli)), bos_grup


def _grup_sayimlari(X: sp.csr_matrix, etiket: np.ndarray, g: int) -> np.ndarray:
    G = sp.csr_matrix((np.ones(len(etiket)), (etiket, np.arange(len(etiket)))), shape=(g, len(etiket)))
    return np.asarray((G @ X).toarray(), dtype=np.float64)  # g × V


def _z_degerleri(Y: np.ndarray, onsel: str) -> tuple:
    """Y: g × V grup sayımları. Her grup için 'grup ve geri kalanlar' δ ve z (g × V)."""
    toplam_w = Y.sum(axis=0)
    alfa = toplam_w.copy() if onsel == "bilgilendirici" else np.full(Y.shape[1], 0.01)
    a0 = alfa.sum()
    n = Y.sum(axis=1, keepdims=True)
    y_i = Y
    y_j = toplam_w[None, :] - Y
    n_j = n.sum() - n
    delta = (np.log((y_i + alfa) / (n + a0 - y_i - alfa)) - np.log((y_j + alfa) / (n_j + a0 - y_j - alfa)))
    sigma2 = 1.0 / (y_i + alfa) + 1.0 / (y_j + alfa)
    return delta, delta / np.sqrt(sigma2)


def grup_sutunu_oner(tablo: pd.DataFrame, haric=()) -> Optional[str]:
    """Kategori gibi davranan sütunlardan en az farklı değerlisini önerir (2 ≤ değer ≤ satır/2 ve ≤ 20)."""
    adaylar = []
    for c in tablo.columns:
        if c in haric:
            continue
        n = tablo[c].dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique()
        if 2 <= n <= min(EN_FAZLA_GRUP, max(2, len(tablo) // 2)):
            adaylar.append((n, list(tablo.columns).index(c), c))
    return min(adaylar)[2] if adaylar else None


def grup_ozeti(veri: GrupVerisi) -> pd.DataFrame:
    Y = _grup_sayimlari(veri.X, veri.etiket, len(veri.gruplar))
    uzunluk = np.asarray(veri.X.sum(axis=1)).ravel()
    satirlar = []
    for k, ad in enumerate(veri.gruplar):
        m = veri.etiket == k
        satirlar.append({"Grup": ad, "Belge": int(m.sum()), "Sözcük": int(Y[k].sum()),
                         "Farklı sözcük": int((Y[k] > 0).sum()),
                         "Ortalama belge uzunluğu": round(float(uzunluk[m].mean()), 1),
                         "Derlemdeki sözcük payı (%)": round(float(Y[k].sum() / Y.sum() * 100), 2)})
    return pd.DataFrame(satirlar)


def yontemsel_uyarilar(ozet: pd.DataFrame) -> List[str]:
    u = []
    if (ozet["Belge"] == 1).all() and len(ozet) > 2:
        u.append("Her grupta yalnızca bir belge var; seçilen sütun bir kimlik ya da başlık sütunu olabilir. Bu "
                 "durumda karşılaştırma grupları değil tek tek belgeleri yansıtır. Amacınız bu değilse belgeleri "
                 "gruplayan kategorik bir sütun seçin (ör. bölüm, yıl, cinsiyet).")
    kucuk = ozet[(ozet["Belge"] < KUCUK_GRUP_BELGE) | (ozet["Sözcük"] < KUCUK_GRUP_SOZCUK)]
    if len(kucuk):
        ornek = ", ".join(f"{r['Grup']} ({r['Belge']} belge, {r['Sözcük']} sözcük)" for _, r in kucuk.head(8).iterrows())
        u.append(f"{len(kucuk)} grup küçük (en az {KUCUK_GRUP_BELGE} belge ve {KUCUK_GRUP_SOZCUK} sözcük önerilir): "
                 f"{ornek}{' …' if len(kucuk) > 8 else ''}. Bu grupların ayırt edici sözcükleri birkaç belgenin "
                 "özelliklerini yansıtabilir; genelleme yapmayın.")
    oran = ozet["Sözcük"].max() / max(ozet["Sözcük"].min(), 1)
    if oran > DENGESIZLIK_ORANI:
        u.append(f"Gruplar dengesiz: en büyük grup en küçüğün {oran:.1f} katı sözcük içeriyor. Ham sıklıkları "
                 "karşılaştırmayın; 'bin sözcükte' değerlerini ve z değerlerini kullanın. Küçük grupta z değerleri "
                 "daha düşük çıkar (daha az kanıt).".replace(".", ",", 1))
    if len(ozet) > 8:
        u.append(f"{len(ozet)} grup karşılaştırılıyor. Her grup tüm diğerlerinin birleşimiyle karşılaştırılır; "
                 "çok sayıda grupta sonuçları dikkatle yorumlayın.")
    return u


def permutasyon_esikleri(veri: GrupVerisi, ayar: GrupAyarlari, maske: np.ndarray) -> Dict:
    """Belge düzeyinde permütasyon: her grup için rastgele gruplamada en büyük z'nin %95'lik dilimi
    ve bir sözcüğün |z| > 1,96 olma oranı (yanlış pozitif oranı)."""
    g = len(veri.gruplar)
    rng = np.random.default_rng(ayar.tohum)
    en_buyuk = np.zeros((ayar.permutasyon, g))
    yanlis = np.zeros(ayar.permutasyon)
    for t in range(ayar.permutasyon):
        Y = _grup_sayimlari(veri.X, rng.permutation(veri.etiket), g)[:, maske]
        _, z = _z_degerleri(Y, ayar.onsel)
        en_buyuk[t] = z.max(axis=1)
        yanlis[t] = float((np.abs(z) > 1.96).mean())
    return {"esik": np.percentile(en_buyuk, 95, axis=0), "yanlis_pozitif_orani": float(yanlis.mean()),
            "tekrar": ayar.permutasyon}


def ayirt_edici_sozcukler(veri: GrupVerisi, ayar: GrupAyarlari) -> tuple:
    """Tüm gruplar için uzun biçimli tablo ve permütasyon özeti döndürür."""
    from .analysis import turkce_sirala_anahtari

    g = len(veri.gruplar)
    Y_tam = _grup_sayimlari(veri.X, veri.etiket, g)
    maske = Y_tam.sum(axis=0) >= ayar.min_siklik
    if not maske.any():
        raise GrupHatasi(f"En az {ayar.min_siklik} kez geçen sözcük yok; 'en az sıklık' değerini düşürün.")
    Y = Y_tam[:, maske]
    sozc = veri.sozcukler[maske]
    delta, z = _z_degerleri(Y, ayar.onsel)
    n = Y.sum(axis=1)
    toplam = Y.sum(axis=0)
    # Belge oranı ve tek belge baskınlığı
    Xm = veri.X[:, np.flatnonzero(maske)].tocsc()
    ikili = Xm.copy()
    ikili.data = np.ones_like(ikili.data)
    perm = permutasyon_esikleri(veri, ayar, maske) if ayar.permutasyon > 0 else None
    alfabe = np.empty(len(sozc), dtype=np.int64)
    alfabe[np.argsort(np.array([turkce_sirala_anahtari(w) for w in sozc], dtype=object), kind="stable")] = \
        np.arange(len(sozc))
    parcalar = []
    for k, ad in enumerate(veri.gruplar):
        m = veri.etiket == k
        belge_orani = np.asarray(ikili[m].sum(axis=0)).ravel() / max(m.sum(), 1) * 100
        tek_belge = np.asarray(Xm[m].max(axis=0).toarray()).ravel()
        pay = np.divide(tek_belge, Y[k], out=np.zeros_like(tek_belge, dtype=float), where=Y[k] > 0) * 100
        sira = np.lexsort((alfabe, -np.round(z[k], 12)))
        t = pd.DataFrame({
            "Grup": ad,
            "Sıra": np.arange(1, len(sira) + 1),
            "Sözcük": sozc[sira],
            "z (ağırlıklı log-odds)": np.round(z[k, sira], 4),
            "Log-odds farkı (δ)": np.round(delta[k, sira], 4),
            "Gruptaki sıklık": Y[k, sira].astype(int),
            "Diğer gruplardaki sıklık": (toplam - Y[k])[sira].astype(int),
            "Gruptaki bin sözcükte": np.round(Y[k, sira] / max(n[k], 1) * 1000, 4),
            "Diğerlerinde bin sözcükte": np.round((toplam - Y[k])[sira] / max(n.sum() - n[k], 1) * 1000, 4),
            "Gruptaki belge oranı (%)": np.round(belge_orani[sira], 2),
            "Tek belgeden gelen en büyük pay (%)": np.round(pay[sira], 2),
        })
        if perm is not None:
            t["Permütasyon eşiğini aşıyor"] = np.where(z[k, sira] > perm["esik"][k], "evet", "hayır")
        parcalar.append(t)
    return pd.concat(parcalar, ignore_index=True), perm


def siklik_karsilastirma(veri: GrupVerisi, min_siklik: int = 1) -> pd.DataFrame:
    """Geniş tablo: sözcük, toplam, her grup için ham sıklık ve bin sözcükte."""
    from .analysis import turkce_sirala_anahtari

    Y = _grup_sayimlari(veri.X, veri.etiket, len(veri.gruplar))
    toplam = Y.sum(axis=0)
    maske = toplam >= min_siklik
    t = pd.DataFrame({"Sözcük": veri.sozcukler[maske], "Toplam": toplam[maske].astype(int)})
    n = Y.sum(axis=1)
    for k, ad in enumerate(veri.gruplar):
        t[f"{ad} · sıklık"] = Y[k, maske].astype(int)
        t[f"{ad} · bin sözcükte"] = np.round(Y[k, maske] / max(n[k], 1) * 1000, 4)
    t["_k"] = t["Sözcük"].map(turkce_sirala_anahtari)
    t = t.sort_values(["Toplam", "_k"], ascending=[False, True], kind="stable").drop(columns="_k")
    return t.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------

def ayirt_edici_grafik(tablo: pd.DataFrame, gruplar: Sequence[str], ilk_n: int = 15,
                       esikler: Optional[np.ndarray] = None, sutun_sayisi: int = 2):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    satir = int(np.ceil(len(gruplar) / sutun_sayisi))
    fig = make_subplots(rows=satir, cols=sutun_sayisi, subplot_titles=list(gruplar), horizontal_spacing=0.15,
                        vertical_spacing=max(0.03, 0.25 / satir))
    for k, ad in enumerate(gruplar):
        t = tablo[tablo["Grup"] == ad].head(ilk_n).iloc[::-1]
        renk = ["#1f5f8b" if (esikler is None or z > esikler[k]) else "#9bb8cc" for z in t["z (ağırlıklı log-odds)"]]
        fig.add_trace(go.Bar(x=t["z (ağırlıklı log-odds)"], y=t["Sözcük"], orientation="h", marker_color=renk,
                             showlegend=False, customdata=t[["Gruptaki sıklık", "Gruptaki belge oranı (%)"]],
                             hovertemplate="%{y}<br>z: %{x:.2f}<br>Sıklık: %{customdata[0]}<br>"
                                           "Belge oranı: %%{customdata[1]}<extra></extra>"),
                      row=k // sutun_sayisi + 1, col=k % sutun_sayisi + 1)
        if esikler is not None:
            fig.add_vline(x=float(esikler[k]), line_dash="dot", line_color="#c0392b", row=k // sutun_sayisi + 1,
                          col=k % sutun_sayisi + 1)
    fig.update_yaxes(type="category")
    fig.update_layout(height=max(360, 22 * ilk_n * satir + 70 * satir), margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def iki_grup_sacilim(tablo: pd.DataFrame, gruplar: Sequence[str], etiket_sayisi: int = 15):
    """İki grup için Monroe vd. tarzı grafik: x = log10(toplam sıklık), y = z (1. grup lehine)."""
    import plotly.graph_objects as go

    t = tablo[tablo["Grup"] == gruplar[0]].copy()
    t["Toplam"] = t["Gruptaki sıklık"] + t["Diğer gruplardaki sıklık"]
    z = t["z (ağırlıklı log-odds)"]
    etiketli = set(t.nlargest(etiket_sayisi, "z (ağırlıklı log-odds)")["Sözcük"]) | \
        set(t.nsmallest(etiket_sayisi, "z (ağırlıklı log-odds)")["Sözcük"])
    renk = np.where(z > 1.96, "#1f5f8b", np.where(z < -1.96, "#c0392b", "#b0b0b0"))
    fig = go.Figure(go.Scatter(
        x=np.log10(t["Toplam"]), y=z, mode="markers+text", marker=dict(color=renk, size=6, opacity=0.7),
        text=[w if w in etiketli else "" for w in t["Sözcük"]], textposition="top center",
        hovertext=[f"{w}<br>z: {zz:.2f}<br>Toplam: {n}" for w, zz, n in zip(t["Sözcük"], z, t["Toplam"])],
        hoverinfo="text"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      title=f"↑ {gruplar[0]} lehine · ↓ {gruplar[1]} lehine",
                      xaxis_title="Toplam sıklık (log10)", yaxis_title="z (ağırlıklı log-odds)")
    fig.add_hline(y=0, line_color="#888")
    return fig
