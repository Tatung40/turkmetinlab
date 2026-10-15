"""Sözcük birliktelik (eş-dizimsel olmayan, bağlamsal birlikte görülme) ağları.

Bağlam türleri
--------------
* "pencere": iki sözcük, metindeki konumları arasındaki uzaklık en fazla k ise birlikte görülmüş
  sayılır (±k sözcük). Uzaklık ÖZGÜN metindeki konumlara göre ölçülür; ön işlemede çıkarılan
  sözcükler (durak sözcük, sayı) de yer kaplar. Böylece "eğitim ve öğretim" ifadesinde eğitim ile
  öğretim arasındaki uzaklık 2'dir. "Cümle sınırlarına uy" seçiliyse pencere bölüt (cümle/ifade,
  bkz. bölütleme ayarı) sınırını aşmaz; seçili değilse belge içinde aşabilir, belgeler arasında asla.
  Ham değer = birbirine en fazla k uzaklıkta bulunan (a, b) konum çiftlerinin sayısı.
* "bolut": iki sözcüğün birlikte geçtiği bölüt (cümle) sayısı.
* "belge": iki sözcüğün birlikte geçtiği belge sayısı.
Aynı sözcüğün kendisiyle birlikteliği sayılmaz.

Ölçütler
--------
* Ham birlikte görülme: yukarıdaki sayım.
* PMI = log2( P(a,b) / (P(a)·P(b)) ), PPMI = max(PMI, 0).
  - pencere: P(a,b) = C(a,b)/T, P(a) = C(a)/T; C(a) = Σ_b C(a,b), T = Σ_a C(a)
    (sözcük–bağlam matrisi tanımı; Church & Hanks, 1990; Levy vd., 2015).
  - bolut / belge: P(a,b) = U(a,b)/U, P(a) = U(a)/U; U = en az bir aday sözcük içeren birim sayısı.
  PMI düşük sıklıklarda kararsızdır ve seyrek sözcükleri abartır; bu yüzden en az birlikte görülme
  eşiği uygulanır.
* Jaccard = U(a,b) / (U(a) + U(b) − U(a,b)); U, birlikteliğe izin verilen en büyük birimdir:
  bölüt (pencere + cümle sınırı ya da "bolut" bağlamı) veya belge.

Birlikte görülme nedensellik ya da doğrudan anlamsal ilişki DEĞİLDİR; yalnızca iki sözcüğün
seçilen bağlamda birlikte kullanıldığını gösterir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .preprocess import IslenmisBelge

BAGLAM_TURLERI = {
    "pencere": "Sözcük penceresi (±k sözcük)",
    "bolut": "Aynı cümle / bölüt",
    "belge": "Aynı belge",
}
# Varsayılan Jaccard: DergiPark özetlerinde yazarların çok sözcüklü anahtar terimlerini ayırt etmede
# dört bağlam türünün dördünde de en yüksek AUC'yi verdi (docs/bilimsel_degerlendirme_2.0.md).
OLCUTLER = {
    "jaccard": "Jaccard benzerliği — önerilen",
    "ppmi": "PPMI (pozitif noktasal karşılıklı bilgi)",
    "ham": "Ham birlikte görülme sıklığı",
}
EN_FAZLA_ADAY = 2000
EN_FAZLA_DUGUM = 150
EN_FAZLA_KENAR = 500

UYARI_METNI = (
    "Birlikte görülme, iki sözcüğün seçilen bağlamda **birlikte kullanıldığını** gösterir; aralarında nedensellik "
    "ya da doğrudan anlamsal ilişki olduğu anlamına **gelmez**. Ağın görünümü bağlam türüne, pencere genişliğine, "
    "eşiklere ve ön işleme ayarlarına bağlıdır. Merkezîlik değerleri yalnızca gösterilen (süzülmüş) ağ için geçerlidir."
)


@dataclass(frozen=True)
class AgAyarlari:
    baglam: str = "pencere"
    pencere: int = 5
    bolut_siniri: bool = True
    aday_sozcuk: int = 1000
    min_birlikte: int = 3

    def __post_init__(self):
        if self.baglam not in BAGLAM_TURLERI:
            raise ValueError(f"Geçersiz bağlam türü: {self.baglam}")
        if not 1 <= self.pencere <= 50:
            raise ValueError("Pencere 1 ile 50 arasında olmalıdır.")
        if not 2 <= self.aday_sozcuk <= EN_FAZLA_ADAY:
            raise ValueError(f"Aday sözcük sayısı 2 ile {EN_FAZLA_ADAY} arasında olmalıdır.")
        if self.min_birlikte < 1:
            raise ValueError("En az birlikte görülme 1 veya daha büyük olmalıdır.")

    def sozluk(self) -> dict:
        return {"bağlam": BAGLAM_TURLERI[self.baglam], "pencere (± sözcük)": self.pencere,
                "pencere cümle sınırını aşmaz": self.bolut_siniri, "aday sözcük sayısı": self.aday_sozcuk,
                "en az birlikte görülme": self.min_birlikte}


@dataclass
class BirliktelikSonucu:
    sozcukler: List[str]
    siklik: np.ndarray       # her aday sözcüğün derlemdeki sıklığı
    C: np.ndarray            # simetrik ham birlikte görülme matrisi (köşegen 0)
    birim_sayisi: np.ndarray  # Jaccard birimi: sözcüğü içeren birim sayısı
    birim_ortak: np.ndarray   # Jaccard birimi: iki sözcüğü birlikte içeren birim sayısı (simetrik)
    birim_toplam: int
    jaccard_birimi: str
    ayarlar: AgAyarlari


def _aday_sozcukler(belgeler: Sequence[IslenmisBelge], n: int):
    from collections import Counter

    from .analysis import turkce_sirala_anahtari

    sayac = Counter(t for b in belgeler for t in b.sozcukler)
    sirali = sorted(sayac.items(), key=lambda kv: (-kv[1], turkce_sirala_anahtari(kv[0])))[:n]
    return [w for w, _ in sirali], np.array([f for _, f in sirali], dtype=np.int64)


def _birim_matrisi(belgeler, indeks: Dict[str, int], birim: str) -> sp.csr_matrix:
    """Birim (bölüt ya da belge) × sözcük ikili matrisi; boş birimler dahil edilmez."""
    satir, sutun, r = [], [], 0
    for b in belgeler:
        parcalar = b.bolutler if birim == "bolut" else [[t for s in b.bolutler for t in s]]
        for parca in parcalar:
            ids = {indeks[t] for t in parca if t is not None and t in indeks}
            if ids:
                satir.extend([r] * len(ids))
                sutun.extend(ids)
                r += 1
    V = len(indeks)
    B = sp.csr_matrix((np.ones(len(satir), dtype=np.float64), (satir, sutun)), shape=(r, V))
    return B


def _pencere_sayimi(belgeler, indeks: Dict[str, int], pencere: int, bolut_siniri: bool) -> np.ndarray:
    """Konum tabanlı pencere sayımı, NumPy ile vektörel.

    Tüm birimler (bölüt ya da belge) arka arkaya dizilir; aralarına 'pencere' kadar −1 dolgu konur,
    böylece hiçbir pencere birim sınırını aşamaz. −1: aday olmayan sözcük, çıkarılmış sözcük ya da dolgu.
    """
    V = len(indeks)
    dizi = []
    dolgu = [-1] * pencere
    for b in belgeler:
        if bolut_siniri:
            for s in b.bolutler:
                dizi.extend(indeks.get(t, -1) if t is not None else -1 for t in s)
                dizi.extend(dolgu)
        else:
            for s in b.bolutler:
                dizi.extend(indeks.get(t, -1) if t is not None else -1 for t in s)
            dizi.extend(dolgu)
    A = np.asarray(dizi, dtype=np.int64)
    sayim = np.zeros(V * V, dtype=np.int64)
    for d in range(1, pencere + 1):
        a, c = A[:-d], A[d:]
        m = (a >= 0) & (c >= 0) & (a != c)
        if not m.any():
            continue
        i, j = np.minimum(a[m], c[m]), np.maximum(a[m], c[m])
        sayim += np.bincount(i * V + j, minlength=V * V)
    U = sayim.reshape(V, V)
    return U + U.T  # simetrik


def birliktelik_hesapla(belgeler: Sequence[IslenmisBelge], ayarlar: AgAyarlari = AgAyarlari()) -> BirliktelikSonucu:
    sozcukler, siklik = _aday_sozcukler(belgeler, ayarlar.aday_sozcuk)
    indeks = {w: i for i, w in enumerate(sozcukler)}
    if ayarlar.baglam == "pencere":
        C = _pencere_sayimi(belgeler, indeks, ayarlar.pencere, ayarlar.bolut_siniri)
        jaccard_birimi = "bolut" if ayarlar.bolut_siniri else "belge"
    else:
        C = None
        jaccard_birimi = ayarlar.baglam
    B = _birim_matrisi(belgeler, indeks, jaccard_birimi)
    ortak = np.asarray((B.T @ B).toarray(), dtype=np.int64)
    birim_sayisi = np.diag(ortak).copy()
    np.fill_diagonal(ortak, 0)
    if C is None:
        C = ortak.copy()
    return BirliktelikSonucu(sozcukler, siklik, C, birim_sayisi, ortak, int(B.shape[0]), jaccard_birimi, ayarlar)


def _pmi(sonuc: BirliktelikSonucu, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    if sonuc.ayarlar.baglam == "pencere":
        satir = sonuc.C.sum(axis=1).astype(np.float64)
        T = satir.sum()
        pay = sonuc.C[i, j] * T
        payda = satir[i] * satir[j]
    else:
        U = float(sonuc.birim_toplam)
        pay = sonuc.birim_ortak[i, j] * U
        payda = sonuc.birim_sayisi[i].astype(np.float64) * sonuc.birim_sayisi[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((pay > 0) & (payda > 0), np.log2(pay / np.where(payda > 0, payda, 1)), -np.inf)


KENAR_DISA_AKTARIM_SINIRI = 100_000
OLCUT_SUTUNU = {"ppmi": "PPMI", "jaccard": "Jaccard", "ham": "Birlikte görülme"}
_KENAR_SUTUNLARI = ["Sıra", "Kaynak", "Hedef", "Birlikte görülme", "PPMI", "PMI", "Jaccard", "Kaynak sıklığı",
                    "Hedef sıklığı"]


def _alfabe_sirasi(sozcukler: Sequence[str]) -> np.ndarray:
    from .analysis import turkce_sirala_anahtari

    sira = np.empty(len(sozcukler), dtype=np.int64)
    sira[np.argsort(np.array([turkce_sirala_anahtari(w) for w in sozcukler], dtype=object), kind="stable")] = \
        np.arange(len(sozcukler))
    return sira


def kenar_tablosu(sonuc: BirliktelikSonucu, min_birlikte: int = None, olcut: str = "jaccard",
                  en_fazla: int = None, havuz: Sequence[str] = None, sifiri_at: bool = False) -> pd.DataFrame:
    """Eşiği geçen sözcük çiftleri (her çift bir kez), seçilen ölçüte göre sıralı.

    Sıralama: ölçüt (azalan) → ham birlikte görülme (azalan) → Türk alfabesi. Sıralama ve kesme
    DataFrame kurulmadan NumPy ile yapılır; büyük derlemlerde bellek dostudur.
    havuz: yalnızca bu sözcükler arasındaki kenarlar. sifiri_at: ölçütü 0 olan kenarları çıkar.
    """
    esik = sonuc.ayarlar.min_birlikte if min_birlikte is None else min_birlikte
    maske = np.triu(sonuc.C >= esik, k=1)
    if havuz is not None:
        indeks = {w: k for k, w in enumerate(sonuc.sozcukler)}
        secili = np.zeros(len(sonuc.sozcukler), dtype=bool)
        secili[[indeks[w] for w in havuz if w in indeks]] = True
        maske &= secili[:, None] & secili[None, :]
    i, j = np.nonzero(maske)
    if len(i) == 0:
        return pd.DataFrame(columns=_KENAR_SUTUNLARI)
    ham = sonuc.C[i, j].astype(np.int64)
    ortak = sonuc.birim_ortak[i, j].astype(np.float64)
    payda = sonuc.birim_sayisi[i] + sonuc.birim_sayisi[j] - ortak
    jac = np.where(payda > 0, ortak / np.where(payda > 0, payda, 1), 0.0)
    pmi = _pmi(sonuc, i, j)
    ppmi = np.maximum(pmi, 0.0)
    deger = {"ppmi": ppmi, "jaccard": jac, "ham": ham.astype(np.float64)}[olcut]
    if sifiri_at:
        k = deger > 0
        i, j, ham, jac, pmi, ppmi, deger = i[k], j[k], ham[k], jac[k], pmi[k], ppmi[k], deger[k]
    alfabe = _alfabe_sirasi(sonuc.sozcukler)
    a_i, a_j = alfabe[i], alfabe[j]
    sira = np.lexsort((a_j, a_i, -ham, -np.round(deger, 12)))
    if en_fazla is not None:
        sira = sira[:en_fazla]
    i, j = i[sira], j[sira]
    return pd.DataFrame({
        "Sıra": np.arange(1, len(sira) + 1),
        "Kaynak": [sonuc.sozcukler[k] for k in i],
        "Hedef": [sonuc.sozcukler[k] for k in j],
        "Birlikte görülme": ham[sira].astype(int),
        "PPMI": np.round(ppmi[sira], 6),
        "PMI": np.round(np.where(np.isfinite(pmi[sira]), pmi[sira], np.nan), 6),
        "Jaccard": np.round(jac[sira], 6),
        "Kaynak sıklığı": sonuc.siklik[i].astype(int),
        "Hedef sıklığı": sonuc.siklik[j].astype(int),
    })


def ag_sec(sonuc: BirliktelikSonucu, olcut: str = "jaccard", en_fazla_dugum: int = 50,
           en_fazla_kenar: int = 150) -> pd.DataFrame:
    """Gösterilecek ağın kenarları.

    Düğüm havuzu: derlemdeki en sık 'en_fazla_dugum' aday sözcük. Havuzdaki kenarlar seçilen ölçüte
    göre sıralanır ve ilk 'en_fazla_kenar' alınır. Ölçütü 0 olan kenarlar (ör. PPMI = 0) ağa alınmaz.
    """
    en_fazla_dugum = min(en_fazla_dugum, EN_FAZLA_DUGUM)
    en_fazla_kenar = min(en_fazla_kenar, EN_FAZLA_KENAR)
    t = kenar_tablosu(sonuc, olcut=olcut, en_fazla=en_fazla_kenar, havuz=sonuc.sozcukler[:en_fazla_dugum],
                      sifiri_at=True)
    t["Ağırlık"] = t[OLCUT_SUTUNU[olcut]] if len(t) else []
    return t


def dugum_tablosu(secili: pd.DataFrame, sonuc: BirliktelikSonucu) -> pd.DataFrame:
    """Gösterilen ağdaki düğümler: sıklık, derece, normalleştirilmiş derece, ağırlıklı derece."""
    from .analysis import turkce_sirala_anahtari

    if secili.empty:
        return pd.DataFrame(columns=["Sözcük", "Sıklık", "Derece", "Normalleştirilmiş derece", "Ağırlıklı derece",
                                     "Toplam birlikte görülme"])
    uzun = pd.concat([secili[["Kaynak", "Ağırlık", "Birlikte görülme"]].rename(columns={"Kaynak": "Sözcük"}),
                      secili[["Hedef", "Ağırlık", "Birlikte görülme"]].rename(columns={"Hedef": "Sözcük"})])
    g = uzun.groupby("Sözcük").agg(Derece=("Ağırlık", "size"), Agirlikli=("Ağırlık", "sum"),
                                     Toplam=("Birlikte görülme", "sum")).reset_index()
    n = len(g)
    siklik = dict(zip(sonuc.sozcukler, sonuc.siklik))
    t = pd.DataFrame({
        "Sözcük": g["Sözcük"],
        "Sıklık": g["Sözcük"].map(siklik).astype(int),
        "Derece": g["Derece"].astype(int),
        "Normalleştirilmiş derece": np.round(g["Derece"] / max(n - 1, 1), 6),
        "Ağırlıklı derece": np.round(g["Agirlikli"], 6),
        "Toplam birlikte görülme": g["Toplam"].astype(int),
    })
    t["_k"] = t["Sözcük"].map(turkce_sirala_anahtari)
    t = t.sort_values(["Ağırlıklı derece", "Derece", "_k"], ascending=[False, False, True], kind="stable")
    return t.drop(columns="_k").reset_index(drop=True)


def _yerlesim(G) -> dict:
    """Belirlenimci yerleşim: her bağlı bileşen ayrı yerleştirilir (Kamada–Kawai; büyük bileşende
    sabit tohumlu yay yerleşimi), bileşenler büyükten küçüğe bir ızgaraya dizilir. Böylece kopuk
    parçalar uzağa savrulmaz ve düğümler üst üste binmez."""
    import networkx as nx

    bilesenler = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: (-len(c), c[0]))
    if not bilesenler:
        return {}
    sutun = int(np.ceil(np.sqrt(len(bilesenler))))
    konum = {}
    for k, dugumler in enumerate(bilesenler):
        H = G.subgraph(dugumler)
        if len(H) == 1:
            yerel = {dugumler[0]: np.zeros(2)}
        elif len(H) <= 60:
            yerel = nx.kamada_kawai_layout(H, weight=None)
        else:
            yerel = nx.spring_layout(H, seed=42, weight=None, k=2.0 / np.sqrt(len(H)), iterations=300)
        olcek = 1.0 if k == 0 else 0.45 * np.sqrt(len(H) / max(len(bilesenler[0]), 1)) + 0.12
        merkez = np.array([k % sutun, -(k // sutun)], dtype=float) * 2.4
        for d, xy in yerel.items():
            konum[d] = merkez + np.asarray(xy) * olcek
    return konum


def ag_figuru(secili: pd.DataFrame, dugumler: pd.DataFrame, olcut: str):
    """Plotly ağ çizimi; yerleşim networkx spring_layout (sabit tohum) ile belirlenimcidir."""
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.Graph()
    G.add_nodes_from(dugumler["Sözcük"])
    enb = float(secili["Ağırlık"].max()) if len(secili) else 1.0
    for a, b, w in zip(secili["Kaynak"], secili["Hedef"], secili["Ağırlık"]):
        G.add_edge(a, b, weight=float(w) / enb if enb > 0 else 1.0)
    konum = _yerlesim(G)
    fig = go.Figure()
    for (a, b, w), ham in zip(G.edges(data="weight"), secili["Birlikte görülme"]):
        x0, y0 = konum[a]
        x1, y1 = konum[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(width=0.6 + 3.4 * w, color="rgba(31,95,139,0.35)")))
    # Kenar ortasında görünmez noktalar: üzerine gelince kenar bilgisi
    fig.add_trace(go.Scatter(
        x=[(konum[a][0] + konum[b][0]) / 2 for a, b in zip(secili["Kaynak"], secili["Hedef"])],
        y=[(konum[a][1] + konum[b][1]) / 2 for a, b in zip(secili["Kaynak"], secili["Hedef"])],
        mode="markers", marker=dict(size=6, opacity=0), showlegend=False,
        text=[f"{a} — {b}<br>{OLCUT_SUTUNU[olcut]}: {w:.3f}<br>Birlikte görülme: {h}"
              for a, b, w, h in zip(secili["Kaynak"], secili["Hedef"], secili["Ağırlık"], secili["Birlikte görülme"])],
        hoverinfo="text"))
    d = dugumler.set_index("Sözcük")
    sozc = list(G.nodes())
    siklik = d.loc[sozc, "Sıklık"].to_numpy(dtype=float)
    boyut = 10 + 30 * np.sqrt(siklik / siklik.max()) if len(siklik) else []
    fig.add_trace(go.Scatter(
        x=[konum[w][0] for w in sozc], y=[konum[w][1] for w in sozc], mode="markers+text", text=sozc,
        textposition="top center", showlegend=False,
        marker=dict(size=boyut, color=d.loc[sozc, "Ağırlıklı derece"], colorscale="Blues", line=dict(width=1, color="#1f5f8b"),
                    colorbar=dict(title="Ağırlıklı<br>derece")),
        hovertext=[f"{w}<br>Sıklık: {int(d.loc[w, 'Sıklık'])}<br>Derece: {int(d.loc[w, 'Derece'])}"
                   f"<br>Ağırlıklı derece: {d.loc[w, 'Ağırlıklı derece']:.3f}" for w in sozc],
        hoverinfo="text"))
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      title="Sözcük birliktelik ağı (düğüm boyutu: sıklık, renk: ağırlıklı derece)")
    return fig
