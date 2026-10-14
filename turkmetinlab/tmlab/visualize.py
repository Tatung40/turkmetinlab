"""İnteraktif grafikler (Plotly) ve kelime bulutu (wordcloud)."""

from __future__ import annotations

import io
import os
from typing import Dict, Optional

import matplotlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

matplotlib.use("Agg")

ANA_RENK = "#1f5f8b"


def turkce_font_yolu() -> str:
    """Türkçe karakterleri (ğ, ş, ı, İ) içeren DejaVu Sans fontu.

    Matplotlib ile birlikte gelir; ayrıca indirme gerektirmez.
    """
    yol = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    if not os.path.exists(yol):
        raise FileNotFoundError("DejaVuSans.ttf bulunamadı.")
    return yol


def yatay_cubuk(tablo: pd.DataFrame, etiket: str, deger: str, baslik: str, ilk_n: int = 20) -> go.Figure:
    """En yüksek değerli ilk_n satır için yatay çubuk grafik."""
    veri = tablo.head(ilk_n).iloc[::-1]  # en büyük en üstte
    fig = px.bar(veri, x=deger, y=etiket, orientation="h", title=baslik,
                 color_discrete_sequence=[ANA_RENK],
                 hover_data={c: True for c in veri.columns if c not in (etiket, deger, "Sıra")})
    fig.update_layout(
        height=max(350, 24 * len(veri) + 120),
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title=None,
        xaxis_title=deger,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    fig.update_yaxes(type="category")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.25)")
    return fig


def tfidf_isi_haritasi(sonuc, ilk_n_terim: int = 20, en_fazla_belge: int = 40) -> Optional[go.Figure]:
    """Derlem genelinde en yüksek TF-IDF'li terimlerin belgelere göre dağılımı."""
    if sonuc.analiz_edilen_belge > en_fazla_belge:
        return None
    terimler = sonuc.derlem_tablosu["Terim"].head(ilk_n_terim).tolist()
    indeks = {t: i for i, t in enumerate(sonuc.terimler)}
    sutunlar = [indeks[t] for t in terimler]
    alt = sonuc.matris[:, sutunlar].toarray()
    fig = go.Figure(go.Heatmap(z=alt, x=terimler, y=sonuc.belge_adlari, colorscale="Blues",
                               hovertemplate="Belge: %{y}<br>Terim: %{x}<br>TF-IDF: %{z:.4f}<extra></extra>"))
    fig.update_layout(title="Öne çıkan terimlerin belgelere göre TF-IDF değerleri",
                      height=max(350, 22 * len(sonuc.belge_adlari) + 160),
                      margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-45)
    fig.update_yaxes(autorange="reversed", type="category")
    return fig


def kelime_bulutu_png(agirliklar: Dict[str, float], genislik: int = 1200, yukseklik: int = 600,
                      en_fazla: int = 150, renk_haritasi: str = "viridis",
                      arka_plan: str = "white") -> bytes:
    """Verilen ağırlıklardan (sıklık veya TF-IDF) kelime bulutu PNG baytları üretir.

    Yerleşim deterministiktir (random_state sabit). Sözcük boyutları
    ağırlıklarla orantılıdır; konumların anlamı yoktur.
    """
    from wordcloud import WordCloud

    agirliklar = {k: float(v) for k, v in agirliklar.items() if v and v > 0}
    if not agirliklar:
        raise ValueError("Kelime bulutu için sözcük bulunamadı.")
    bulut = WordCloud(
        width=genislik, height=yukseklik, background_color=arka_plan,
        max_words=en_fazla, font_path=turkce_font_yolu(), colormap=renk_haritasi,
        prefer_horizontal=0.9, random_state=42, collocations=False, normalize_plurals=False,
        regexp=None, include_numbers=True,
    ).generate_from_frequencies(agirliklar)
    tampon = io.BytesIO()
    bulut.to_image().save(tampon, format="PNG")
    return tampon.getvalue()
