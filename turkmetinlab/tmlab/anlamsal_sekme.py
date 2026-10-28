"""DENEYSEL (2.5) anlamsal analiz sekmesi. Yalnızca TML_DENEYSEL_ANLAMSAL=1 iken app.py tarafından çağrılır.

Gizlilik: model (kullanıcı verisi içermez) st.cache_resource ile tüm oturumlarca paylaşılır; vektörler, skorlar ve
sorgular yalnızca bu oturumun belleğinde (st.session_state, "_tml_" önekiyle) tutulur ve "Verilerimi temizle" ile
silinir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from tmlab import anlamsal as an
from tmlab.analysis import belge_ngramlari
from tmlab.export import excel_bayt_ve_uyarilar, sozluk_tablosu
from tmlab.oturum import ozet
from tmlab.preprocess import derlem_on_isle

def _tr(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


_HAZIR: set = set()  # bu süreçte belleğe yüklenmiş model kimlikleri (kullanıcı verisi değil)

UYARI = ("**Deneysel modül.** Sonuçlar keşif amaçlıdır ve kesin yargı olarak kullanılmamalıdır. Anlamsal benzerlik; "
         "intihal, metin kökeni ya da içerik eşdeğerliği kanıtı değildir.")

YONTEM = f"""
**Model:** `intfloat/multilingual-e5-small` ({an.LISANS} lisansı), ONNX Runtime ile CPU'da, 32 bit (FP32). Ücretli
servis kullanılmaz; metinleriniz modele bu sunucuda verilir, dışarı gönderilmez.

**Yöntem**
- Her belge en fazla **{an.AZAMI_TOKEN} belirteçle** temsil edilir. Daha uzun belgeler için sizden seçim istenir.
- Vektörler, belirteç vektörlerinin ortalaması alınıp uzunluğu 1 olacak biçimde normalleştirilerek üretilir.
  Benzerlik kosinüs değeridir (−1 ile 1 arası).
- Belge → belge benzerliğinde iki belgeye de `query: `, aramada sorguya `query: `, belgelere `passage: ` öneki
  eklenir (modelin önerdiği kullanım).
- E5'e **özgün metin** verilir. Ön işleme ayarları yalnızca TF-IDF'yi etkiler.
- **Hibrit:** Her sorgu için TF-IDF ve E5 benzerlikleri ayrı ayrı z-puanına çevrilir ve toplanır. Hibrit puanın
  birimi yoktur; yalnızca sıralama içindir.

**Doğrulama durumu**
- 500 Türkçe akademik özette (DergiPark, çoğu mühendislik) ONNX FP32 sürümü PyTorch sürümüyle aynı sıralamaları
  üretti.
- Özet → özet görevinde nDCG@5: TF-IDF **{_tr(an.GECERLI_SONUCLAR['TF-IDF'])}**, E5
  **{_tr(an.GECERLI_SONUCLAR['E5'])}**, hibrit **{_tr(an.GECERLI_SONUCLAR['Hibrit'])}**.
  - Hibritin kazancı (+0,025), önceden belirlenen **0,03** eşiğinin altında kaldı.
  - İlgililik, yazar anahtar sözcüklerinden türetildi; **uzman yargısı değildir.**
  - Bu nedenle hiçbir yöntem genel olarak "daha doğru" kabul edilmemelidir.
- Başka alanlara ve veri türlerine genellenebilirliği test edilmedi. 8 bitlik (INT8) sürüm uyum testini geçemediği
  için kullanılmaz.
"""


@st.cache_resource(show_spinner=False, max_entries=1)
def _kodlayici(kimlik: str, klasor: str) -> an.E5Kodlayici:  # kimlik: önbellek anahtarı
    return an.E5Kodlayici(klasor)


def _model(kaynak: an.ModelKaynagi):
    if kaynak.kimlik in _HAZIR:
        return _kodlayici(kaynak.kimlik, an.onbellek_klasoru(kaynak))
    st.info("Model henüz bu sunucuda hazır değil. İlk hazırlıkta model dosyaları (yaklaşık 450 MB) bir kez "
            "indirilir ve doğrulanır; bu birkaç dakika sürebilir.")
    if not st.button("Modeli hazırla", key="e5_model_hazirla", type="primary"):
        return None
    cubuk = st.progress(0.0, "Model dosyaları denetleniyor…")
    try:
        klasor = an.model_hazirla(kaynak, lambda ad, o: cubuk.progress(min(o, 1.0), f"İndiriliyor: {ad} (%{o * 100:.0f})"))
        cubuk.progress(1.0, "Model belleğe yükleniyor…")
        kod = _kodlayici(kaynak.kimlik, klasor)
    except an.ModelHatasi as e:
        cubuk.empty()
        st.error(f"{e} Diğer sekmeler etkilenmez.")
        return None
    except Exception as e:  # ör. bozuk ONNX dosyası, yetersiz bellek
        cubuk.empty()
        st.error(f"Model yüklenemedi ({type(e).__name__}). Diğer sekmeler etkilenmez.")
        return None
    cubuk.empty()
    _HAZIR.add(kaynak.kimlik)
    return kod


def _tfidf_satirlari(bg: dict):
    """TF-IDF matrisindeki satırların derlemdeki belge sıraları (boş kalan belgeler matriste yoktur)."""
    if bg["tfidf_hata"] or bg["tfidf"] is None:
        return None
    ust = bg["tf_ayar"][0]
    dolu = [i for i, b in enumerate(bg["belgeler"]) if any(belge_ngramlari(b, k) for k in range(1, ust + 1))]
    return np.array(dolu) if len(dolu) == bg["tfidf"].analiz_edilen_belge else None


def _yan_yana(skorlar: dict, adlar, k: int, haric=None) -> pd.DataFrame:
    tablo = {"Sıra": list(range(1, k + 1))}
    for y, s in skorlar.items():
        idx = an.en_yakinlar(s, k, haric)
        tablo[f"{y}: belge"] = [adlar[i] for i in idx] + [""] * (k - len(idx))
        tablo[f"{y}: puan"] = [round(float(s[i]), 4) for i in idx] + [None] * (k - len(idx))
    return pd.DataFrame(tablo)


def goster(bg: dict, hesapla, bul, tablo_goster, md):
    st.subheader("Anlamsal analiz (deneysel)")
    st.warning(UYARI, icon="🧪")
    with st.expander("Yöntem, doğrulama durumu ve sınırlılıklar"):
        st.markdown(YONTEM)
    kaynak = an.ModelKaynagi.ortamdan()
    neden = kaynak.eksik()
    if neden:
        st.error(neden + " Diğer sekmeler etkilenmez.")
        return
    metinler, adlar_tum = list(bg["metinler"]), [str(a) for a in bg["adlar"]]
    n = len(metinler)
    if n < 2:
        st.info("Anlamsal karşılaştırma için en az 2 belge gerekir.")
        return
    if n > an.BELGE_SINIRI:
        st.error(f"Bu deneysel modül en fazla {an.BELGE_SINIRI} belgeyle çalışır (doğrulanan ölçek). Verinizde {n} "
                 "belge var; daha küçük bir alt küme yükleyin.")
        return
    kod = _model(kaynak)
    if kod is None:
        return

    # --- Belirteç sınırı: sessiz kırpma yok ----------------------------------
    uzunluk = hesapla("e5_belirtec", ozet(bg["hazirlik_anahtari"], kaynak.kimlik),
                      lambda: kod.belirtec_sayilari(metinler), 1)
    uzun = uzunluk > kod.azami_token
    mod = "hepsi"
    if uzun.any():
        st.warning(f"{int(uzun.sum())} belge {kod.azami_token} belirteç sınırını aşıyor. Model bu belgelerin "
                   "yalnızca başını görebilir. Nasıl ilerleneceğini seçin:")
        with st.expander("Sınırı aşan belgeler"):
            tablo_goster(pd.DataFrame({"Belge": np.array(adlar_tum)[uzun], "Belirteç": uzunluk[uzun]}), 240)
        mod = st.radio("Uzun belgeler", ["kirp", "disla"], key="e5_uzun", horizontal=True,
                       format_func={"kirp": f"İlk {kod.azami_token} belirteci kullan (doğrulamada kullanılan yöntem)",
                                    "disla": "Analiz dışında bırak"}.get)
    satirlar = _tfidf_satirlari(bg)
    secili = np.flatnonzero(~uzun) if mod == "disla" else np.arange(n)
    if satirlar is not None:
        secili = np.intersect1d(secili, satirlar)
    else:
        st.caption("TF-IDF hesaplanamadığı için yalnızca E5 sonuçları gösterilir.")
    if len(secili) < 2:
        st.warning("Analiz için en az 2 belge kalmalıdır.")
        return
    if len(secili) < n:
        st.caption(f"{n - len(secili)} belge analiz dışında: uzun belge seçimi ya da ön işleme sonrası boş kalan "
                   "metinler.")

    # --- Vektörler (oturum belleğinde) ---------------------------------------
    v_anahtar = ozet(bg["hazirlik_anahtari"], kaynak.kimlik)  # tüm belgeler; seçim aşağıda uygulanır
    V = bul("e5_vektor", v_anahtar)
    if V is None:
        st.caption(f"Tahmini süre: GitHub Actions ölçümüne göre yaklaşık {an.sure_tahmini(2 * n):.0f} sn. Paylaşılan "
                   "sunucuda daha uzun sürebilir; başka kullanıcılar modeli kullanıyorsa sıraya girilir.")
        if not st.button("Anlamsal vektörleri hesapla", key="e5_hesapla", type="primary"):
            return
        cubuk = st.progress(0.0, "Belgeler kodlanıyor (1/2)…")
        bekle = lambda: st.info("Model başka bir kullanıcı için çalışıyor; sıranız bekleniyor…")  # noqa: E731

        def _hesap():
            sim = kod.kodla(metinler, an.SORGU_ONEKI, lambda o: cubuk.progress(o / 2, "Belgeler kodlanıyor (1/2)…"), bekle)
            gec = kod.kodla(metinler, an.BELGE_ONEKI, lambda o: cubuk.progress(0.5 + o / 2, "Arama dizini (2/2)…"), bekle)
            return {"simetrik": sim, "belge": gec}
        try:
            V = hesapla("e5_vektor", v_anahtar, _hesap, 1)
        except Exception as e:
            cubuk.empty()
            st.error(f"Vektörler hesaplanamadı ({type(e).__name__}). Daha az belgeyle yeniden deneyin.")
            return
        st.rerun()  # düğme ve süre tahmini kaybolsun; sonuçlar oturum belleğinden çizilir

    adlar = [adlar_tum[i] for i in secili]
    E = V["simetrik"][secili]
    S = {"E5": E @ E.T}
    if satirlar is not None:
        tf_idx = np.searchsorted(satirlar, secili)
        M = bg["tfidf"].matris[tf_idx]
        S["TF-IDF"] = (M @ M.T).toarray()
        np.fill_diagonal(S["TF-IDF"], -np.inf)
        np.fill_diagonal(S["E5"], -np.inf)
        S["Hibrit"] = an.hibrit(S["TF-IDF"], S["E5"])
    else:
        np.fill_diagonal(S["E5"], -np.inf)
    sirali = {y: S[y] for y in an.YONTEMLER if y in S}
    bg["ek_ayarlar"].update({"Anlamsal (deneysel): model": an.MODEL_ADI,
                             "Anlamsal: model kaynağı": kaynak.kimlik,
                             "Anlamsal: uzun belgeler": {"hepsi": "sınırı aşan yok", "kirp": f"ilk {kod.azami_token} "
                                                         "belirteç", "disla": "dışarıda bırakıldı"}[mod],
                             "Anlamsal: analiz edilen belge": len(secili),
                             "Anlamsal: hibrit": "satır başına z-puanı toplamı (TF-IDF + E5)"})

    k = st.radio("Gösterilecek belge sayısı", [5, 10], horizontal=True, key="e5_k")
    k = min(int(k), len(secili) - 1)
    satir = []
    ad_ = list(sirali)
    ust = np.triu_indices(len(adlar), 1)
    for i, a in enumerate(ad_):
        for b in ad_[i + 1:]:
            r = float(np.corrcoef(S[a][ust], S[b][ust])[0, 1]) if len(ust[0]) > 1 else float("nan")
            satir.append({"Yöntemler": f"{a} – {b}", f"İlk {k} örtüşmesi": round(an.ust_k_ortusme(S[a], S[b], k), 3),
                          "Puan korelasyonu (Pearson)": round(r, 3)})
    karsilastirma = pd.DataFrame(satir)
    alt = st.tabs(["Belge benzerliği", "Belge arama", "Yöntem karşılaştırması"])

    with alt[0]:
        sec = st.selectbox("Belge seçin", range(len(adlar)), format_func=lambda i: adlar[i], key="e5_belge")
        st.caption("Puanlar: TF-IDF ve E5 için kosinüs benzerliği, hibrit için birimsiz z-puanı toplamı.")
        tablo_goster(_yan_yana({y: s[sec] for y, s in sirali.items()}, adlar, k), 300)

    with alt[1]:
        sorgu = st.text_input("Arama metni (ör. bir başlık ya da kısa açıklama)", key="e5_sorgu", max_chars=2000)
        if sorgu.strip():
            q = kod.kodla([sorgu.strip()], an.SORGU_ONEKI)[0]
            puan = {"E5": V["belge"][secili] @ q}
            if satirlar is not None:
                terim = derlem_on_isle([sorgu], bg["ayarlar"], None, bg["terimci"])[0]
                tf = bg["tf_ayar"]
                terimler = [t for j in range(1, tf[0] + 1) for t in belge_ngramlari(terim, j)]
                qv = an.tfidf_sorgu_vektoru(bg["tfidf"].matris, bg["tfidf"].terimler, terimler, tf[1], tf[2])
                puan["TF-IDF"] = np.asarray(M @ qv).ravel()
                puan["Hibrit"] = an.hibrit(puan["TF-IDF"][None], puan["E5"][None])[0]
                if not terimler:
                    st.caption("Sorgu, ön işleme sonrası TF-IDF terimi içermiyor; TF-IDF puanları 0'dır.")
            if kod.belirtec_sayilari([sorgu.strip()], an.SORGU_ONEKI)[0] > kod.azami_token:
                st.caption(f"Sorgu {kod.azami_token} belirteçten uzun; yalnızca başı kullanıldı.")
            tablo_goster(_yan_yana({y: puan[y] for y in an.YONTEMLER if y in puan}, adlar, k), 300)
            st.caption("Arama metni yalnızca bu ekranda kullanılır; saklanmaz ve indirilen dosyalara yazılmaz.")

    with alt[2]:
        if len(karsilastirma):
            tablo_goster(karsilastirma, 160)
            st.caption("Örtüşme: her belgenin ilk k komşusundan kaçının iki yöntemde ortak olduğu (0–1). Düşük örtüşme "
                       "yöntemlerin farklı ilişkiler yakaladığını gösterir; hangisinin doğru olduğunu göstermez.")
        referans = pd.DataFrame({"Yöntem": list(an.GECERLI_SONUCLAR),
                                 "nDCG@5 (500 DergiPark özeti, dolaylı etiket)": list(an.GECERLI_SONUCLAR.values())})
        st.markdown("**Fizibilite çalışmasındaki ölçüm (sizin verinizde değil)**")
        tablo_goster(referans, 150)

    # --- Dışa aktarma -----------------------------------------------------
    uzun_tablo = pd.concat([_yan_yana({y: s[i] for y, s in sirali.items()}, adlar, k).assign(Belge=adlar[i])
                            for i in range(len(adlar))], ignore_index=True)
    uzun_tablo = uzun_tablo[["Belge"] + [c for c in uzun_tablo.columns if c != "Belge"]]
    tablolar = {"Anlamsal en yakınlar": uzun_tablo, "Anlamsal ayarlar": sozluk_tablosu(
        {kk.split(": ", 1)[1]: v for kk, v in bg["ek_ayarlar"].items() if kk.startswith("Anlamsal")})}
    if len(karsilastirma):
        tablolar["Yöntem karşılaştırması"] = karsilastirma
    for ad, t in tablolar.items():
        bg["ek_tablolar"][ad] = (lambda t=t: t, len(t))
    veri, uyarilar = excel_bayt_ve_uyarilar(tablolar)
    st.download_button("Anlamsal sonuçları indir (.xlsx)", veri, "anlamsal_sonuclar.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="e5_indir",
                       on_click="ignore")
    for u in uyarilar:
        st.caption(u)
