"""TürkMetinLab — Türkçe akademik metin madenciliği (Aşama 1).

Çalıştırmak için:  streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from tmlab import __version__
from tmlab.analysis import (TfidfHatasi, derlem_ozeti, ngram_sikligi, sozcuk_sikligi,
                            tfidf_hesapla, turkce_sirala_anahtari)
from tmlab.export import csv_bayt, excel_bayt, sozluk_tablosu
from tmlab.loader import (TXT_MODLARI, VeriOkumaHatasi, dosya_oku, excel_sayfalari,
                          metin_sutunu_oner, txt_oku)
from tmlab.preprocess import (KESME_SECENEKLERI, OnIslemeAyarlari, derlem_on_isle,
                              ozel_durak_sozcukleri_ayristir, turkce_kucuk_harf)
from tmlab.stopwords import AKADEMIK_KALIP_SOZCUKLER, GENEL_DURAK_SOZCUKLER
from tmlab.visualize import kelime_bulutu_png, tfidf_isi_haritasi, yatay_cubuk

KOK = os.path.dirname(os.path.abspath(__file__))
ORNEK_CSV = os.path.join(KOK, "ornek_veri", "ornek_akademik_ozetler.csv")

st.set_page_config(page_title="TürkMetinLab", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Önbellekli hesaplama işlevleri (aynı veri ve ayarlarla tekrar hesaplanmaz)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _oku(ad: str, veri: bytes, sayfa, txt_modu: str):
    return dosya_oku(ad, veri, sayfa, txt_modu)


@st.cache_data(show_spinner=False)
def _txt_coklu(dosyalar: tuple, txt_modu: str):
    return txt_oku(list(dosyalar), txt_modu)


@st.cache_data(show_spinner=False)
def _sayfalar(veri: bytes):
    return excel_sayfalari(veri)


@st.cache_data(show_spinner="Metinler ön işleniyor…")
def _on_isle(metinler: tuple, ayarlar: OnIslemeAyarlari):
    return derlem_on_isle(metinler, ayarlar)


@st.cache_data(show_spinner="TF-IDF hesaplanıyor…")
def _tfidf(metinler: tuple, adlar: tuple, ayarlar: OnIslemeAyarlari, ngram_ust, smooth, sublinear, min_belge,
           belge_basina):
    belgeler = derlem_on_isle(metinler, ayarlar)
    return tfidf_hesapla(belgeler, list(adlar), ngram_ust, smooth, sublinear, min_belge, belge_basina)


@st.cache_data(show_spinner="Kelime bulutu çiziliyor…")
def _bulut(agirliklar: tuple, en_fazla: int, renk: str):
    return kelime_bulutu_png(dict(agirliklar), en_fazla=en_fazla, renk_haritasi=renk)


def _tekil_adlar(adlar):
    """Aynı ada sahip belgeleri ayırt etmek için sonlarına (2), (3) ekler."""
    sayac, sonuc = {}, []
    for a in adlar:
        a = str(a).strip() or "Adsız"
        if a in sayac:
            sayac[a] += 1
            sonuc.append(f"{a} ({sayac[a]})")
        else:
            sayac[a] = 1
            sonuc.append(a)
    return sonuc


def _tablo_goster(tablo: pd.DataFrame, yukseklik: int = 420, sira_goster: bool = False):
    # Az satırlı tablolarda boş satır görünmemesi için yükseklik içeriğe göre küçültülür
    yukseklik = max(80, min(yukseklik, 35 * (len(tablo) + 1) + 3))
    st.dataframe(tablo, hide_index=not sira_goster, width="stretch", height=yukseklik)


# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------

st.title("📚 TürkMetinLab")
st.caption(f"Türkçe akademik metin madenciliği · Sürüm {__version__} (Aşama 1)")

# ---------------------------------------------------------------------------
# Kenar çubuğu: 1) Veri
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1 · Veri")
    kaynak = st.radio("Veri kaynağı", ["Örnek veri setini kullan", "Kendi dosyamı yükle"], key="kaynak",
                      help="Uygulamayı tanımak için önce örnek veriyle deneyebilirsiniz.")

    tablo = None
    okuma = None
    hata = None
    metin_sutunu = None
    ad_sutunu = None
    veri_adi = ""

    if kaynak == "Örnek veri setini kullan":
        with open(ORNEK_CSV, "rb") as f:
            okuma = _oku("ornek_akademik_ozetler.csv", f.read(), 0, "dosya")
        tablo = okuma.tablo
        veri_adi = "Örnek veri: 16 kurgusal akademik özet"
        st.info("Örnek veri seti, uygulamayı denemek için yazılmış **kurgusal** Türkçe akademik özetlerden "
                "oluşur; gerçek yayınlara ait değildir.")
    else:
        dosyalar = st.file_uploader(
            "CSV, Excel (.xlsx) veya TXT dosyası seçin",
            type=["csv", "xlsx", "txt"], accept_multiple_files=True, key="yukleme",
            help="Birden fazla TXT dosyası birlikte yüklenebilir. CSV/Excel dosyalarında her seferinde tek "
                 "dosya analiz edilir (son yüklenen).",
        )
        if dosyalar:
            # Son yüklenen dosyanın türü belirleyicidir: CSV/Excel ise yalnızca o dosya,
            # TXT ise yüklenen tüm TXT dosyaları birlikte kullanılır.
            son_txt = dosyalar[-1].name.lower().endswith(".txt")
            txtler = [d for d in dosyalar if d.name.lower().endswith(".txt")] if son_txt else []
            tablolar = [] if son_txt else [dosyalar[-1]]
            kullanilmayan = [d.name for d in dosyalar if d not in txtler and d not in tablolar]
            if kullanilmayan:
                st.info("Yalnızca **" + (", ".join(d.name for d in txtler) if son_txt else dosyalar[-1].name)
                        + "** analiz ediliyor. Kullanılmayan dosyalar: " + ", ".join(kullanilmayan)
                        + ". Bunları yukarıdaki listeden ✕ ile kaldırabilirsiniz.")
            try:
                if txtler:
                    txt_modu = st.selectbox("TXT dosyasını belgelere nasıl ayıralım?", list(TXT_MODLARI),
                                            format_func=TXT_MODLARI.get, index=1 if len(txtler) == 1 else 0,
                                            help="TF-IDF için en az iki belge gerekir. Tek bir uzun metin "
                                                 "yüklediyseniz 'Her paragraf bir belge' seçin.")
                    okuma = _txt_coklu(tuple((d.name, d.getvalue()) for d in txtler), txt_modu)
                    veri_adi = ", ".join(d.name for d in txtler)
                else:
                    d = tablolar[0]
                    sayfa = 0
                    if d.name.lower().endswith(".xlsx"):
                        sayfalar = _sayfalar(d.getvalue())
                        if len(sayfalar) > 1:
                            sayfa = st.selectbox("Excel sayfası", sayfalar, key="sayfa")
                    okuma = _oku(d.name, d.getvalue(), sayfa, "dosya")
                    veri_adi = d.name
                tablo = okuma.tablo
            except VeriOkumaHatasi as e:
                hata = str(e)
                st.error(hata)

    if tablo is not None:
        if list(tablo.columns) == ["belge", "metin"] and kaynak != "Örnek veri setini kullan" and okuma.ayirici is None:
            metin_sutunu, ad_sutunu = "metin", "belge"
        else:
            sutunlar = list(tablo.columns)
            onerilen = metin_sutunu_oner(tablo)
            metin_sutunu = st.selectbox(
                "Metinlerin bulunduğu sütun", sutunlar,
                index=sutunlar.index(onerilen) if onerilen in sutunlar else 0, key="metin_sutunu",
                help="Analiz edilecek metinleri içeren sütun. En uzun metinleri içeren sütun önerildi.")
            ad_secenekleri = ["(Sıra numarası kullan)"] + [c for c in sutunlar if c != metin_sutunu]
            varsayilan_ad = ad_secenekleri.index("baslik") if "baslik" in ad_secenekleri else 0
            ad_secim = st.selectbox("Belge adı sütunu (isteğe bağlı)", ad_secenekleri, index=varsayilan_ad,
                                    key="ad_sutunu",
                                    help="Grafik ve tablolarda belgeleri tanımak için kullanılır (ör. başlık, kimlik).")
            ad_sutunu = None if ad_secim == ad_secenekleri[0] else ad_secim

    # -----------------------------------------------------------------------
    # 2) Ön işleme ayarları
    # -----------------------------------------------------------------------
    st.header("2 · Ön işleme")
    kucuk_harf = st.checkbox("Türkçe kurallarıyla küçük harfe çevir", True, key="kucuk_harf",
                             help="'Işık' → 'ışık', 'İstanbul' → 'istanbul'. Kapatılırsa 'Eğitim' ve 'eğitim' "
                                  "ayrı sözcük sayılır.")
    kisaltma = st.checkbox("Büyük harfli kısaltmaları koru (COVID, TÜİK)", False, key="kisaltma",
                           disabled=not kucuk_harf,
                           help="Tamamı büyük harfle yazılmış sözcükler olduğu gibi bırakılır. Türkçe kurala göre "
                                "'COVID' → 'covıd' olurdu. Not: Büyük harfle yazılmış başlıklardaki sözcükler de "
                                "korunur ve küçük yazılışlarından ayrı sayılır.")
    durak_genel = st.checkbox(f"Türkçe durak sözcükleri çıkar ({len(GENEL_DURAK_SOZCUKLER)} sözcük)", True,
                              key="durak_genel",
                              help="ve, ile, bu, için, olarak gibi tek başına anlam taşımayan sözcükler.")
    durak_akademik = st.checkbox("Akademik kalıp sözcükleri de çıkar", False, key="durak_akademik",
                                 help="çalışma, araştırma, amaç, bulgular, sonuç, yöntem gibi özetlerde çok sık "
                                      "geçen sözcükler. Araştırma sorunuza göre karar verin.")
    with st.expander("Durak sözcük listesini gör / kendi sözcüklerini ekle"):
        st.caption("Genel liste")
        st.write(", ".join(sorted(GENEL_DURAK_SOZCUKLER, key=turkce_sirala_anahtari)))
        st.caption("Akademik kalıp listesi")
        st.write(", ".join(sorted(AKADEMIK_KALIP_SOZCUKLER, key=turkce_sirala_anahtari)))
        ozel_metin = st.text_area("Çıkarılacak ek sözcükler (virgülle ayırın)", "", key="ozel_durak",
                                  placeholder="örnek: bulunmuştur, belirlenmiştir")
    kesme = st.selectbox("Kesme işaretli sözcükler", list(KESME_SECENEKLERI), format_func=KESME_SECENEKLERI.get,
                         key="kesme")
    with st.expander("Diğer ön işleme ayarları"):
        sayilar = st.checkbox("Sayıları çıkar (2023, 3,5)", True, key="sayilar")
        url = st.checkbox("Web adreslerini, DOI ve e-postaları çıkar", True, key="url")
        tire = st.checkbox("Tireli sözcükleri tek sözcük say (sosyo-ekonomik)", True, key="tire")
        sapka = st.checkbox("Şapkalı harfleri sadeleştir (kâğıt → kağıt)", False, key="sapka",
                            help="Dikkat: 'hâlâ' ile 'hala' gibi anlamı farklı sözcükler birleşir.")
        min_uzunluk = st.number_input("En kısa sözcük uzunluğu (harf)", 1, 10, 2, key="min_uzunluk")

    ayarlar = OnIslemeAyarlari(
        kucuk_harf=kucuk_harf, kisaltmalari_koru=kisaltma and kucuk_harf, url_eposta_kaldir=url,
        sayilari_kaldir=sayilar, kesme_isareti=kesme, sapkali_harf_normallestir=sapka, tireli_birlesik_koru=tire,
        durak_sozcuk_genel=durak_genel, durak_sozcuk_akademik=durak_akademik,
        ozel_durak_sozcukler=ozel_durak_sozcukleri_ayristir(ozel_metin, sapka), min_uzunluk=int(min_uzunluk),
    )

    # -----------------------------------------------------------------------
    # 3) Analiz ayarları
    # -----------------------------------------------------------------------
    st.header("3 · Analiz")
    ilk_n = st.slider("Grafiklerde gösterilecek terim sayısı", 5, 50, 20, key="ilk_n")
    n_deger = st.selectbox("N-gram uzunluğu", [2, 3, 4], format_func=lambda n: {2: "2 (ikili)", 3: "3 (üçlü)",
                                                                                  4: "4 (dörtlü)"}[n], key="n")
    min_siklik = st.number_input("N-gram en düşük sıklık", 1, 100, 2, key="min_siklik",
                                 help="Bu değerden az geçen n-gramlar tabloya alınmaz.")
    with st.expander("TF-IDF ayarları"):
        tf_ngram = st.selectbox("Terim türü", [1, 2], format_func=lambda v: "Tek sözcük" if v == 1
                                else "Tek sözcük + ikili (bigram)", key="tf_ngram")
        tf_min_belge = st.number_input("Terim en az kaç belgede geçsin?", 1, 50, 1, key="tf_min_belge")
        tf_smooth = st.checkbox("Yumuşatılmış IDF (önerilen)", True, key="tf_smooth")
        tf_sublinear = st.checkbox("Logaritmik TF (1 + ln tf)", False, key="tf_sublinear",
                                   help="Çok sık geçen terimlerin ağırlığını yumuşatır.")
        tf_belge_basina = st.number_input("Belge başına gösterilecek terim", 3, 50, 10, key="tf_belge_basina")

    st.divider()
    st.caption("🔒 Yüklediğiniz dosyalar yalnızca bu oturum süresince bellekte işlenir; hiçbir yere kaydedilmez.")

# ---------------------------------------------------------------------------
# Ana alan
# ---------------------------------------------------------------------------

sekmeler = st.tabs(["🏠 Başlarken", "📄 Veri", "🧹 Ön İşleme", "📊 Sözcük Sıklığı", "🔗 N-gram", "⚖️ TF-IDF",
                    "☁️ Kelime Bulutu", "⬇️ İndir"])

KILAVUZ = """
#### Nasıl kullanılır?
1. **Veri seçin.** Sol menüden örnek veriyle başlayabilir ya da kendi dosyanızı yükleyebilirsiniz.
   - **CSV / Excel:** Her satır bir belgedir (ör. bir makale özeti). Metinlerin bulunduğu sütunu seçin.
   - **TXT:** Bir veya daha fazla düz metin dosyası. Metnin belgelere nasıl bölüneceğini seçin.
2. **Ön işleme ayarlarını gözden geçirin.** Varsayılan ayarlar çoğu akademik metin için uygundur.
   *Ön İşleme* sekmesinde her belgenin işlenmiş halini özgün metinle yan yana görebilirsiniz.
3. **Sonuçları inceleyin.** Sekmeler arasında gezinin; grafiklerin üzerine gelince ayrıntılar görünür.
4. **İndirin.** *İndir* sekmesinden tüm tabloları ve kullandığınız ayarları Excel veya CSV olarak alın.

#### Analizler ne anlatır?
- **Sözcük sıklığı:** Her sözcüğün tüm metinlerde kaç kez geçtiği. *Belge sıklığı*, sözcüğün kaç farklı
  belgede geçtiğidir. *Bin sözcükte* değeri, farklı uzunluktaki derlemleri karşılaştırmayı kolaylaştırır.
- **N-gram:** Yan yana geçen sözcük grupları (ör. "uzaktan eğitim"). N-gramlar cümle sınırlarını aşmaz;
  arada çıkarılmış bir sözcük varsa (ör. "eğitim **ve** öğretim") yapay "eğitim öğretim" ikilisi üretilmez.
- **TF-IDF:** Bir terimin bir belgede sık, diğer belgelerde seyrek geçmesiyle yükselen ağırlık. Belgeleri
  birbirinden ayıran terimleri bulmaya yarar. En az iki belge gerekir.
- **Kelime bulutu:** Görsel bir özettir; sözcük boyutu sıklıkla ilişkilidir ama kesin karşılaştırma için
  tabloları kullanın. Sözcüklerin konum ve renklerinin bir anlamı yoktur.

#### Bilinmesi gereken sınırlılıklar (Aşama 1)
- Sözcükler **köklerine indirgenmez**: "eğitim", "eğitimi", "eğitimde" ayrı sözcükler olarak sayılır.
- Türkçe küçük harf kuralı nedeniyle İngilizce kısaltmalar değişebilir ("COVID" → "covıd").
  Gerekirse *Büyük harfli kısaltmaları koru* seçeneğini kullanın.
- Durak sözcük listesi genel amaçlıdır; alanınıza özgü sözcükleri ekleyebilirsiniz.
"""

if tablo is None or metin_sutunu is None:
    with sekmeler[0]:
        st.markdown("Bu uygulama Türkçe akademik metinlerde **sözcük sıklığı**, **n-gram** ve **TF-IDF** "
                    "analizleri yapar; sonuçları grafik, kelime bulutu ve indirilebilir tablolar olarak sunar.")
        if hata:
            st.error(hata)
        else:
            st.info("👈 Başlamak için sol menüden bir dosya yükleyin ya da örnek veri setini seçin.")
        st.markdown(KILAVUZ)
    for s in sekmeler[1:]:
        with s:
            st.info("Önce sol menüden veri seçin.")
    st.stop()

# ---------------------------------------------------------------------------
# Veriyi hazırla
# ---------------------------------------------------------------------------

ham_metinler = tablo[metin_sutunu]
gecerli = ham_metinler.notna() & (ham_metinler.astype(str).str.strip() != "")
atlanan = int((~gecerli).sum())
veri = tablo[gecerli]
metinler = tuple(veri[metin_sutunu].astype(str).tolist())
if ad_sutunu:
    adlar = _tekil_adlar(veri[ad_sutunu].fillna("").astype(str).tolist())
else:
    # Etiket, 'Veri' sekmesindeki satır numarasıyla aynıdır (boş satırlar atlansa da eşleşir)
    adlar = [f"Satır {i + 1}" for i in veri.index]

if not metinler:
    st.error("Seçilen sütunda metin bulunamadı. Lütfen başka bir sütun seçin.")
    st.stop()

belgeler = _on_isle(metinler, ayarlar)
ozet = derlem_ozeti(belgeler)
frekans = sozcuk_sikligi(belgeler)
ngram = ngram_sikligi(belgeler, n_deger, int(min_siklik))
try:
    tfidf = _tfidf(metinler, tuple(adlar), ayarlar, tf_ngram, tf_smooth, tf_sublinear, int(tf_min_belge),
                   int(tf_belge_basina))
    tfidf_hata = None
except TfidfHatasi as e:
    tfidf, tfidf_hata = None, str(e)

# --- Başlarken -------------------------------------------------------------
with sekmeler[0]:
    st.markdown(f"**Veri:** {veri_adi}")
    for u in (okuma.uyarilar if okuma else []):
        st.warning(u)
    if atlanan:
        st.warning(f"{atlanan} satır boş olduğu için analize alınmadı.")
    k = st.columns(4)
    k[0].metric("Belge", f"{ozet['Belge sayısı']:,}".replace(",", "."))
    k[1].metric("Analiz edilen sözcük", f"{ozet['Analiz edilen sözcük (ön işleme sonrası)']:,}".replace(",", "."))
    k[2].metric("Farklı sözcük", f"{ozet['Farklı sözcük sayısı']:,}".replace(",", "."))
    k[3].metric("Belge başına ort. sözcük", f"{ozet['Belge başına ortalama sözcük']}".replace(".", ","))
    if ozet["Analiz edilen sözcük (ön işleme sonrası)"] == 0:
        st.error("Ön işleme sonrası hiç sözcük kalmadı. Ayarları gevşetmeyi deneyin (ör. durak sözcük seçenekleri).")
    with st.expander("Derlem özeti (ayrıntılı)"):
        _tablo_goster(sozluk_tablosu(ozet, "Ölçüt", "Değer"), 330)
        st.caption("TTR (Tür/Sözcük oranı) = farklı sözcük sayısı ÷ toplam sözcük sayısı. Metin uzunluğundan "
                   "etkilendiği için farklı uzunluktaki derlemleri karşılaştırırken dikkatli yorumlayın.")
    st.markdown(KILAVUZ)

# --- Veri ------------------------------------------------------------------
with sekmeler[1]:
    st.subheader("Yüklenen veri")
    bilgi = [f"{len(tablo)} satır", f"{len(tablo.columns)} sütun"]
    if okuma and okuma.kodlama:
        bilgi.append(f"karakter kodlaması: {okuma.kodlama}")
    if okuma and okuma.ayirici:
        bilgi.append(f"ayırıcı: '{'sekme' if okuma.ayirici == chr(9) else okuma.ayirici}'")
    st.caption(" · ".join(bilgi))
    gosterim = tablo.head(200).copy()
    gosterim.index = gosterim.index + 1
    _tablo_goster(gosterim, sira_goster=True)
    if len(tablo) > 200:
        st.caption("İlk 200 satır gösteriliyor.")

# --- Ön işleme -------------------------------------------------------------
with sekmeler[2]:
    st.subheader("Ön işleme sonuçları")
    st.markdown("Aşağıda her belgenin özgün metni ile ön işleme sonrası analize giren sözcükler yer alır. "
                "Ayarları sol menüden değiştirdiğinizde bu tablo ve tüm analizler kendiliğinden güncellenir.")
    onizleme = pd.DataFrame({
        "Belge": adlar,
        "Özgün metin": list(metinler),
        "İşlenmiş sözcükler": [" ".join(b.sozcukler) for b in belgeler],
        "Sözcük (önce)": [b.ham_sozcuk_sayisi for b in belgeler],
        "Sözcük (sonra)": [len(b.sozcukler) for b in belgeler],
    })
    _tablo_goster(onizleme, 460)
    with st.expander("Uygulanan ön işleme adımları"):
        adimlar = ["Unicode normalleştirme (NFKC), yumuşak tire ve görünmez karakterlerin temizlenmesi",
                   "Satır sonunda tireyle bölünmüş sözcüklerin birleştirilmesi (eği-/tim → eğitim)",
                   "Metnin noktalama ve satır sonlarına göre bölütlere ayrılması (n-gramlar bölüt aşmaz)"]
        if url:
            adimlar.append("Web adresleri, DOI numaraları ve e-posta adreslerinin çıkarılması")
        if kucuk_harf:
            adimlar.append("Türkçe kurallarıyla küçük harfe çevirme (I → ı, İ → i)"
                           + (" — tamamı büyük harfli kısaltmalar korunarak" if ayarlar.kisaltmalari_koru else ""))
        if sapka:
            adimlar.append("Şapkalı harflerin sadeleştirilmesi (â → a, î → i, û → u)")
        adimlar.append("Kesme işareti: " + KESME_SECENEKLERI[kesme])
        if sayilar:
            adimlar.append("Sayıların çıkarılması")
        if durak_genel or durak_akademik or ayarlar.ozel_durak_sozcukler:
            adimlar.append("Durak sözcüklerin çıkarılması")
        adimlar.append(f"{ayarlar.min_uzunluk} harften kısa sözcüklerin çıkarılması")
        st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(adimlar, 1)))

# --- Sözcük sıklığı ---------------------------------------------------------
with sekmeler[3]:
    st.subheader("Sözcük sıklığı")
    if frekans.empty:
        st.warning("Gösterilecek sözcük yok.")
    else:
        st.plotly_chart(yatay_cubuk(frekans, "Sözcük", "Sıklık", f"En sık geçen {min(ilk_n, len(frekans))} sözcük",
                                    ilk_n), width="stretch")
        ara = st.text_input("Tabloda sözcük ara", "", key="ara_sozcuk", placeholder="örnek: eğitim")
        gosterilen = frekans
        if ara.strip():
            gosterilen = frekans[frekans["Sözcük"].map(turkce_kucuk_harf).str.contains(
                turkce_kucuk_harf(ara.strip()), regex=False)]
            st.caption(f"'{ara}' içeren {len(gosterilen)} sözcük bulundu.")
        _tablo_goster(gosterilen)
        st.caption("Göreli sıklık ve 'bin sözcükte' değerleri ön işleme sonrası toplam sözcük sayısına "
                   f"({ozet['Analiz edilen sözcük (ön işleme sonrası)']}) göre hesaplanmıştır.")

# --- N-gram -----------------------------------------------------------------
with sekmeler[4]:
    st.subheader(f"{n_deger}-gram sıklığı")
    if ngram.empty:
        st.warning(f"En az {min_siklik} kez geçen {n_deger}-gram bulunamadı. Sol menüden 'N-gram en düşük "
                   "sıklık' değerini düşürmeyi deneyin.")
    else:
        st.plotly_chart(yatay_cubuk(ngram, "N-gram", "Sıklık", f"En sık geçen {min(ilk_n, len(ngram))} "
                                    f"{n_deger}-gram", ilk_n), width="stretch")
        _tablo_goster(ngram)
        st.caption("Göreli sıklık, tüm n-gramların toplam sayısına göre hesaplanır (en düşük sıklık süzgecinden "
                   "önce). N-gramlar cümle/ifade sınırlarını aşmaz.")

# --- TF-IDF -----------------------------------------------------------------
with sekmeler[5]:
    st.subheader("TF-IDF analizi")
    if tfidf_hata:
        st.warning(tfidf_hata)
    else:
        st.caption(f"{tfidf.analiz_edilen_belge} belge ve {len(tfidf.terimler)} terim üzerinden hesaplandı.")
        alt1, alt2 = st.tabs(["Derlem geneli", "Belge bazında"])
        with alt1:
            st.plotly_chart(yatay_cubuk(tfidf.derlem_tablosu, "Terim", "Ortalama TF-IDF",
                                        "Ortalama TF-IDF değeri en yüksek terimler", ilk_n), width="stretch")
            harita = tfidf_isi_haritasi(tfidf, min(ilk_n, 30))
            if harita is not None:
                st.plotly_chart(harita, width="stretch")
            else:
                st.caption("Isı haritası 40'tan fazla belge için gösterilmez.")
            _tablo_goster(tfidf.derlem_tablosu)
        with alt2:
            secili = st.selectbox("Belge seçin", tfidf.belge_adlari, key="tfidf_belge")
            belge_t = tfidf.belge_tablosu[tfidf.belge_tablosu["Belge"] == secili]
            st.plotly_chart(yatay_cubuk(belge_t, "Terim", "TF-IDF", f"“{secili}” için en ayırt edici terimler",
                                        int(tf_belge_basina)), width="stretch")
            _tablo_goster(belge_t, 380)
        with st.expander("TF-IDF nasıl hesaplanır?"):
            st.markdown(
                "- **TF (terim sıklığı):** Terimin belgede geçme sayısı"
                + (" (logaritmik: 1 + ln tf)" if tf_sublinear else "") + ".\n"
                "- **IDF (ters belge sıklığı):** "
                + ("ln((1 + N) / (1 + df)) + 1" if tf_smooth else "ln(N / df) + 1")
                + " — N: belge sayısı, df: terimi içeren belge sayısı.\n"
                "- **TF-IDF = TF × IDF**; her belgenin vektörü uzunluğu 1 olacak biçimde (L2) normalleştirilir.\n"
                "- *Ortalama TF-IDF*, terimin tüm belgelerdeki değerlerinin ortalamasıdır (terim geçmeyen "
                "belgeler 0 sayılır).\n\nHesaplama scikit-learn kütüphanesinin `TfidfVectorizer` sınıfıyla yapılır.")

# --- Kelime bulutu ----------------------------------------------------------
with sekmeler[6]:
    st.subheader("Kelime bulutu")
    c1, c2, c3 = st.columns(3)
    kaynaklar = ["Sözcük sıklığı", f"{n_deger}-gram sıklığı"] + ([] if tfidf_hata else ["TF-IDF (toplam)"])
    bulut_kaynak = c1.selectbox("Sözcük boyutunu belirleyen ölçüt", kaynaklar, key="bulut_kaynak")
    bulut_n = c2.slider("En fazla sözcük", 20, 300, 100, step=10, key="bulut_n")
    renk = c3.selectbox("Renk düzeni", ["viridis", "cividis", "plasma", "Blues", "Greys", "Dark2"],
                        key="bulut_renk")
    if bulut_kaynak == "Sözcük sıklığı":
        agirlik = dict(zip(frekans["Sözcük"], frekans["Sıklık"]))
    elif bulut_kaynak.endswith("gram sıklığı"):
        agirlik = dict(zip(ngram["N-gram"], ngram["Sıklık"]))
    else:
        agirlik = dict(zip(tfidf.derlem_tablosu["Terim"], tfidf.derlem_tablosu["Toplam TF-IDF"]))
    if not agirlik:
        st.warning("Kelime bulutu için yeterli terim yok.")
    else:
        png = _bulut(tuple(agirlik.items()), bulut_n, renk)
        st.image(png, width="stretch")
        st.download_button("Kelime bulutunu indir (PNG)", png, "kelime_bulutu.png", "image/png", key="indir_png")
        st.caption("Sözcük boyutu seçilen ölçütle ilişkilidir; konum ve renklerin anlamı yoktur. Yerleşim her "
                   "seferinde aynıdır (sabit tohum değeri).")

# --- İndir ------------------------------------------------------------------
with sekmeler[7]:
    st.subheader("Sonuçları indir")
    zaman = datetime.now().strftime("%Y-%m-%d_%H-%M")
    ayar_tablosu = sozluk_tablosu({
        "Uygulama": f"TürkMetinLab {__version__}",
        "Analiz tarihi": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Veri": veri_adi,
        "Metin sütunu": metin_sutunu,
        **{f"Ön işleme: {k}": v for k, v in ayarlar.sozluk().items()},
        "N-gram uzunluğu": n_deger,
        "N-gram en düşük sıklık": min_siklik,
        "TF-IDF terim türü": "tek sözcük" if tf_ngram == 1 else "tek sözcük + ikili",
        "TF-IDF en az belge": tf_min_belge,
        "TF-IDF yumuşatılmış IDF": tf_smooth,
        "TF-IDF logaritmik TF": tf_sublinear,
    })
    tablolar = {
        "Özet": sozluk_tablosu(ozet, "Ölçüt", "Değer"),
        "Ayarlar": ayar_tablosu,
        "Sözcük sıklığı": frekans,
        f"{n_deger}-gram": ngram,
    }
    if tfidf is not None:
        tablolar["TF-IDF derlem"] = tfidf.derlem_tablosu
        tablolar["TF-IDF belge"] = tfidf.belge_tablosu
    tablolar["Ön işlenmiş metinler"] = onizleme

    st.markdown("**Tüm sonuçlar tek Excel dosyasında** (her tablo ayrı sayfada; kullanılan ayarlar dahil):")
    st.download_button("📥 Excel olarak indir (.xlsx)", excel_bayt(tablolar), f"turkmetinlab_{zaman}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                       key="indir_excel")
    st.divider()
    st.markdown("**Tabloları ayrı ayrı CSV olarak indir:**")
    ayirici = st.radio("CSV ayırıcısı", [";", ","], horizontal=True, key="csv_ayirici",
                       format_func=lambda a: "Noktalı virgül ( ; ) — Türkçe Excel için önerilir" if a == ";"
                       else "Virgül ( , ) — R, Python, SPSS için")
    kolonlar = st.columns(2)
    for i, (ad, t) in enumerate(tablolar.items()):
        dosya_adi = ad.lower().replace(" ", "_").replace("ı", "i").replace("ö", "o").replace("ü", "u") \
            .replace("ş", "s").replace("ç", "c").replace("ğ", "g").replace("Ö", "o").replace("-", "")
        kolonlar[i % 2].download_button(f"{ad} (.csv)", csv_bayt(t, ayirici), f"{dosya_adi}_{zaman}.csv",
                                        "text/csv", key=f"indir_csv_{i}", width="stretch")
