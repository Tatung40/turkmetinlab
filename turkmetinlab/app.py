"""TürkMetinLab — Türkçe akademik metin madenciliği.

Çalıştırmak için:  streamlit run app.py

Gizlilik mimarisi (ayrıntı: tmlab/oturum.py):
* Kullanıcı verisinden türeyen hiçbir şey st.cache_data'ya konmaz (o önbellek
  tüm kullanıcılar arasında paylaşılır). Hesaplamalar yalnızca bu oturumun
  st.session_state alanında saklanır.
* Uygulama hiçbir veriyi diske yazmaz; beklenmeyen hatalarda sunucu günlüğüne
  yalnızca hata türü ve kod konumu yazılır, metin yazılmaz.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

import re
from collections import Counter

from tmlab import __version__
from tmlab.analysis import (TfidfHatasi, derlem_ozeti, ngram_sikligi, sozcuk_sikligi,
                            tfidf_hesapla, turkce_sirala_anahtari)
from tmlab.export import EXCEL_SATIR_SINIRI, csv_bayt, excel_bayt_ve_uyarilar, sozluk_tablosu
from tmlab.kalite import belgeleri_hazirla
from tmlab.lemma import OLCULEN_DOGRULUK, YONTEM as LEMMA_YONTEMI, kullanilabilir_mi
from tmlab.loader import (TXT_MODLARI, VeriOkumaHatasi, dosya_oku, excel_sayfalari,
                          metin_sutunu_oner, txt_oku)
from tmlab.oturum import (OturumBellegi, guvenli_hata_kaydi, oturum_verilerini_sil, ozet,
                          yuklenen_dosyalari_sil)
from tmlab.performans import (GOSTERIM_SATIR_SINIRI, ONIZLEME_BELGE_SINIRI, boyut_degerlendir,
                              sozcuk_tahmini)
from tmlab.preprocess import (BOLUTLEME_SECENEKLERI, KESME_SECENEKLERI, OnIslemeAyarlari, derlem_on_isle,
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
# Hesaplama belleği
# ---------------------------------------------------------------------------
# st.cache_data YALNIZCA herkese açık örnek veri için kullanılır. Kullanıcı verisinden
# türeyen her şey oturuma özel bellekte (st.session_state) tutulur; başka oturumlar
# erişemez ve "Verilerimi temizle" ile hemen silinir.

@st.cache_data(show_spinner=False)
def _ornek_oku(baslik_satiri: bool = True):
    with open(ORNEK_CSV, "rb") as f:
        return dosya_oku("ornek_akademik_ozetler.csv", f.read(), baslik_satiri=baslik_satiri)


def _hesapla(ad: str, anahtar: str, fn, en_fazla: int = 2):
    return OturumBellegi(st.session_state).al(ad, anahtar, fn, en_fazla)


def _md(metin) -> str:
    """Dosya adı gibi kullanıcı kaynaklı metni Markdown biçimlendirmesinden arındırır."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~])", r"\\\1", str(metin))


def _sayi(n) -> str:
    return f"{n:,}".replace(",", ".")


def verilerimi_temizle():
    """Bu oturumdaki tüm kullanıcı verilerini siler (düğme geri çağrısı)."""
    yuklenen_dosyalari_sil()
    oturum_verilerini_sil(st.session_state)
    st.session_state["yukleme_surumu"] = st.session_state.get("yukleme_surumu", 0) + 1
    for anahtar in ("ozel_durak", "ara_sozcuk"):
        if anahtar in st.session_state:
            st.session_state[anahtar] = ""
    st.session_state["_tml_temizlendi"] = True


def _tablo_goster(tablo: pd.DataFrame, yukseklik: int = 420, sira_goster: bool = False):
    # Az satırlı tablolarda boş satır görünmemesi için yükseklik içeriğe göre küçültülür
    yukseklik = max(80, min(yukseklik, 35 * (len(tablo) + 1) + 3))
    st.dataframe(tablo, hide_index=not sira_goster, width="stretch", height=yukseklik)


GIZLILIK = """
**Verileriniz nasıl işleniyor?**

- Yüklediğiniz dosyalar ve sonuçlar yalnızca **sizin oturumunuza ait bellekte** tutulur. Uygulamayı aynı anda
  kullanan başka kişiler verilerinize erişemez.
- Uygulama hiçbir veriyi **diske, veritabanına veya GitHub'a yazmaz**. Excel dosyaları da tamamen bellekte
  oluşturulur.
- **🗑️ Verilerimi şimdi temizle** düğmesi yüklenen dosyaları ve tüm sonuçları hemen siler.
- Düğmeyi kullanmadan sekmeyi kapatırsanız veriler, bağlantı koptuktan yaklaşık **2 dakika** sonra
  Streamlit tarafından sunucu belleğinden silinir.
- Beklenmeyen bir hata olursa sunucu günlüğüne yalnızca hata türü ve kod konumu yazılır; metinleriniz yazılmaz.
- Uygulama, Streamlit Community Cloud (Snowflake) sunucularında çalışır; dosyanız işlenmek için bu sunuculara
  şifreli bağlantıyla gönderilir. **Kişisel veya gizli veriler** (ör. görüşme dökümleri, KVKK kapsamındaki
  bilgiler) için verileri yüklemeden önce anonimleştirmenizi ya da uygulamayı kendi bilgisayarınızda
  çalıştırmanızı öneririz (README'de anlatılıyor).
"""

KILAVUZ = """
#### Nasıl kullanılır?
1. **Veri seçin.** Sol menüden örnek veriyle başlayabilir ya da kendi dosyanızı yükleyebilirsiniz.
   - **CSV / Excel:** Her satır bir belgedir (ör. bir makale özeti). Metinlerin bulunduğu sütunu seçin.
   - **TXT:** Bir veya daha fazla düz metin dosyası. Metnin belgelere nasıl bölüneceğini seçin.
2. **Ön işleme ayarlarını gözden geçirin.** Varsayılan ayarlar çoğu akademik metin için uygundur.
   *Ön İşleme* sekmesinde her belgenin işlenmiş halini özgün metinle yan yana görebilirsiniz.
3. **Sonuçları inceleyin.** Sekmeler arasında gezinin; grafiklerin üzerine gelince ayrıntılar görünür.
4. **İndirin.** *İndir* sekmesinden tabloları ve kullandığınız ayarları Excel veya CSV olarak alın.

#### Analizler ne anlatır?
- **Sözcük sıklığı:** Her sözcüğün tüm metinlerde kaç kez geçtiği. *Belge sıklığı*, sözcüğün kaç farklı
  belgede geçtiğidir. *Bin sözcükte* değeri, farklı uzunluktaki derlemleri karşılaştırmayı kolaylaştırır.
- **N-gram:** Yan yana geçen sözcük grupları (ör. "uzaktan eğitim"). N-gramlar varsayılan olarak cümle sonu,
  virgül, noktalı virgül ve iki noktayı aşmaz; arada çıkarılmış bir sözcük varsa (ör. "eğitim **ve**
  öğretim") yapay "eğitim öğretim" ikilisi üretilmez. Bölütleme düzeyi *Diğer ön işleme ayarları*ndan
  değiştirilebilir.
- **TF-IDF:** Bir terimin bir belgede sık, diğer belgelerde seyrek geçmesiyle yükselen ağırlık. Belgeleri
  birbirinden ayıran terimleri bulmaya yarar. En az iki belge gerekir.
- **Kelime bulutu:** Görsel bir özettir; sözcük boyutu sıklıkla ilişkilidir ama kesin karşılaştırma için
  tabloları kullanın. Sözcüklerin konum ve renklerinin bir anlamı yoktur.

#### Bilinmesi gereken sınırlılıklar
- Varsayılan olarak sözcükler **köklerine indirgenmez** ("eğitim", "eğitimi" ayrı sayılır). İsteğe bağlı
  **lemmatizasyon** açılabilir; yöntemi ve ölçülen doğruluğu ayarın yanında açıklanmıştır.
- Türkçe küçük harf kuralı nedeniyle İngilizce kısaltmalar değişebilir ("COVID" → "covıd").
  Gerekirse *Büyük harfli kısaltmaları koru* seçeneğini kullanın.
- Durak sözcük listesi genel amaçlıdır; alanınıza özgü sözcükleri ekleyebilirsiniz.
"""


# ---------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------

def ana():
    st.title("📚 TürkMetinLab")
    st.caption(f"Türkçe akademik metin madenciliği · Sürüm {__version__}")
    if st.session_state.pop("_tml_temizlendi", False):
        st.success("Yüklediğiniz dosyalar ve tüm sonuçlar bu oturumdan silindi.")

    # -----------------------------------------------------------------------
    # Kenar çubuğu: 1) Veri
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("1 · Veri")
        kaynak = st.radio("Veri kaynağı", ["Örnek veri setini kullan", "Kendi dosyamı yükle"], key="kaynak",
                          help="Uygulamayı tanımak için önce örnek veriyle deneyebilirsiniz.")

        tablo = okuma = hata = metin_sutunu = ad_sutunu = None
        veri_adi, veri_kimligi = "", ""

        if kaynak == "Örnek veri setini kullan":
            okuma = _ornek_oku()
            tablo = okuma.tablo
            veri_adi, veri_kimligi = "Örnek veri: 16 kurgusal akademik özet", "ornek"
            st.info("Örnek veri seti, uygulamayı denemek için yazılmış **kurgusal** Türkçe akademik özetlerden "
                    "oluşur; gerçek yayınlara ait değildir.")
        else:
            dosyalar = st.file_uploader(
                "CSV, Excel (.xlsx) veya TXT dosyası seçin",
                type=["csv", "xlsx", "txt"], accept_multiple_files=True,
                key=f"yukleme_{st.session_state.get('yukleme_surumu', 0)}",
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
                    st.info("Yalnızca **" + _md(", ".join(d.name for d in txtler) if son_txt else dosyalar[-1].name)
                            + "** analiz ediliyor. Kullanılmayan dosyalar: " + _md(", ".join(kullanilmayan))
                            + ". Bunları yukarıdaki listeden ✕ ile kaldırabilirsiniz.")
                try:
                    if txtler:
                        txt_modu = st.selectbox("TXT dosyasını belgelere nasıl ayıralım?", list(TXT_MODLARI),
                                                format_func=TXT_MODLARI.get, index=1 if len(txtler) == 1 else 0,
                                                help="TF-IDF için en az iki belge gerekir. Tek bir uzun metin "
                                                     "yüklediyseniz 'Her paragraf bir belge' seçin.")
                        icerik = tuple((d.name, d.getvalue()) for d in txtler)
                        veri_kimligi = ozet("txt", icerik, txt_modu)
                        okuma = _hesapla("okuma", veri_kimligi, lambda: txt_oku(list(icerik), txt_modu), 1)
                        veri_adi = ", ".join(d.name for d in txtler)
                    else:
                        d = tablolar[0]
                        bayt = d.getvalue()
                        baslik = st.checkbox("İlk satır sütun başlıklarını içeriyor", True, key="baslik_satiri",
                                             help="Dosyanızın ilk satırında sütun adları yoksa işareti kaldırın; "
                                                  "aksi halde ilk belge başlık sanılıp analiz dışında kalır.")
                        sayfa = 0
                        if d.name.lower().endswith(".xlsx"):
                            sayfalar = _hesapla("sayfalar", ozet(bayt), lambda: excel_sayfalari(bayt), 1)
                            if len(sayfalar) > 1:
                                sayfa = st.selectbox("Excel sayfası", sayfalar, key="sayfa")
                        veri_kimligi = ozet("tablo", d.name, bayt, sayfa, baslik)
                        okuma = _hesapla("okuma", veri_kimligi,
                                         lambda: dosya_oku(d.name, bayt, sayfa, "dosya", baslik), 1)
                        veri_adi = d.name
                    tablo = okuma.tablo
                except VeriOkumaHatasi as e:
                    hata = str(e)
                    st.error(hata)

        if tablo is not None:
            if list(tablo.columns) == ["belge", "metin"] and kaynak != "Örnek veri setini kullan" \
                    and okuma.ayirici is None and not veri_adi.lower().endswith(".xlsx"):
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
                                        help="Grafik ve tablolarda belgeleri tanımak için kullanılır "
                                             "(ör. başlık, kimlik).")
                ad_sutunu = None if ad_secim == ad_secenekleri[0] else ad_secim
            yinelenen_cikar = st.checkbox("Yinelenen metinleri bir kez say", False, key="yinelenen_cikar",
                                          help="Aynı metin birden fazla satırda varsa yalnızca ilki analize alınır. "
                                               "Aynı makalenin iki kez eklenmesi gibi durumlar için.")
        else:
            yinelenen_cikar = False

        if kaynak == "Kendi dosyamı yükle":
            st.button("🗑️ Verilerimi şimdi temizle", on_click=verilerimi_temizle, key="temizle",
                      help="Yüklediğiniz dosyaları ve bu oturumdaki tüm sonuçları sunucu belleğinden hemen siler.")

        # -------------------------------------------------------------------
        # 2) Ön işleme ayarları
        # -------------------------------------------------------------------
        st.header("2 · Ön işleme")
        kucuk_harf = st.checkbox("Türkçe kurallarıyla küçük harfe çevir", True, key="kucuk_harf",
                                 help="'Işık' → 'ışık', 'İstanbul' → 'istanbul'. Kapatılırsa 'Eğitim' ve 'eğitim' "
                                      "ayrı sözcük sayılır.")
        kisaltma = st.checkbox("Büyük harfli kısaltmaları koru (COVID, TÜİK)", False, key="kisaltma",
                               disabled=not kucuk_harf,
                               help="Tamamı büyük harfle yazılmış sözcükler olduğu gibi bırakılır. Türkçe kurala "
                                    "göre 'COVID' → 'covıd' olurdu. Not: Büyük harfle yazılmış başlıklardaki "
                                    "sözcükler de korunur ve küçük yazılışlarından ayrı sayılır.")
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
        kesme = st.selectbox("Kesme işaretli sözcükler", list(KESME_SECENEKLERI),
                             format_func=KESME_SECENEKLERI.get, key="kesme")
        lemma_var = kullanilabilir_mi()
        lemma = st.checkbox(
            "Lemmatizasyon (sözcükleri sözlük biçimine indir)", False, key="lemma", disabled=not lemma_var,
            help=f"Yöntem: {LEMMA_YONTEMI}. Türkçe altın standart verilerde ölçülen doğruluk: "
                 + ", ".join(f"{k} %{v * 100:.1f}".replace(".", ",") for k, v in OLCULEN_DOGRULUK.items())
                 + ". Bağlama bakmaz; eylem adları fiil köküne iner (çalışma → çalış, araştırma → araştır). "
                   "Ayrıntılar 'Ön İşleme' sekmesinde.")
        if lemma:
            st.caption("⚠️ Lemmatizasyon her sözcüğü doğru köke indirmez (yaklaşık 20 sözcükte 1 hata) ve kalıp "
                       "terimleri de değiştirir: *uzaktan eğitim → uzak eğitim*. Dönüşümleri *Ön İşleme* "
                       "sekmesindeki tablodan denetleyin.")
        with st.expander("Diğer ön işleme ayarları"):
            bolutleme = st.selectbox(
                "N-gram sınırları (bölütleme)", list(BOLUTLEME_SECENEKLERI),
                format_func=BOLUTLEME_SECENEKLERI.get, key="bolutleme",
                help="N-gramlar bu işaretleri aşmaz. Türkçe ağaç bankalarında (UD BOUN ve IMST) noktalamasız "
                     "yan yana sözcüklerin %60'ı aynı sözdizim öbeğindeyken virgülün iki yanında bu oran %3,4, "
                     "noktalı virgülde %1,3, iki noktada %2,4'tür; bu yüzden 'Katı' önerilir.")
            satir_sonu = st.checkbox(
                "Tek satır sonunu da sınır say", False, key="satir_sonu",
                help="PDF'ten kopyalanan metinlerde satırlar cümle ortasında bitebilir; bu yüzden varsayılan "
                     "olarak yalnızca boş satır (paragraf) sınırdır. Her satırı ayrı bir madde olan listeler "
                     "için açın.")
            sayilar = st.checkbox("Sayıları çıkar (2023, 3,5)", True, key="sayilar")
            url = st.checkbox("Web adreslerini, DOI ve e-postaları çıkar", True, key="url")
            tire = st.checkbox("Tireli sözcükleri tek sözcük say (sosyo-ekonomik)", True, key="tire")
            sapka = st.checkbox("Şapkalı harfleri sadeleştir (kâğıt → kağıt)", False, key="sapka",
                                help="Dikkat: 'hâlâ' ile 'hala' gibi anlamı farklı sözcükler birleşir.")
            min_uzunluk = st.number_input("En kısa sözcük uzunluğu (harf)", 1, 10, 2, key="min_uzunluk")

        ayarlar = OnIslemeAyarlari(
            kucuk_harf=kucuk_harf, kisaltmalari_koru=kisaltma and kucuk_harf, url_eposta_kaldir=url,
            sayilari_kaldir=sayilar, kesme_isareti=kesme, sapkali_harf_normallestir=sapka,
            tireli_birlesik_koru=tire, durak_sozcuk_genel=durak_genel, durak_sozcuk_akademik=durak_akademik,
            ozel_durak_sozcukler=ozel_durak_sozcukleri_ayristir(ozel_metin, sapka), min_uzunluk=int(min_uzunluk),
            bolutleme=bolutleme, satir_sonu_sinir=satir_sonu, lemmatizasyon=bool(lemma and lemma_var),
        )

        # -------------------------------------------------------------------
        # 3) Analiz ayarları
        # -------------------------------------------------------------------
        st.header("3 · Analiz")
        ilk_n = st.slider("Grafiklerde gösterilecek terim sayısı", 5, 50, 20, key="ilk_n")
        n_deger = st.selectbox("N-gram uzunluğu", [2, 3, 4],
                               format_func=lambda n: {2: "2 (ikili)", 3: "3 (üçlü)", 4: "4 (dörtlü)"}[n], key="n")
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
        st.caption("🔒 Verileriniz yalnızca bu oturumun belleğinde işlenir; diske ya da başka bir yere kaydedilmez. "
                   "Ayrıntılar *Başlarken* sekmesinde.")

    # -----------------------------------------------------------------------
    # Ana alan
    # -----------------------------------------------------------------------
    sekmeler = st.tabs(["🏠 Başlarken", "📄 Veri", "🧹 Ön İşleme", "📊 Sözcük Sıklığı", "🔗 N-gram", "⚖️ TF-IDF",
                        "☁️ Kelime Bulutu", "⬇️ İndir"])

    def bos_sekmeler(mesaj: str, hata_metni=None, veri_goster: bool = False):
        with sekmeler[0]:
            st.markdown("Bu uygulama Türkçe akademik metinlerde **sözcük sıklığı**, **n-gram** ve **TF-IDF** "
                        "analizleri yapar; sonuçları grafik, kelime bulutu ve indirilebilir tablolar olarak sunar.")
            if hata_metni:
                st.error(hata_metni)
            else:
                st.info(mesaj)
            with st.expander("🔒 Verileriniz nasıl işleniyor?"):
                st.markdown(GIZLILIK)
            st.markdown(KILAVUZ)
        for i, sk in enumerate(sekmeler[1:], start=1):
            with sk:
                if veri_goster and i == 1:
                    _veri_sekmesi(tablo, okuma)
                else:
                    st.info("Önce sol menüden veri seçin." if not hata_metni else "Analiz yapılamadı; "
                            "ayrıntılar Başlarken sekmesinde.")
        st.stop()

    if tablo is None or metin_sutunu is None:
        bos_sekmeler("👈 Başlamak için sol menüden bir dosya yükleyin ya da örnek veri setini seçin.", hata)

    # -----------------------------------------------------------------------
    # Veriyi hazırla ve denetle
    # -----------------------------------------------------------------------
    hazirlik_anahtari = ozet(veri_kimligi, metin_sutunu, ad_sutunu, yinelenen_cikar)
    metinler, adlar, satir_nolari, rapor = _hesapla(
        "hazirlik", hazirlik_anahtari, lambda: belgeleri_hazirla(tablo, metin_sutunu, ad_sutunu, yinelenen_cikar), 1)

    if not metinler:
        bos_sekmeler("", "Seçilen sütunda analiz edilebilecek metin bulunamadı. Lütfen başka bir sütun seçin.",
                     veri_goster=True)

    sozcuk = _hesapla("boyut", hazirlik_anahtari, lambda: sozcuk_tahmini(metinler), 1)
    boyut = boyut_degerlendir(sozcuk, ayarlar.lemmatizasyon)
    if boyut.duzey == "engel":
        bos_sekmeler("", boyut.mesaj, veri_goster=True)

    on_isleme_anahtari = ozet(hazirlik_anahtari, repr(ayarlar))
    with st.spinner("Metinler ön işleniyor…"):
        def _on_isle():
            kayit = Counter()
            return derlem_on_isle(metinler, ayarlar, kayit if ayarlar.lemmatizasyon else None), kayit
        belgeler, lemma_kaydi = _hesapla("onisleme", on_isleme_anahtari, _on_isle)
    ozet_tablo = _hesapla("ozet", on_isleme_anahtari, lambda: derlem_ozeti(belgeler))
    frekans = _hesapla("frekans", on_isleme_anahtari, lambda: sozcuk_sikligi(belgeler))
    ngram = _hesapla("ngram", ozet(on_isleme_anahtari, n_deger, int(min_siklik)),
                     lambda: ngram_sikligi(belgeler, n_deger, int(min_siklik)))
    tfidf_anahtari = ozet(on_isleme_anahtari, tf_ngram, tf_smooth, tf_sublinear, int(tf_min_belge),
                          int(tf_belge_basina))
    try:
        with st.spinner("TF-IDF hesaplanıyor…"):
            tfidf = _hesapla("tfidf", tfidf_anahtari, lambda: tfidf_hesapla(
                belgeler, list(adlar), tf_ngram, tf_smooth, tf_sublinear, int(tf_min_belge), int(tf_belge_basina)))
        tfidf_hata = None
    except TfidfHatasi as e:
        tfidf, tfidf_hata = None, str(e)

    def onizleme_tablosu(sinir=None):
        n = len(belgeler) if sinir is None else min(sinir, len(belgeler))
        return pd.DataFrame({
            "Belge": adlar[:n],
            "Satır": satir_nolari[:n],
            "Özgün metin": list(metinler[:n]),
            "İşlenmiş sözcükler": [" ".join(b.sozcukler) for b in belgeler[:n]],
            "Sözcük (önce)": [b.ham_sozcuk_sayisi for b in belgeler[:n]],
            "Sözcük (sonra)": [len(b.sozcukler) for b in belgeler[:n]],
        })

    lemma_tablosu = None
    if ayarlar.lemmatizasyon:
        lemma_tablosu = pd.DataFrame([(s_, l_, n_) for (s_, l_), n_ in lemma_kaydi.most_common()],
                                     columns=["Sözcük", "Lemma", "Sıklık"])

    # --- Başlarken ---------------------------------------------------------
    with sekmeler[0]:
        st.markdown(f"**Veri:** {_md(veri_adi)}")
        if boyut.duzey == "uyari":
            st.warning(boyut.mesaj)
        for u in (okuma.uyarilar if okuma else []):
            st.warning(u)
        for u in rapor.uyarilar():
            st.warning(u)
        k = st.columns(4)
        k[0].metric("Belge", _sayi(ozet_tablo["Belge sayısı"]))
        k[1].metric("Analiz edilen sözcük", _sayi(ozet_tablo["Analiz edilen sözcük (ön işleme sonrası)"]))
        k[2].metric("Farklı sözcük", _sayi(ozet_tablo["Farklı sözcük sayısı"]))
        k[3].metric("Belge başına ort. sözcük", f"{ozet_tablo['Belge başına ortalama sözcük']}".replace(".", ","))
        if ozet_tablo["Analiz edilen sözcük (ön işleme sonrası)"] == 0:
            st.error("Ön işleme sonrası hiç sözcük kalmadı. Ayarları gevşetmeyi deneyin "
                     "(ör. durak sözcük seçenekleri).")
        with st.expander("Derlem özeti (ayrıntılı)"):
            _tablo_goster(sozluk_tablosu(ozet_tablo, "Ölçüt", "Değer"), 330)
            st.caption("TTR (Tür/Sözcük oranı) = farklı sözcük sayısı ÷ toplam sözcük sayısı. Metin uzunluğundan "
                       "etkilendiği için farklı uzunluktaki derlemleri karşılaştırırken dikkatli yorumlayın.")
        with st.expander("🔒 Verileriniz nasıl işleniyor?"):
            st.markdown(GIZLILIK)
        st.markdown(KILAVUZ)

    # --- Veri --------------------------------------------------------------
    with sekmeler[1]:
        _veri_sekmesi(tablo, okuma)

    # --- Ön işleme ---------------------------------------------------------
    with sekmeler[2]:
        st.subheader("Ön işleme sonuçları")
        st.markdown("Aşağıda her belgenin özgün metni ile ön işleme sonrası analize giren sözcükler yer alır. "
                    "Ayarları sol menüden değiştirdiğinizde bu tablo ve tüm analizler kendiliğinden güncellenir.")
        _tablo_goster(onizleme_tablosu(ONIZLEME_BELGE_SINIRI), 460)
        if len(belgeler) > ONIZLEME_BELGE_SINIRI:
            st.caption(f"İlk {_sayi(ONIZLEME_BELGE_SINIRI)} belge gösteriliyor; tamamı İndir sekmesinde.")
        if lemma_tablosu is not None:
            st.markdown("##### Lemmatizasyon dönüşümleri")
            st.caption(f"Yöntem: {LEMMA_YONTEMI}. Değişen her sözcük ve kaç kez geçtiği listelenir; şüpheli "
                       "dönüşümleri bu tablodan denetleyin.")
            _tablo_goster(lemma_tablosu.head(GOSTERIM_SATIR_SINIRI), 320)
            with st.expander("Lemmatizasyon nasıl değerlendirildi?"):
                st.markdown(
                    "Yöntemler, dilbilimcilerin elle etiketlediği Türkçe ağaç bankalarında (UD Turkish-BOUN ve "
                    "UD Turkish-IMST test kümeleri, toplam 17.336 sözcük) karşılaştırıldı:\n\n"
                    "| Yöntem | BOUN | IMST |\n|---|---|---|\n"
                    "| Hiç işlem yapmamak | %45,8 | %43,0 |\n"
                    "| Snowball kök bulucu (kural tabanlı) | %57,0 | %54,3 |\n"
                    "| Zemberek (Python uyarlaması, bağlamlı) | %92,3 | %93,3 |\n"
                    "| **simplemma (kullanılan)** | **%94,5** | **%94,4** |\n\n"
                    "Zemberek'in Python uyarlaması daha yavaştır (sözcük başına ~4 ms) ve güncel Python "
                    "sürümlerinde çalışmadığı için kullanılmadı.\n\n"
                    "**Bilinen sınırlılıklar:** Bağlama bakılmaz (eş yazımlı sözcüklerde tek lemma seçilir). "
                    "Eylem adları fiil köküne iner (*çalışma → çalış*); bu, kullanılan dilbilim standardının "
                    "(Universal Dependencies) kuralıdır. Belirteçler de köke inebilir; bu yüzden kalıp terimler "
                    "değişir (*uzaktan eğitim → uzak eğitim*). Sabit terimleri incelerken lemmatizasyonu kapatın. Bazı türetim ekleri fazla soyulur (*politikacı → "
                    "politika*). Anlamı tersine çeviren *-sız/-siz* ekinin atılmasını önleyen bir koruma "
                    "kuralı uygulanır (*anlamsız → anlamsız*).")
        with st.expander("Uygulanan ön işleme adımları"):
            adimlar = ["Unicode normalleştirme (NFKC), yumuşak tire ve görünmez karakterlerin temizlenmesi",
                       "Satır sonunda tireyle bölünmüş sözcüklerin birleştirilmesi (eği-/tim → eğitim)",
                       "Bölütleme (n-gramlar bölüt aşmaz): " + BOLUTLEME_SECENEKLERI[bolutleme]
                       + ("; tek satır sonları da sınır" if satir_sonu else "; boş satır (paragraf) sınırdır")
                       + "; '19. yüzyıl' ve 'Prof. Dr.' gibi noktalar cümle sonu sayılmaz"]
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
            if ayarlar.lemmatizasyon:
                adimlar.append(f"Lemmatizasyon ({LEMMA_YONTEMI}); lemması durak sözcük olanlar (ör. ol, et) çıkarılır")
            st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(adimlar, 1)))

    # --- Sözcük sıklığı -----------------------------------------------------
    with sekmeler[3]:
        st.subheader("Sözcük sıklığı" + (" (lemma)" if ayarlar.lemmatizasyon else ""))
        if frekans.empty:
            st.warning("Gösterilecek sözcük yok.")
        else:
            st.plotly_chart(yatay_cubuk(frekans, "Sözcük", "Sıklık",
                                        f"En sık geçen {min(ilk_n, len(frekans))} sözcük", ilk_n), width="stretch")
            ara = st.text_input("Tabloda sözcük ara", "", key="ara_sozcuk", placeholder="örnek: eğitim")
            gosterilen = frekans
            if ara.strip():
                gosterilen = frekans[frekans["Sözcük"].map(turkce_kucuk_harf).str.contains(
                    turkce_kucuk_harf(ara.strip()), regex=False)]
                st.caption(f"'{ara}' içeren {_sayi(len(gosterilen))} sözcük bulundu.")
            _sinirli_tablo(gosterilen)
            st.caption("Göreli sıklık ve 'bin sözcükte' değerleri ön işleme sonrası toplam sözcük sayısına "
                       f"({_sayi(ozet_tablo['Analiz edilen sözcük (ön işleme sonrası)'])}) göre hesaplanmıştır.")

    # --- N-gram -------------------------------------------------------------
    with sekmeler[4]:
        st.subheader(f"{n_deger}-gram sıklığı")
        if ngram.empty:
            st.warning(f"En az {min_siklik} kez geçen {n_deger}-gram bulunamadı. Sol menüden 'N-gram en düşük "
                       "sıklık' değerini düşürmeyi deneyin.")
        else:
            st.plotly_chart(yatay_cubuk(ngram, "N-gram", "Sıklık", f"En sık geçen {min(ilk_n, len(ngram))} "
                                        f"{n_deger}-gram", ilk_n), width="stretch")
            _sinirli_tablo(ngram)
            st.caption("Göreli sıklık, tüm n-gramların toplam sayısına göre hesaplanır (en düşük sıklık "
                       f"süzgecinden önce). Bölütleme: {BOLUTLEME_SECENEKLERI[bolutleme]}.")

    # --- TF-IDF -------------------------------------------------------------
    with sekmeler[5]:
        st.subheader("TF-IDF analizi")
        if tfidf_hata:
            st.warning(tfidf_hata)
        else:
            st.caption(f"{_sayi(tfidf.analiz_edilen_belge)} belge ve {_sayi(len(tfidf.terimler))} terim üzerinden "
                       "hesaplandı.")
            alt1, alt2 = st.tabs(["Derlem geneli", "Belge bazında"])
            with alt1:
                st.plotly_chart(yatay_cubuk(tfidf.derlem_tablosu, "Terim", "Ortalama TF-IDF",
                                            "Ortalama TF-IDF değeri en yüksek terimler", ilk_n), width="stretch")
                harita = tfidf_isi_haritasi(tfidf, min(ilk_n, 30))
                if harita is not None:
                    st.plotly_chart(harita, width="stretch")
                else:
                    st.caption("Isı haritası 40'tan fazla belge için gösterilmez.")
                _sinirli_tablo(tfidf.derlem_tablosu)
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
                    "belgeler 0 sayılır).\n\nHesaplama scikit-learn kütüphanesinin `TfidfVectorizer` sınıfıyla "
                    "yapılır ve testlerde bağımsız bir uygulamayla karşılaştırılarak doğrulanır.")

    # --- Kelime bulutu ------------------------------------------------------
    with sekmeler[6]:
        st.subheader("Kelime bulutu")
        c1, c2, c3 = st.columns(3)
        kaynaklar = ["Sözcük sıklığı", f"{n_deger}-gram sıklığı"] + ([] if tfidf_hata else ["TF-IDF (toplam)"])
        bulut_kaynak = c1.selectbox("Sözcük boyutunu belirleyen ölçüt", kaynaklar, key="bulut_kaynak")
        bulut_n = c2.slider("En fazla sözcük", 20, 300, 100, step=10, key="bulut_n")
        renk = c3.selectbox("Renk düzeni", ["viridis", "cividis", "plasma", "Blues", "Greys", "Dark2"],
                            key="bulut_renk")
        if bulut_kaynak == "Sözcük sıklığı":
            kaynak_tablo, sutunlar_ = frekans, ("Sözcük", "Sıklık")
        elif bulut_kaynak.endswith("gram sıklığı"):
            kaynak_tablo, sutunlar_ = ngram, ("N-gram", "Sıklık")
        else:
            kaynak_tablo, sutunlar_ = tfidf.derlem_tablosu, ("Terim", "Toplam TF-IDF")
        if kaynak_tablo.empty:
            st.warning("Kelime bulutu için yeterli terim yok.")
        else:
            # Yalnızca çizilecek kadar terim kullanılır (tablolar azalan sırada)
            ust = kaynak_tablo.sort_values(sutunlar_[1], ascending=False, kind="stable").head(bulut_n)
            agirlik = dict(zip(ust[sutunlar_[0]], ust[sutunlar_[1]]))
            png = _hesapla("bulut", ozet(tfidf_anahtari, n_deger, int(min_siklik), bulut_kaynak, bulut_n, renk),
                           lambda: kelime_bulutu_png(agirlik, en_fazla=bulut_n, renk_haritasi=renk))
            st.image(png, width="stretch")
            st.download_button("Kelime bulutunu indir (PNG)", png, "kelime_bulutu.png", "image/png",
                               key="indir_png", on_click="ignore")
            st.caption("Sözcük boyutu seçilen ölçütle ilişkilidir; konum ve renklerin anlamı yoktur. Yerleşim her "
                       "seferinde aynıdır (sabit tohum değeri).")

    # --- İndir --------------------------------------------------------------
    with sekmeler[7]:
        st.subheader("Sonuçları indir")
        zaman = datetime.now().strftime("%Y-%m-%d_%H-%M")
        ayar_tablosu = sozluk_tablosu({
            "Uygulama": f"TürkMetinLab {__version__}",
            "Analiz tarihi": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Veri": veri_adi,
            "Metin sütunu": metin_sutunu,
            "Belge adı sütunu": ad_sutunu or "(sıra numarası)",
            "Yinelenen metinler bir kez sayıldı": yinelenen_cikar,
            "Analiz edilen belge": len(metinler),
            **{f"Ön işleme: {k_}": v_ for k_, v_ in ayarlar.sozluk().items()},
            "Lemmatizasyon yöntemi": LEMMA_YONTEMI if ayarlar.lemmatizasyon else "kullanılmadı",
            "N-gram uzunluğu": n_deger,
            "N-gram en düşük sıklık": min_siklik,
            "TF-IDF terim türü": "tek sözcük" if tf_ngram == 1 else "tek sözcük + ikili",
            "TF-IDF en az belge": tf_min_belge,
            "TF-IDF yumuşatılmış IDF": tf_smooth,
            "TF-IDF logaritmik TF": tf_sublinear,
        })
        tablolar = {
            "Özet": lambda: sozluk_tablosu(ozet_tablo, "Ölçüt", "Değer"),
            "Ayarlar": lambda: ayar_tablosu,
            "Sözcük sıklığı": lambda: frekans,
            f"{n_deger}-gram": lambda: ngram,
        }
        satir_sayilari = {"Sözcük sıklığı": len(frekans), f"{n_deger}-gram": len(ngram)}
        if tfidf is not None:
            tablolar["TF-IDF derlem"] = lambda: tfidf.derlem_tablosu
            tablolar["TF-IDF belge"] = lambda: tfidf.belge_tablosu
            satir_sayilari.update({"TF-IDF derlem": len(tfidf.derlem_tablosu),
                                   "TF-IDF belge": len(tfidf.belge_tablosu)})
        if lemma_tablosu is not None:
            tablolar["Lemma dönüşümleri"] = lambda: lemma_tablosu
            satir_sayilari["Lemma dönüşümleri"] = len(lemma_tablosu)
        tablolar["Ön işlenmiş metinler"] = onizleme_tablosu
        satir_sayilari["Ön işlenmiş metinler"] = len(belgeler)

        def excel_uret():
            return excel_bayt_ve_uyarilar({ad_: fn_() for ad_, fn_ in tablolar.items()})[0]

        st.markdown("**Tüm sonuçlar tek Excel dosyasında** (her tablo ayrı sayfada; kullanılan ayarlar dahil). "
                    "Dosya, düğmeye bastığınızda hazırlanır.")
        buyukler = [f"{a_} ({_sayi(n_)} satır)" for a_, n_ in satir_sayilari.items() if n_ > EXCEL_SATIR_SINIRI]
        if buyukler:
            st.info(f"Excel'de her sayfaya en fazla {_sayi(EXCEL_SATIR_SINIRI)} satır yazılır (bellek sınırı). "
                    f"Şu tabloların tamamı için CSV indirin: {', '.join(buyukler)}.")
        st.download_button("📥 Excel olarak indir (.xlsx)", excel_uret, f"turkmetinlab_{zaman}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                           key="indir_excel", on_click="ignore")
        st.divider()
        st.markdown("**Tabloları ayrı ayrı CSV olarak indir** (tabloların tamamı):")
        c1, c2 = st.columns(2)
        ayirici = c1.radio("CSV ayırıcısı", [";", ","], key="csv_ayirici",
                           format_func=lambda a: "Noktalı virgül ( ; ) — Türkçe Excel için önerilir" if a == ";"
                           else "Virgül ( , ) — R, Python, SPSS için")
        formul_korumasi = c2.checkbox(
            "Formül güvenliği (önerilen)", True, key="formul_korumasi",
            help="'=', '+', '-' veya '@' ile başlayan metin hücrelerinin başına ' eklenir; böylece dosya Excel'de "
                 "açıldığında bu hücreler formül olarak çalıştırılamaz. Veriyi R/Python ile işleyecekseniz "
                 "kapatabilirsiniz.")
        kolonlar = st.columns(2)
        for i, (ad_, fn_) in enumerate(tablolar.items()):
            dosya_adi = ad_.lower().translate(str.maketrans("ıöüşçğ ", "iouscg_")).replace("-", "")
            kolonlar[i % 2].download_button(
                f"{ad_} (.csv)", (lambda f=fn_: csv_bayt(f(), ayirici, formul_korumasi)),
                f"{dosya_adi}_{zaman}.csv", "text/csv", key=f"indir_csv_{i}", width="stretch", on_click="ignore")


def _sinirli_tablo(tablo: pd.DataFrame):
    """Büyük tabloların arayüzde yalnızca ilk satırlarını gösterir (tamamı indirilebilir)."""
    _tablo_goster(tablo.head(GOSTERIM_SATIR_SINIRI))
    if len(tablo) > GOSTERIM_SATIR_SINIRI:
        st.caption(f"{_sayi(len(tablo))} satırın ilk {_sayi(GOSTERIM_SATIR_SINIRI)} satırı gösteriliyor; tamamı "
                   "İndir sekmesinde.")


def _veri_sekmesi(tablo: pd.DataFrame, okuma):
    st.subheader("Yüklenen veri")
    bilgi = [f"{_sayi(len(tablo))} satır", f"{len(tablo.columns)} sütun"]
    if okuma and okuma.kodlama:
        bilgi.append(f"karakter kodlaması: {okuma.kodlama}")
    if okuma and okuma.ayirici:
        bilgi.append(f"ayırıcı: '{'sekme' if okuma.ayirici == chr(9) else okuma.ayirici}'")
    st.caption(" · ".join(bilgi))
    gosterim = tablo.head(200).copy()
    gosterim.index = gosterim.index + 1
    _tablo_goster(gosterim, sira_goster=True)
    if len(tablo) > 200:
        st.caption("İlk 200 satır gösteriliyor. Satır numaraları uyarılardaki numaralarla aynıdır.")


try:
    ana()
except Exception as _hata:  # st.stop()/rerun BaseException olduğu için burada yakalanmaz
    _kayit = guvenli_hata_kaydi(_hata)
    st.error("Beklenmeyen bir hata oluştu. Verileriniz günlüğe yazılmadı. Sayfayı yenileyip tekrar deneyin; "
             "sorun sürerse ayarları varsayılana döndürün ya da verinizi daha küçük parçalara bölün.")
    st.caption(f"Teknik ayrıntı (metin içermez): {_kayit}")
