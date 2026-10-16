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

import numpy as np
import pandas as pd
import streamlit as st

import re
from collections import Counter

from tmlab import __version__
from tmlab import benzerlik as bz
from tmlab import birliktelik as ba
from tmlab import grup as gr
from tmlab import konu as km
from tmlab import terimler as tm
from tmlab.analysis import (TfidfHatasi, derlem_ozeti, ngram_sikligi, sozcuk_sikligi,
                            tfidf_hesapla, turkce_sirala_anahtari)
from tmlab.export import EXCEL_SATIR_SINIRI, csv_bayt, excel_bayt_ve_uyarilar, sozluk_tablosu
from tmlab.kalite import belgeleri_hazirla, ingilizce_iceren
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
    for anahtar in ("ozel_durak", "ara_sozcuk", "terim_yeni", "terim_yeni_var"):
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
- **🗑️ Verilerimi şimdi temizle** düğmesi yüklenen dosyaları ve tüm sonuçları hemen siler; terim sözlüğüne
  eklediğiniz ya da yüklediğiniz terimler de silinir (bunlar da yalnızca oturumunuzda tutulur).
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
- **Akademik terim sözlüğü (isteğe bağlı):** "uzaktan eğitime", "yapay zekânın" gibi ek almış çok sözcüklü
  terimleri tanır ve analizlerde tek kavram olarak sayar (`uzaktan_eğitim`). Sözlükteki 637 terim **uzman
  doğrulaması bekleyen aday terimlerdir**. Ayrıntılar *Terimler* sekmesinde.
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
        durak_ingilizce = st.checkbox("İngilizce durak sözcükleri de çıkar (the, of, and…)", False,
                                      key="durak_ingilizce",
                                      help="Türkçe akademik kayıtlarda İngilizce özet ya da başlık da varsa açın. "
                                           "Uygulama İngilizce içeren belgeleri Başlarken sekmesinde bildirir.")
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
                       "sekmesindeki tablodan denetleyin. Çok sözcüklü terimleri korumak için aşağıdaki "
                       "*Akademik terim sözlüğü*nü açabilirsiniz.")
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
            durak_sozcuk_ingilizce=durak_ingilizce,
        )

        terim_ayar = _terim_kenar_cubugu()

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
    sekmeler = st.tabs(["🏠 Başlarken", "📄 Veri", "🧹 Ön İşleme", "📚 Terimler", "📊 Sözcük Sıklığı", "🔗 N-gram",
                        "⚖️ TF-IDF", "🧭 Benzerlik", "🕸️ Sözcük Ağı", "🧩 Konular", "👥 Gruplar", "☁️ Kelime Bulutu",
                        "⬇️ İndir"])

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

    terimci = _terim_eslestirici(terim_ayar)
    on_isleme_anahtari = ozet(hazirlik_anahtari, repr(ayarlar), terimci.parmak_izi if terimci else None)
    with st.spinner("Metinler ön işleniyor…"):
        def _on_isle():
            kayit = Counter()
            return derlem_on_isle(metinler, ayarlar, kayit if ayarlar.lemmatizasyon else None, terimci), kayit
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
            **({"Normalize terimler": ["; ".join(e.standart for e in b.terimler) for b in belgeler[:n]]}
               if terimci is not None else {}),
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
        ingilizce = _hesapla("ingilizce", on_isleme_anahtari, lambda: ingilizce_iceren(metinler), 1)
        if ingilizce:
            nolar_ = ", ".join(str(satir_nolari[i]) for i in ingilizce[:10]) + (" …" if len(ingilizce) > 10 else "")
            st.warning(f"{len(ingilizce)} belge belirgin miktarda İngilizce metin içeriyor (satır: {nolar_}). "
                       "İngilizce sözcükler sıklık, ağ ve konu sonuçlarını bozabilir"
                       + ("; İngilizce durak sözcükler şu an çıkarılıyor." if ayarlar.durak_sozcuk_ingilizce else
                          ". Ön işleme bölümünde 'İngilizce durak sözcükleri de çıkar' seçeneğini açabilir ya da bu "
                          "metinleri dosyanızdan ayırabilirsiniz."))
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
            if terimci is not None:
                adimlar.insert(0, "Akademik terim sözlüğü: terimler özgün metinde bulunur ("
                               + ("tek birim olarak korunur; bu birimler küçük harf ve şapka ayarı dışındaki adımlardan "
                                  "geçmez" if terimci.temsil == "birim" else "yalnızca işaretlenir") + ")")
            st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(adimlar, 1)))

    # --- Sözcük sıklığı -----------------------------------------------------
    with sekmeler[4]:
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
    with sekmeler[5]:
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
    with sekmeler[6]:
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

    # --- 2.0 analiz modülleri ------------------------------------------------
    # Her modül, indirilecek tabloları (ek_tablolar) ve yöntemsel ayarlarını (ek_ayarlar) kaydeder.
    ek_tablolar, ek_ayarlar = {}, {}
    baglam = dict(belgeler=belgeler, adlar=adlar, metinler=metinler, tfidf=tfidf, tfidf_hata=tfidf_hata,
                  tfidf_anahtari=tfidf_anahtari, on_isleme_anahtari=on_isleme_anahtari, ayarlar=ayarlar,
                  tablo=tablo, satir_nolari=satir_nolari, metin_sutunu=metin_sutunu, ad_sutunu=ad_sutunu,
                  ilk_n=ilk_n, ek_tablolar=ek_tablolar, ek_ayarlar=ek_ayarlar, kaynak=kaynak,
                  toplam_sozcuk=int(ozet_tablo["Analiz edilen sözcük (ön işleme sonrası)"]),
                  terimci=terimci, terim_ayar=terim_ayar, hazirlik_anahtari=hazirlik_anahtari, n_deger=n_deger,
                  min_siklik=int(min_siklik), tf_ayar=(tf_ngram, tf_smooth, tf_sublinear, int(tf_min_belge),
                                                      int(tf_belge_basina)))
    with sekmeler[3]:
        _terim_sekmesi(baglam)
    with sekmeler[7]:
        _benzerlik_sekmesi(baglam)
    with sekmeler[8]:
        _ag_sekmesi(baglam)
    with sekmeler[9]:
        _konu_sekmesi(baglam)
    with sekmeler[10]:
        _grup_sekmesi(baglam)

    # --- Kelime bulutu ------------------------------------------------------
    with sekmeler[11]:
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
    with sekmeler[12]:
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
            **_terim_ayar_sozlugu(terim_ayar, terimci),
            **ek_ayarlar,
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
        for ad_, (fn_, n_) in ek_tablolar.items():
            tablolar[ad_] = fn_
            satir_sayilari[ad_] = n_
        tablolar["Ön işlenmiş metinler"] = onizleme_tablosu
        satir_sayilari["Ön işlenmiş metinler"] = len(belgeler)
        st.download_button("📄 Temizlenmiş metinleri indir (.txt)",
                           lambda: tm.temizlenmis_txt(belgeler, adlar), f"temizlenmis_metinler_{zaman}.txt",
                           "text/plain", key="indir_txt", on_click="ignore",
                           help="Her belge için bir başlık satırı (### belge adı) ve ön işlenmiş metin. Terim "
                                "sözlüğü açıksa terimler alt çizgiyle birleşik yazılır (uzaktan_eğitim). Aynı metinler "
                                "özgün metinle yan yana 'Ön işlenmiş metinler' tablosunda (Excel/CSV) da var.")

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


OTOMATIK_SOZCUK = 300_000  # bu büyüklüğün üstünde ağır modüller düğmeyle başlar


def _kapili_hesapla(bg: dict, ad: str, anahtar: str, fn, dugme: str, aciklama: str):
    """Küçük/orta derlemde hemen hesaplar; büyük derlemde düğmeye basılınca (sonuç oturumda saklanır)."""
    if bg["toplam_sozcuk"] <= OTOMATIK_SOZCUK:
        return _hesapla(ad, anahtar, fn, 1)
    sonuc = _bul(ad, anahtar)
    if sonuc is None:
        st.caption(f"Büyük derlem ({_sayi(bg['toplam_sozcuk'])} sözcük): {aciklama} Sayfanın hızlı kalması için "
                   "yalnızca istendiğinde hesaplanır.")
        if st.button(dugme, key=f"{ad}_hesapla", type="primary"):
            sonuc = _hesapla(ad, anahtar, fn, 1)
    return sonuc


def _bul(ad: str, anahtar: str):
    """Düğmeyle başlatılan analizlerin oturum belleğindeki sonucunu (varsa) döndürür."""
    return OturumBellegi(st.session_state).bul(ad, anahtar)


# ---------------------------------------------------------------------------
# Modül 1 — Belge benzerliği
# ---------------------------------------------------------------------------

def _benzerlik_sekmesi(bg: dict):
    st.subheader("Belge benzerliği")
    st.info(bz.UYARI_METNI, icon="ℹ️")
    tfidf = bg["tfidf"]
    if bg["tfidf_hata"]:
        st.warning(bg["tfidf_hata"])
        return
    adlar = tfidf.belge_adlari
    n = tfidf.analiz_edilen_belge
    anahtar = bg["tfidf_anahtari"]
    st.caption(f"{_sayi(n)} belge · TF-IDF sekmesindeki ayarlarla hesaplanan vektörler kullanılır "
               "(ön işleme ve TF-IDF ayarlarını değiştirmek sonuçları değiştirir).")
    c1, c2 = st.columns(2)
    cift_n = c1.number_input("Gösterilecek en benzer çift sayısı", 5, 500, 20, key="benz_cift_n")
    komsu_k = c2.number_input("Belge başına en yakın komşu sayısı", 1, 50, 5, key="benz_komsu_k")
    bg["ek_ayarlar"].update({"Benzerlik: ölçüt": "TF-IDF kosinüs benzerliği",
                             "Benzerlik: en benzer çift sayısı": int(cift_n),
                             "Benzerlik: belge başına komşu": int(komsu_k)})

    if n <= bz.YOGUN_MATRIS_SINIRI:
        S = _hesapla("benzerlik", anahtar, lambda: bz.kosinus_matrisi(tfidf.matris), 1)
        ozet_ = bz.benzerlik_ozeti(S)
        m = st.columns(4)
        m[0].metric("Belge çifti", _sayi(ozet_["Belge çifti sayısı"]))
        m[1].metric("Ortalama benzerlik", f"{ozet_['Ortalama benzerlik']:.3f}".replace(".", ","))
        m[2].metric("En yüksek benzerlik", f"{ozet_['En yüksek benzerlik']:.3f}".replace(".", ","))
        m[3].metric("Ortak terimsiz çift", f"%{ozet_['Hiç ortak terimi olmayan çiftler (%)']:.1f}".replace(".", ","))
        ciftler = bz.en_benzer_ciftler(S, adlar, int(cift_n))
        komsular = _hesapla("benzerlik_komsu", ozet(anahtar, int(komsu_k)),
                            lambda: bz.komsu_tablosu(*bz.en_yakin_komsular(tfidf.matris, int(komsu_k)), adlar), 1)
    else:
        S = None
        st.warning(f"{_sayi(n)} belge için tam benzerlik matrisi, ısı haritası ve kümeleme hesaplanmaz (bellek "
                   f"sınırı: {_sayi(bz.YOGUN_MATRIS_SINIRI)} belge). Her belgenin en yakın komşuları ve en benzer "
                   "çiftler, bellek dostu parçalı yöntemle tam olarak hesaplanabilir; bu birkaç saniye ile bir "
                   "dakika sürebilir.")
        knn_anahtari = ozet(anahtar, max(int(cift_n), int(komsu_k)))
        sonuc = _bul("benzerlik_knn", knn_anahtari)
        if sonuc is None and st.button("En yakın komşuları hesapla", key="benz_knn", type="primary"):
            with st.spinner("En yakın komşular hesaplanıyor…"):
                sonuc = _hesapla("benzerlik_knn", knn_anahtari,
                                 lambda: bz.en_yakin_komsular(tfidf.matris, max(int(cift_n), int(komsu_k))), 1)
        if sonuc is None:
            return
        I, V = sonuc
        ciftler = bz.ciftler_komsulardan(I, V, adlar, int(cift_n))
        komsular = bz.komsu_tablosu(I[:, :int(komsu_k)], V[:, :int(komsu_k)], adlar)

    st.markdown("##### En benzer belge çiftleri")
    _tablo_goster(ciftler, 360)
    bg["ek_tablolar"]["Benzer çiftler"] = (lambda: ciftler, len(ciftler))
    bg["ek_tablolar"]["En yakın komşular"] = (lambda: komsular, len(komsular))

    st.markdown("##### Bir belgeye en benzer belgeler")
    secili = st.selectbox("Belge seçin", adlar, key="benz_belge")
    _tablo_goster(komsular[komsular["Belge"] == secili].drop(columns="Belge"), 240)

    if S is None:
        return

    kume = st.checkbox("Hiyerarşik kümeleme yap", False, key="benz_kume",
                       help="Belgeleri benzerliklerine göre ağaç biçiminde gruplar. Kümeler keşif amaçlıdır; "
                            "araştırmacının yorumlaması gerekir.")
    sira = None
    if kume and n < 3:
        st.warning("Kümeleme için en az 3 belge gerekir.")
    elif kume:
        c1, c2 = st.columns(2)
        yontem = c1.selectbox("Bağlantı yöntemi", list(bz.KUMELEME_YONTEMLERI),
                              format_func=bz.KUMELEME_YONTEMLERI.get, key="benz_yontem")
        k = int(c2.number_input("Küme sayısı", 2, max(2, min(30, n - 1)), min(4, n - 1), key="benz_k"))
        Z = _hesapla("benzerlik_kume", ozet(anahtar, yontem), lambda: bz.hiyerarsik_kumeleme(S, yontem), 1)
        etiket = bz.kume_etiketleri(Z, k)
        siluet = _hesapla("benzerlik_siluet", ozet(anahtar, yontem),
                          lambda: bz.siluet_tablosu(S, Z, yontem, 10), 1)
        kume_t = bz.kume_tablosu(etiket, adlar)
        bg["ek_tablolar"]["Kümeler"] = (lambda: kume_t, len(kume_t))
        bg["ek_tablolar"]["Siluet değerleri"] = (lambda: siluet, len(siluet))
        bg["ek_ayarlar"].update({"Kümeleme: yöntem": bz.KUMELEME_YONTEMLERI[yontem], "Kümeleme: küme sayısı": k})
        sol, sag = st.columns([3, 2])
        with sag:
            st.markdown("**Siluet değerleri**")
            _tablo_goster(siluet, 300)
            en_iyi = float(siluet["Ortalama siluet"].max()) if len(siluet) else 0
            if en_iyi < 0.25:
                st.caption("⚠️ Siluet değerleri düşük (< 0,25): belgeler belirgin kümeler oluşturmuyor; kümeleri "
                           "kesin gruplar gibi yorumlamayın.")
            else:
                st.caption("Siluet −1 ile 1 arasındadır; yüksek değer kümelerin birbirinden daha belirgin "
                           "ayrıldığını gösterir.")
        with sol:
            if n <= bz.DENDROGRAM_SINIRI:
                fig, sira = bz.dendrogram_figuru(Z, adlar, yontem)
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption(f"Dendrogram en fazla {bz.DENDROGRAM_SINIRI} belge için çizilir; küme üyelikleri "
                           "tablodadır.")
        st.markdown("**Küme üyelikleri**")
        _sinirli_tablo(kume_t)

    if n <= bz.ISI_HARITASI_SINIRI:
        st.plotly_chart(bz.isi_haritasi(S, adlar, list(reversed(sira)) if sira else None), width="stretch")
        if sira:
            st.caption("Isı haritası dendrogramdaki sıraya göre dizildi; benzer belgeler yan yana görünür.")
    else:
        st.caption(f"Isı haritası en fazla {bz.ISI_HARITASI_SINIRI} belge için çizilir; tablolar tüm belgeleri kapsar.")
    if n <= 300:
        matris = pd.DataFrame(np.round(S, 6), index=adlar, columns=adlar).reset_index(names="Belge")
        bg["ek_tablolar"]["Benzerlik matrisi"] = (lambda: matris, len(matris))


# ---------------------------------------------------------------------------
# Modül 2 — Sözcük birliktelik ağı
# ---------------------------------------------------------------------------

def _ag_sekmesi(bg: dict):
    st.subheader("Sözcük birliktelik ağı")
    st.info(ba.UYARI_METNI, icon="ℹ️")
    c1, c2, c3 = st.columns(3)
    baglam = c1.selectbox("Birlikte görülme bağlamı", list(ba.BAGLAM_TURLERI), format_func=ba.BAGLAM_TURLERI.get,
                          key="ag_baglam",
                          help="Pencere: iki sözcük arasında en fazla k sözcük varsa birlikte sayılır. Cümle/belge: "
                               "iki sözcüğün birlikte geçtiği cümle ya da belge sayısı.")
    pencere = c2.slider("Pencere genişliği (± sözcük)", 1, 15, 5, key="ag_pencere", disabled=baglam != "pencere",
                        help="Uzaklık özgün metne göre ölçülür; çıkarılan sözcükler (ve, ile…) de yer kaplar.")
    sinir = c3.checkbox("Pencere cümle sınırını aşmasın", True, key="ag_sinir", disabled=baglam != "pencere",
                        help="Cümle sınırları, Ön işleme > Diğer ayarlar > N-gram sınırları seçimine göre belirlenir.")
    c1, c2, c3, c4 = st.columns(4)
    olcut = c1.selectbox("Bağlantı ölçütü", list(ba.OLCUTLER), format_func=ba.OLCUTLER.get, key="ag_olcut",
                         help="Jaccard ve PPMI, çok sık sözcüklerin her şeyle bağlantılı görünmesini düzeltir. Gerçek "
                              "akademik özetlerde yazarların anahtar terimlerini en iyi Jaccard yakaladı.")
    min_b = int(c2.number_input("En az birlikte görülme", 1, 500, 3, key="ag_min",
                                help="Bu sayıdan az birlikte geçen çiftler alınmaz (PMI seyrek çiftleri abartır)."))
    dugum = int(c3.slider("Düğüm havuzu (en sık sözcük)", 10, ba.EN_FAZLA_DUGUM, 50, key="ag_dugum"))
    kenar = int(c4.slider("En fazla kenar", 10, ba.EN_FAZLA_KENAR, 100, key="ag_kenar"))
    with st.expander("Gelişmiş"):
        aday = int(st.number_input("Hesaplamaya alınan en sık sözcük sayısı", 100, ba.EN_FAZLA_ADAY, 1000, step=100,
                                   key="ag_aday", help="Birliktelikler yalnızca derlemin en sık bu kadar sözcüğü "
                                                       "arasında hesaplanır (bellek sınırı)."))
    ayar = ba.AgAyarlari(baglam=baglam, pencere=int(pencere), bolut_siniri=bool(sinir), aday_sozcuk=aday,
                         min_birlikte=min_b)
    with st.spinner("Birliktelikler hesaplanıyor…"):
        sonuc = _kapili_hesapla(bg, "ag", ozet(bg["on_isleme_anahtari"], repr(ayar)),
                                lambda: ba.birliktelik_hesapla(bg["belgeler"], ayar), "Ağı hesapla",
                                "birliktelik hesabı yaklaşık 5–15 saniye sürebilir.")
    if sonuc is None:
        return
    sec = ba.ag_sec(sonuc, olcut, dugum, kenar)
    dugumler = ba.dugum_tablosu(sec, sonuc)
    bg["ek_ayarlar"].update({f"Ağ: {k}": v for k, v in ayar.sozluk().items()})
    bg["ek_ayarlar"].update({"Ağ: ölçüt": ba.OLCUTLER[olcut], "Ağ: düğüm havuzu": dugum, "Ağ: en fazla kenar": kenar,
                             "Ağ: Jaccard birimi": "cümle/bölüt" if sonuc.jaccard_birimi == "bolut" else "belge"})
    if sec.empty:
        st.warning("Seçilen ayarlarla gösterilecek bağlantı yok. 'En az birlikte görülme' değerini düşürmeyi ya da "
                   "düğüm havuzunu büyütmeyi deneyin.")
        return
    st.plotly_chart(ba.ag_figuru(sec, dugumler, olcut), width="stretch")
    if not bg["ayarlar"].durak_sozcuk_akademik:
        st.caption("💡 Akademik metinlerde ağın merkezini 'elde edilmiştir', 'kullanılarak' gibi kalıp ifadeler "
                   "kaplayabilir. Ön işleme bölümünde 'Akademik kalıp sözcükleri de çıkar' seçeneğini deneyin.")
    st.caption(f"{len(dugumler)} düğüm, {len(sec)} kenar. Konumların ve uzaklıkların kendi başına anlamı yoktur; "
               "yerleşim her seferinde aynıdır (sabit tohum). Merkezîlik değerleri yalnızca gösterilen ağ içindir.")
    sol, sag = st.columns(2)
    with sol:
        st.markdown("**Düğümler**")
        _tablo_goster(dugumler, 380)
        st.caption("Derece: bağlantı sayısı. Normalleştirilmiş derece: derece ÷ (düğüm sayısı − 1). Ağırlıklı "
                   "derece: bağlantı ağırlıklarının toplamı (seçilen ölçüt).")
    with sag:
        st.markdown("**Kenarlar (gösterilen ağ)**")
        _tablo_goster(sec[["Sıra", "Kaynak", "Hedef", "Birlikte görülme", "Jaccard", "PPMI"]], 380)
    tum_sayi = int(min(np.triu(sonuc.C >= min_b, 1).sum(), ba.KENAR_DISA_AKTARIM_SINIRI))
    bg["ek_tablolar"]["Ağ düğümleri"] = (lambda: dugumler, len(dugumler))
    bg["ek_tablolar"]["Ağ kenarları"] = (lambda: sec.drop(columns="Ağırlık"), len(sec))
    bg["ek_tablolar"]["Birliktelik çiftleri"] = (
        lambda: ba.kenar_tablosu(sonuc, olcut=olcut, en_fazla=ba.KENAR_DISA_AKTARIM_SINIRI), tum_sayi)
    st.caption(f"İndir sekmesinde eşiği geçen tüm çiftler de bulunur (en güçlü {_sayi(ba.KENAR_DISA_AKTARIM_SINIRI)} "
               "çifte kadar). Kenar ve düğüm tabloları Gephi gibi ağ programlarına aktarılabilir.")


# ---------------------------------------------------------------------------
# Modül 3 — Konu modellemesi
# ---------------------------------------------------------------------------

def _konu_sekmesi(bg: dict):
    st.subheader("Konu modellemesi")
    st.info(km.UYARI_METNI, icon="ℹ️")
    belgeler = bg["belgeler"]
    toplam_sozcuk = bg["toplam_sozcuk"]
    c1, c2, c3 = st.columns(3)
    yontem = c1.radio("Yöntem", list(km.YONTEMLER), format_func=km.YONTEMLER.get, key="konu_yontem",
                      help="NMF, gerçek Türkçe akademik özetlerde daha tutarlı, daha kararlı ve çok daha hızlı sonuç "
                           "verdi. LDA olasılıksal bir modeldir ve yazında yaygındır; uzun belgelerde tercih edilebilir.")
    k = int(c2.slider("Konu sayısı", 2, km.EN_FAZLA_KONU, 5, key="konu_k",
                      help="Doğru konu sayısı veriden kendiliğinden çıkmaz; aşağıdaki karşılaştırmayı kullanın."))
    ust = int(c3.slider("Konu başına gösterilecek sözcük", 5, 25, 10, key="konu_ust"))
    with st.expander("Gelişmiş ayarlar"):
        g1, g2, g3 = st.columns(3)
        min_df = int(g1.number_input("Bir sözcük en az kaç belgede geçsin?", 1, 100, 2, key="konu_min_df"))
        max_df = float(g2.slider("Bir sözcük en fazla belgelerin yüzde kaçında geçsin?", 50, 100, 95, key="konu_max_df")) / 100
        sozluk = int(g3.number_input("En fazla sözcük sayısı (sözlük)", 500, 20000, 5000, step=500, key="konu_sozluk"))
        g1, g2, g3 = st.columns(3)
        tohum = int(g1.number_input("Rastgele başlangıç tohumu", 0, 10_000, 42, key="konu_tohum",
                                    help="Sabit tohum, aynı ayarlarla her seferinde aynı sonucu verir. Farklı bir tohumla "
                                         "konular belirgin biçimde değişiyorsa konular kararsızdır."))
        lda_iter = int(g2.number_input("LDA yineleme sayısı", 10, 200, 30, key="konu_lda_iter",
                                       disabled=yontem != "lda"))
        tekrar = int(g3.number_input("Kararlılık için alt örneklem sayısı", 0, 10, 3, key="konu_tekrar",
                                     help="Model, belgelerin rastgele %80'i üzerinde bu kadar kez yeniden kurulur ve "
                                          "konuların yeniden bulunup bulunmadığı ölçülür. 0: ölçme."))
    try:
        uyarilar = km.uygunluk_denetimi(belgeler, k)
        ayar = km.KonuAyarlari(yontem=yontem, konu_sayisi=k, en_az_belge=min_df, en_fazla_belge_orani=max_df,
                               en_fazla_sozcuk=sozluk, ust_sozcuk=ust, tohum=tohum, lda_yineleme=lda_iter)
    except (km.KonuHatasi, ValueError) as e:
        st.error(str(e))
        return
    for u in uyarilar:
        st.warning(u)
    ingilizce = _hesapla("ingilizce", bg["on_isleme_anahtari"], lambda: ingilizce_iceren(bg["metinler"]), 1)
    if ingilizce and not bg["ayarlar"].durak_sozcuk_ingilizce:
        st.warning(f"{len(ingilizce)} belge İngilizce metin içeriyor. İngilizce sözcükler ayrı bir 'konu' "
                   "oluşturabilir; Ön işleme bölümünde 'İngilizce durak sözcükleri de çıkar' seçeneğini açın.")
    if yontem == "lda" and toplam_sozcuk > km.LDA_SOZCUK_SINIRI:
        st.error(f"LDA bu büyüklükte bir derlem ({_sayi(toplam_sozcuk)} sözcük) için ücretsiz sunucuda çok uzun sürer "
                 f"(sınır: {_sayi(km.LDA_SOZCUK_SINIRI)} sözcük). NMF'yi seçin ya da uygulamayı kendi bilgisayarınızda "
                 "çalıştırın.")
        return
    tahmin = km.sure_tahmini(yontem, toplam_sozcuk, k, 1 + tekrar)
    bg["ek_ayarlar"].update({f"Konu: {a}": v for a, v in ayar.sozluk().items()})
    bg["ek_ayarlar"]["Konu: kararlılık alt örneklem sayısı"] = tekrar
    anahtar = ozet(bg["on_isleme_anahtari"], repr(ayar), tekrar)
    sonuc = _bul("konu", anahtar)
    if sonuc is None:
        st.caption(f"Tahmini süre: yaklaşık {_sure(tahmin)} (sunucu yoğunsa daha uzun).")
        if st.button("Konu modelini çalıştır", key="konu_calistir", type="primary"):
            with st.spinner("Konu modeli kuruluyor…"):
                try:
                    sonuc = _hesapla("konu", anahtar, lambda: km.konu_modeli(belgeler, ayar, tekrar), 2)
                except km.KonuHatasi as e:
                    st.error(str(e))
                    return
    if sonuc is not None:
        _konu_sonucu_goster(bg, sonuc, anahtar)
    _konu_karsilastirma(bg, ayar, toplam_sozcuk)


def _sure(sn: float) -> str:
    return f"{max(1, round(sn))} saniye" if sn < 90 else f"{sn / 60:.1f} dakika".replace(".", ",")


def _konu_sonucu_goster(bg: dict, sonuc, anahtar: str):
    ayar = sonuc.ayarlar
    o = sonuc.olcutler
    m = st.columns(4)
    m[0].metric("NPMI tutarlılığı", f"{o['Ortalama NPMI tutarlılığı']:.3f}".replace(".", ","),
                help="−1 … 1. Yüksek: konunun sözcükleri aynı belgelerde birlikte geçme eğiliminde.")
    m[1].metric("UMass tutarlılığı", f"{o['Ortalama UMass tutarlılığı']:.2f}".replace(".", ","),
                help="0'a yakın değerler daha tutarlı konulara işaret eder.")
    cesit = next(v for a, v in o.items() if a.startswith("Konu çeşitliliği"))
    m[2].metric("Konu çeşitliliği", f"{cesit:.2f}".replace(".", ","),
                help="Konuların ilk 25 sözcüğü içinde benzersiz sözcüklerin oranı. 1: konular sözcük paylaşmıyor.")
    kar = o.get("Ortalama kararlılık (AJ)")
    m[3].metric("Kararlılık", "—" if kar is None else f"{kar:.2f}".replace(".", ","),
                help="0 … 1. Model belgelerin %80'iyle yeniden kurulduğunda aynı konular ne ölçüde bulunuyor?")
    if kar is not None and kar < 0.5:
        st.warning("Konular kararsız (kararlılık < 0,5): belgelerin bir kısmı değiştiğinde büyük ölçüde farklı konular "
                   "ortaya çıkıyor. Konu sayısını değiştirmeyi, NMF'yi denemeyi ya da sonuçları çok dikkatli "
                   "yorumlamayı düşünün.")
    if o.get("NMF yakınsadı mı", "evet") != "evet":
        st.caption("⚠️ NMF yineleme sınırına ulaştı; sonuçlar tam yakınsamamış olabilir.")
    st.markdown("##### Konuları adlandırın")
    st.caption("Her konunun sözcüklerini ve aşağıdaki tipik belgelerini okuyarak konuya kendi adınızı verin. Adlar "
               "tablolara ve indirilen dosyalara yansır; boş bırakılan konular 'Konu N' olarak kalır.")
    kisa = anahtar[:10]
    sutunlar = st.columns(min(4, ayar.konu_sayisi))
    ozel = {}
    for i in range(ayar.konu_sayisi):
        ozel[i] = sutunlar[i % len(sutunlar)].text_input(f"Konu {i + 1}", "", key=f"konu_ad_{kisa}_{i}",
                                                          placeholder=" ".join(sonuc.ust_sozcukler(3)[i]))
    adlar = km.konu_adlari(sonuc, ozel)
    st.plotly_chart(km.konu_grafigi(sonuc, adlar), width="stretch")
    st.caption("LDA'da değerler olasılıktır; NMF'de negatif olmayan ağırlıklardır (olasılık değildir).")
    st.markdown("##### Her konunun en tipik belgeleri")
    for i in range(ayar.konu_sayisi):
        with st.expander(f"{i + 1}. {adlar[i]} — {', '.join(sonuc.ust_sozcukler(6)[i])}"):
            for j in km.tipik_belgeler(sonuc, i, 3):
                agirlik = sonuc.belge_konu[list(sonuc.belge_indisleri).index(j), i]
                metin = str(bg["metinler"][j])
                st.markdown(f"**{_md(bg['adlar'][j])}** · konu ağırlığı {agirlik:.2f}".replace(".", ",", 1))
                st.caption(metin[:400] + ("…" if len(metin) > 400 else ""))
    konu_olc = sonuc.konu_olcutleri.copy()
    konu_olc.insert(1, "Konu adı", adlar)
    st.markdown("##### Konu ölçütleri")
    _tablo_goster(konu_olc, 300)
    bk = km.belge_konu_tablosu(sonuc, bg["adlar"], adlar)
    st.markdown("##### Belge–konu dağılımı")
    _sinirli_tablo(bk)
    if o["Modeldışı kalan (boş) belge"]:
        st.caption(f"{o['Modeldışı kalan (boş) belge']} belge, sözlük süzgecinden sonra hiç sözcüğü kalmadığı için "
                   "modele girmedi.")
    ks = km.konu_sozcuk_tablosu(sonuc, adlar)
    ozet_t = sozluk_tablosu({**o, "Konu adları": "araştırmacı tarafından verilmiştir" if any(ozel.values())
                             else "verilmedi (Konu 1, Konu 2…)"}, "Ölçüt", "Değer")
    bg["ek_tablolar"].update({"Konu sözcükleri": (lambda: ks, len(ks)), "Belge-konu dağılımı": (lambda: bk, len(bk)),
                              "Konu ölçütleri": (lambda: konu_olc, len(konu_olc)),
                              "Konu modeli özeti": (lambda: ozet_t, len(ozet_t))})


def _konu_karsilastirma(bg: dict, ayar, toplam_sozcuk: int):
    from dataclasses import replace

    with st.expander("Farklı konu sayılarını karşılaştır"):
        st.caption("Her konu sayısı için model kurulur ve ölçütler yan yana gösterilir. Tek bir 'doğru' konu sayısı "
                   "yoktur: tutarlılığın ve kararlılığın yüksek, çeşitliliğin düşmediği aralıklara bakın ve konuları "
                   "okuyarak karar verin.")
        c1, c2, c3 = st.columns(3)
        aralik = c1.slider("Konu sayısı aralığı", 2, km.EN_FAZLA_KONU, (3, 12), key="konu_aralik")
        adim = int(c2.number_input("Adım", 1, 5, 1, key="konu_adim"))
        tekrar = int(c3.number_input("Kararlılık alt örneklemi", 0, 5, 2, key="konu_k_tekrar"))
        sayilar = list(range(aralik[0], aralik[1] + 1, adim))[:12]
        tahmin = sum(km.sure_tahmini(ayar.yontem, toplam_sozcuk, kk, 1 + tekrar) for kk in sayilar)
        st.caption(f"{len(sayilar)} model ({', '.join(map(str, sayilar))}); tahmini süre: yaklaşık {_sure(tahmin)}. "
                   "En fazla 12 konu sayısı karşılaştırılır.")
        anahtar = ozet(bg["on_isleme_anahtari"], repr(replace(ayar, konu_sayisi=2)), tuple(sayilar), tekrar)
        tablo = _bul("konu_karsilastirma", anahtar)
        if tablo is None and st.button("Karşılaştırmayı çalıştır", key="konu_karsilastir"):
            cubuk = st.progress(0.0, text="Modeller kuruluyor…")

            def ilerleme(oran, kk):
                cubuk.progress(min(1.0, oran), text=f"Konu sayısı {kk} kuruluyor…" if kk else "Tamamlandı")

            tablo = _hesapla("konu_karsilastirma", anahtar,
                             lambda: km.konu_sayisi_karsilastir(bg["belgeler"], ayar, sayilar, tekrar, ilerleme), 1)
        if tablo is not None:
            gosterim = tablo[[c for c in tablo.columns if c in ("Konu sayısı", "Ortalama NPMI tutarlılığı",
                                                                "Konu çeşitliliği (ilk 25 sözcük)",
                                                                "Ortalama kararlılık (AJ)")]]
            st.plotly_chart(km.karsilastirma_grafigi(gosterim), width="stretch")
            _tablo_goster(tablo, 360)
            bg["ek_tablolar"]["Konu sayısı karşılaştırması"] = (lambda: tablo, len(tablo))


# ---------------------------------------------------------------------------
# Modül 4 — Grup karşılaştırması
# ---------------------------------------------------------------------------

def _grup_sekmesi(bg: dict):
    st.subheader("Grup karşılaştırması")
    st.info(gr.UYARI_METNI, icon="ℹ️")
    tablo = bg["tablo"]
    adaylar = [c for c in tablo.columns if c not in (bg["metin_sutunu"],)]
    if not adaylar or (list(tablo.columns) == ["belge", "metin"] and bg["kaynak"] != "Örnek veri setini kullan"):
        st.info("Grup karşılaştırması için CSV/Excel dosyanızda metin sütununun yanında bir kategori sütunu (ör. bölüm, "
                "yıl, cinsiyet, okul türü) bulunmalıdır.")
        return
    oneri = gr.grup_sutunu_oner(tablo, haric=(bg["metin_sutunu"], bg["ad_sutunu"])) or adaylar[0]
    c1, c2, c3 = st.columns(3)
    sutun = c1.selectbox("Grupları belirleyen sütun", adaylar, index=adaylar.index(oneri), key="grup_sutunu")
    onsel = c2.selectbox("Önsel (yumuşatma)", list(gr.ONSELLER), format_func=gr.ONSELLER.get, key="grup_onsel",
                         help="Bilgilendirici önsel, seyrek sözcüklerin abartılı değerler almasını önler ve gerçek "
                              "akademik özetlerde rastgele gruplamada yanlış pozitif üretmedi (tutucu). Zayıf önsel "
                              "daha duyarlıdır ama yanlış pozitif oranı biraz yüksektir.")
    min_s = int(c3.number_input("Bir sözcük toplamda en az kaç kez geçsin?", 1, 1000, 5, key="grup_min"))
    c1, c2 = st.columns(2)
    perm = int(c1.number_input("Permütasyon sayısı", 0, 1000, 200, step=50, key="grup_perm",
                               help="Grup etiketleri belgeler arasında bu kadar kez rastgele karıştırılarak, z "
                                    "değerlerinin tesadüfen ne kadar yükselebileceği ölçülür. 0: hesaplama."))
    ilk = int(c2.slider("Grup başına gösterilecek sözcük", 5, 40, 15, key="grup_ilk"))
    grup_degerleri = [tablo[sutun].iloc[n - 1] for n in bg["satir_nolari"]]
    try:
        ayar = gr.GrupAyarlari(onsel=onsel, min_siklik=min_s, permutasyon=perm)
        anahtar = ozet(bg["on_isleme_anahtari"], sutun, repr(ayar))

        def hesapla():
            veri, bos = gr.grup_verisi_hazirla(bg["belgeler"], grup_degerleri)
            t, p = gr.ayirt_edici_sozcukler(veri, ayar)
            return veri, bos, t, p, gr.grup_ozeti(veri), gr.siklik_karsilastirma(veri, min_s)

        with st.spinner("Gruplar karşılaştırılıyor…"):
            sonuc = _kapili_hesapla(bg, "grup", anahtar, hesapla, "Grupları karşılaştır",
                                    "karşılaştırma ve permütasyonlar yaklaşık 10–30 saniye sürebilir.")
        if sonuc is None:
            return
        veri, bos, ayirt, perm_s, ozet_t, siklik = sonuc
    except (gr.GrupHatasi, ValueError) as e:
        st.error(str(e))
        return
    bg["ek_ayarlar"].update({"Grup: sütun": sutun, **{f"Grup: {a}": v for a, v in ayar.sozluk().items()},
                             "Grup: yöntem": "Monroe vd. (2008) Dirichlet önselli log-odds; her grup ↔ diğer gruplar"})
    if bos:
        nolar_ = ", ".join(str(bg["satir_nolari"][i]) for i in bos[:10]) + (" …" if len(bos) > 10 else "")
        st.warning(f"{len(bos)} belgenin grup değeri boş olduğu için karşılaştırmaya alınmadı (satır: {nolar_}).")
    for u in gr.yontemsel_uyarilar(ozet_t):
        st.warning(u)
    st.markdown("##### Gruplar")
    _tablo_goster(ozet_t, 260)
    st.markdown("##### Ayırt edici sözcükler")
    esik = perm_s["esik"] if perm_s else None
    st.plotly_chart(gr.ayirt_edici_grafik(ayirt, veri.gruplar, ilk, esik), width="stretch")
    if perm_s:
        st.caption(f"Koyu çubuklar ve kırmızı kesikli çizginin sağı: {perm_s['tekrar']} permütasyonla belirlenen, bu "
                   "veride rastgele gruplamada %95 olasılıkla aşılmayan eşiği geçen sözcükler. Açık renkli "
                   "çubuklar daha zayıf kanıttır. Rastgele gruplamada sözcüklerin ortalama "
                   f"%{perm_s['yanlis_pozitif_orani'] * 100:.1f}'i |z| > 1,96 verdi (kuramsal beklenti %5)."
                   .replace(".", ",", 1))
    else:
        st.caption("z > 1,96 gibi geleneksel eşikler bu yöntemde anlamlılığı abartabilir; permütasyonu açmanız önerilir.")
    secili = st.selectbox("Ayrıntılı tablo için grup", veri.gruplar, key="grup_secili")
    gt = ayirt[ayirt["Grup"] == secili].head(max(ilk, 50))
    _tablo_goster(gt.drop(columns="Grup"), 380)
    if (gt.head(ilk)["Tek belgeden gelen en büyük pay (%)"] > 50).any():
        st.caption("⚠️ Bazı sözcüklerin bu gruptaki kullanımının yarıdan fazlası tek bir belgeden geliyor (son sütun); "
                   "bunlar grubu değil o belgeyi yansıtıyor olabilir.")
    if len(veri.gruplar) == 2:
        st.plotly_chart(gr.iki_grup_sacilim(ayirt, veri.gruplar), width="stretch")
        st.caption("Her nokta bir sözcüktür. Yukarıdakiler birinci, aşağıdakiler ikinci grupta orantılı olarak daha "
                   "sık; sağdakiler derlemde daha sık. Renkli noktalar |z| > 1,96 (yalnızca yol gösterici).")
    with st.expander("Gruplara göre sıklık tablosu (ham ve bin sözcükte)"):
        _sinirli_tablo(siklik)
        st.caption("Gruplar farklı büyüklükte olduğunda ham sıklıklar yerine 'bin sözcükte' değerlerini karşılaştırın.")
    with st.expander("Yöntem nasıl çalışır?"):
        st.markdown(
            "Her grup, geri kalan tüm grupların birleşimiyle karşılaştırılır. Bir sözcüğün gruptaki ve diğer "
            "gruplardaki **log-odds** değerleri arasındaki fark (δ), yumuşatma için **Dirichlet önseli** eklenerek "
            "hesaplanır ve standart hatasına bölünerek **z** değeri elde edilir (Monroe, Colaresi & Quinn, 2008). "
            "Pozitif z: sözcük grupta daha sık; negatif z: daha seyrek.\n\n"
            "z, sözcükleri birbirinden bağımsız sayar; oysa aynı belgedeki sözcükler bağımlıdır ve binlerce sözcük "
            "aynı anda sınanır. Bu nedenle uygulama, grup etiketlerini **belgeler arasında** rastgele karıştırarak "
            "(permütasyon) bu veriye özgü bir eşik hesaplar. Gerçek akademik özetlerle yapılan sınamada bu eşik, "
            "anlamsız gruplamaların yaklaşık %5'inde (grup başına) aşıldı; yani beklendiği gibi çalıştı.")
    bg["ek_tablolar"].update({"Grup özeti": (lambda: ozet_t, len(ozet_t)),
                              "Ayırt edici sözcükler": (lambda: ayirt, len(ayirt)),
                              "Gruplara göre sıklık": (lambda: siklik, len(siklik))})
    if perm_s:
        pt = pd.DataFrame({"Grup": veri.gruplar, "Permütasyon eşiği (z, %95)": np.round(perm_s["esik"], 4)})
        bg["ek_tablolar"]["Permütasyon eşikleri"] = (lambda: pt, len(pt))


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


# ---------------------------------------------------------------------------
# 2.1 — Akademik terim sözlüğü
# ---------------------------------------------------------------------------
# Kullanıcının eklediği/yüklediği terimler ve devre dışı bıraktığı terimler YALNIZCA bu oturumun
# st.session_state alanında ("_tml_" önekiyle) tutulur; "Verilerimi temizle" bunları da siler.

T_KULLANICI = "_tml_kullanici_terimleri"
T_YUKLENEN = "_tml_yuklenen_sozluk"
T_KAPALI = "_tml_kapali_terimler"
T_MESAJ = "_tml_terim_mesaj"
T_SAYAC = "_tml_terim_sayac"
EN_FAZLA_ELLE_TERIM = 500

TERIM_DOGRULUK = """
**Nasıl çalışır?** Terimler özgün metinde aranır; özgün metin değiştirilmez. Terimin son sözcüğü dışındaki
sözcükler birebir eşleşmelidir; son sözcükte yalnızca **çekim ekleri** kabul edilir (uzaktan eğitim**e**, yapay
zekâ**nın**, risk faktör**leri**). Türetim ekleri eşleşmez (*uzaktan eğitimci*, *yapay zekâsız*) ve ayrı tabloda
gösterilir. En uzun terim önceliklidir (*üretken yapay zekâ* ⊃ *yapay zekâ*). Zincirleme tamlamada iyeliği
düşmüş biçimler (*makine öğrenme yöntemleri*) daha düşük güven düzeyiyle kabul edilir. "Güven düzeyi" kurala
dayalı sıralı bir değerdir, olasılık değildir.

**Ne kadar doğru?** (ayrıntı: `docs/bilimsel_degerlendirme_2.1.md`)

| Değerlendirme | Kesinlik | Duyarlılık |
|---|---|---|
| Gerçek akademik metin: 699 DergiPark özeti (mühendislik ağırlıklı), 652 terim geçişi, sözlük dizisinin doğru bulunması | %99,7 | %99,2 |
| Aynı veri, terimin **sözlükteki anlamıyla** geçmesi (6 alan seçili) | %90,1 | %99,1 |
| Aynı veri, yalnızca mühendislik + genel araştırma alanları seçili | %99,5 | %99,1 |
| Akademik olmayan genel Türkçe metin (400 cümle) | 0 yanlış alarm | — |

Gerçek veri etiketlemesini geliştirici yapmıştır (uzman etiketlemesi değildir); veri büyük ölçüde mühendislik
özetleridir, diğer alanlarda doğruluk farklı olabilir. Anlam hatalarının çoğu, eğitim alanındaki "derin
öğrenme" teriminin makine öğrenmesi anlamında geçmesinden kaynaklandı: **alanınızla ilgisi olmayan alanları
seçmeyin.**
"""


def _terim_kenar_cubugu() -> dict:
    with st.expander("📚 Akademik terim sözlüğü (isteğe bağlı)", expanded=bool(st.session_state.get("terim_acik"))):
        acik = st.checkbox(
            "Çok sözcüklü akademik terimleri tanı ve koru", False, key="terim_acik",
            help="637 aday terimlik sözlük. 'uzaktan eğitime', 'yapay zekânın' gibi ek almış biçimler standart terime "
                 "bağlanır ve analizlerde tek birim sayılır (uzaktan_eğitim). Varsayılan olarak kapalıdır.")
        if not acik:
            st.caption("Kapalı. Açınca ayrıntılar ve sözlük düzenleme *Terimler* sekmesinde.")
            return {"acik": False}
        tum = tm.alanlar()
        secili = st.multiselect("Alanlar", tum, default=tum, key="terim_alanlar",
                                help="Yalnızca çalışmanızla ilgili alanları seçin. Gerçek veride, ilgisiz alanların "
                                     "terimleri (ör. eğitimdeki 'derin öğrenme') başka anlamda eşleşebildi.")
        temsil = st.radio("Terimler analizlerde nasıl yer alsın?", list(tm.TEMSILLER), format_func=tm.TEMSILLER.get,
                          key="terim_temsil",
                          help="'Tek birim': sıklık, n-gram, TF-IDF, ağ, konu ve benzerlik analizlerinde terim tek "
                               "öğe olur. 'Yalnızca işaretle': analizler sözlük kapalıyken olduğu gibi kalır; "
                               "terimler yalnızca Terimler sekmesinde listelenir.")
        kisaltma = st.checkbox("Kısaltmaları da tanı (YEM, NLP, IoT…)", False, key="terim_kisaltma",
                               help="Büyük/küçük harfe duyarlıdır ('yem' eşleşmez). Tamamı büyük harfli başlık "
                                    "satırlarında kısaltma eşleştirilmez.")
        st.caption("⚠️ Sözlükteki terimler uzman doğrulaması bekleyen **aday terimlerdir**.")
    return {"acik": True, "alanlar": tuple(secili), "temsil": temsil, "kisaltma": bool(kisaltma)}


def _kullanici_terimleri() -> list:
    yuklenen = st.session_state.get(T_YUKLENEN) or {}
    return list(st.session_state.get(T_KULLANICI, [])) + list(yuklenen.get("terimler", []))


def _etkin_sozluk(ayar: dict) -> list:
    return tm.sozluk_olustur(ayar.get("alanlar", ()), st.session_state.get(T_KAPALI, []), _kullanici_terimleri())


def _terim_eslestirici(ayar: dict):
    if not ayar.get("acik"):
        return None
    sozluk = _etkin_sozluk(ayar)
    anahtar = ozet(tuple(repr(t) for t in sozluk), ayar["kisaltma"], ayar["temsil"])
    return _hesapla("terimci", anahtar, lambda: tm.TerimEslestirici(sozluk, ayar["kisaltma"], ayar["temsil"]), 2)


def _terim_ayar_sozlugu(ayar: dict, terimci) -> dict:
    if terimci is None:
        return {"Akademik terim sözlüğü": "kullanılmadı"}
    yuklenen = st.session_state.get(T_YUKLENEN) or {}
    return {
        "Akademik terim sözlüğü": "kullanıldı (aday terimler; uzman doğrulaması yok)",
        "Sözlük sürümü": tm.sozluk_surumu(),
        "Seçilen alanlar": "; ".join(ayar["alanlar"]) or "(yok)",
        "Terim temsili": tm.TEMSILLER[ayar["temsil"]],
        "Kısaltmalar tanındı mı": ayar["kisaltma"],
        "Elle eklenen terim sayısı": len(st.session_state.get(T_KULLANICI, [])),
        "Yüklenen kullanıcı sözlüğü": (f"{yuklenen.get('ad')} ({len(yuklenen.get('terimler', []))} terim)"
                                       if yuklenen else "yok"),
        "Devre dışı bırakılan terimler": ", ".join(st.session_state.get(T_KAPALI, [])) or "yok",
        "Etkin terim sayısı": len(terimci),
        "Terim eşleştirici parmak izi": terimci.parmak_izi[:16],
    }


def _terim_mesaj(tur: str, metin: str):
    st.session_state[T_MESAJ] = (tur, metin)


def _terim_ekle():
    try:
        liste = st.session_state.setdefault(T_KULLANICI, [])
        if len(liste) >= EN_FAZLA_ELLE_TERIM:
            raise tm.SozlukHatasi(f"En fazla {EN_FAZLA_ELLE_TERIM} terim elle eklenebilir; daha fazlası için sözlük "
                                  "dosyası yükleyin.")
        n = st.session_state.get(T_SAYAC, 0) + 1
        t = tm.kullanici_terimi(st.session_state.get("terim_yeni", ""), st.session_state.get("terim_yeni_var", ""),
                                terim_id=f"K{n:04d}")
        if any(tm.katla(x.standart) == tm.katla(t.standart) for x in liste):
            raise tm.SozlukHatasi(f"“{t.standart}” zaten eklenmiş.")
        liste.append(t)
        st.session_state[T_SAYAC] = n
        st.session_state["terim_yeni"] = ""
        st.session_state["terim_yeni_var"] = ""
        _terim_mesaj("success", f"“{t.standart}” eklendi (kimlik {t.terim_id}).")
    except tm.SozlukHatasi as e:
        _terim_mesaj("error", str(e))


def _terim_sil():
    sil = set(st.session_state.get("terim_sil_sec", []))
    st.session_state[T_KULLANICI] = [t for t in st.session_state.get(T_KULLANICI, []) if t.terim_id not in sil]
    st.session_state["terim_sil_sec"] = []
    _terim_mesaj("success", f"{len(sil)} terim silindi.")


def _terim_kapat():
    sec = st.session_state.get("terim_kapat_sec", [])
    kapali = list(dict.fromkeys(list(st.session_state.get(T_KAPALI, [])) + list(sec)))
    st.session_state[T_KAPALI] = kapali
    st.session_state["terim_kapat_sec"] = []
    _terim_mesaj("success", f"{len(sec)} terim devre dışı bırakıldı.")


def _terim_ac():
    ac = set(st.session_state.get("terim_ac_sec", []))
    st.session_state[T_KAPALI] = [t for t in st.session_state.get(T_KAPALI, []) if t not in ac]
    st.session_state["terim_ac_sec"] = []
    _terim_mesaj("success", f"{len(ac)} terim yeniden etkinleştirildi.")


def _terim_dosyasi_degisti():
    anahtar = f"terim_dosya_{st.session_state.get('yukleme_surumu', 0)}"
    dosya = st.session_state.get(anahtar)
    if dosya is None:
        st.session_state.pop(T_YUKLENEN, None)
        return
    try:
        okuma = dosya_oku(dosya.name, dosya.getvalue())
        terimler, uyarilar = tm.sozluk_tablosundan(okuma.tablo)
        st.session_state[T_YUKLENEN] = {"ad": dosya.name, "terimler": terimler,
                                        "uyarilar": list(okuma.uyarilar) + uyarilar}
        _terim_mesaj("success", f"{dosya.name}: {len(terimler)} terim yüklendi.")
    except (VeriOkumaHatasi, tm.SozlukHatasi) as e:
        st.session_state.pop(T_YUKLENEN, None)
        _terim_mesaj("error", f"Sözlük dosyası okunamadı: {e}")


def _terim_yonetimi(ayar: dict):
    st.markdown("##### Sözlüğü düzenle")
    st.caption("Değişiklikler yalnızca bu oturumda geçerlidir; kalıcı olarak saklamak için etkin sözlüğü indirip "
               "sonra yeniden yükleyebilirsiniz. Sizin terimleriniz çekirdek sözlüğe göre önceliklidir.")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Terim ekle**")
        st.text_input("Standart biçim", key="terim_yeni", placeholder="örnek: makine öğrenimi")
        st.text_input("Yazım varyantları (isteğe bağlı, ; ile ayırın)", key="terim_yeni_var",
                      placeholder="örnek: makine öğrenmesi; ML")
        st.button("➕ Ekle", on_click=_terim_ekle, key="terim_ekle")
        liste = st.session_state.get(T_KULLANICI, [])
        if liste:
            st.multiselect("Eklediğim terimlerden sil", [t.terim_id for t in liste], key="terim_sil_sec",
                           format_func=lambda i: next((t.standart for t in liste if t.terim_id == i), i))
            st.button("Seçilenleri sil", on_click=_terim_sil, key="terim_sil")
    with k2:
        st.markdown("**Kendi sözlüğünü yükle (CSV / Excel)**")
        st.file_uploader("İlk sütun ya da 'standart_terim' sütunu terimleri içermeli; 'yazim_varyantlari', 'alan', "
                         "'terim_id' sütunları isteğe bağlıdır.", type=["csv", "xlsx"],
                         key=f"terim_dosya_{st.session_state.get('yukleme_surumu', 0)}",
                         on_change=_terim_dosyasi_degisti)
        yuklenen = st.session_state.get(T_YUKLENEN)
        if yuklenen:
            st.caption(f"Yüklü: {_md(yuklenen['ad'])} — {len(yuklenen['terimler'])} terim.")
            for u in yuklenen["uyarilar"][:5]:
                st.caption("⚠️ " + _md(u))
    cekirdek = [t for t in tm.sozluk_olustur(ayar.get("alanlar") if ayar.get("acik") else None)]
    kapali = st.session_state.get(T_KAPALI, [])
    ad = {t.terim_id: f"{t.standart} ({t.alan})" for t in tm.cekirdek_sozluk()}
    st.markdown("**Çekirdek sözlükteki bir terimi devre dışı bırak**")
    c1, c2 = st.columns([3, 1])
    c1.multiselect("Terim seçin", [t.terim_id for t in cekirdek if t.terim_id not in kapali], key="terim_kapat_sec",
                   format_func=lambda i: ad.get(i, i), label_visibility="collapsed",
                   placeholder="Devre dışı bırakılacak terimleri yazarak seçin")
    c2.button("Devre dışı bırak", on_click=_terim_kapat, key="terim_kapat")
    if kapali:
        c1.multiselect(f"Devre dışı terimler ({len(kapali)}) — yeniden etkinleştirmek için seçin", kapali,
                       key="terim_ac_sec", format_func=lambda i: ad.get(i, i))
        c2.button("Etkinleştir", on_click=_terim_ac, key="terim_ac")


def _vurgulu_metin(metin: str, eslesmeler) -> str:
    parcalar, son = [], 0
    for e in eslesmeler:
        parcalar.append(_md(metin[son:e.bas]))
        parcalar.append(f"**{_md(metin[e.bas:e.bit])}** ⟨{_md(e.standart)}⟩")
        son = e.bit
    parcalar.append(_md(metin[son:]))
    return "".join(parcalar)


def _terim_sekmesi(bg: dict):
    terimci, ayar = bg["terimci"], bg["terim_ayar"]
    st.subheader("Akademik terim sözlüğü")
    mesaj = st.session_state.pop(T_MESAJ, None)
    if mesaj:
        (st.success if mesaj[0] == "success" else st.error)(mesaj[1])
    if terimci is None:
        st.info("Sözlük **kapalı**. Sol menüdeki *📚 Akademik terim sözlüğü* bölümünden açabilirsiniz. Açıldığında "
                "\"uzaktan eğitime\", \"yapay zekânın\", \"ölçme ve değerlendirmede\" gibi ek almış çok sözcüklü "
                "terimler standart biçimlerine bağlanır ve analizlerde tek kavram olarak sayılır.")
        st.caption(tm.ADAY_UYARISI.replace("**", ""))
        with st.expander("Yöntem ve ölçülen doğruluk"):
            st.markdown(TERIM_DOGRULUK)
        _terim_yonetimi(ayar)
        return
    belgeler, adlar, metinler = bg["belgeler"], bg["adlar"], bg["metinler"]
    anahtar = bg["on_isleme_anahtari"]
    st.caption(f"{tm.sozluk_surumu()} · etkin terim: {_sayi(len(terimci))} · temsil: "
               f"{tm.TEMSILLER[terimci.temsil]} · {tm.ADAY_UYARISI.replace('**', '')}")
    belirsiz = _hesapla("terim_belirsiz", ozet(terimci.parmak_izi), lambda: tm.alanlar_arasi_belirsiz(terimci.terimler), 1)
    ozet_t = _hesapla("terim_ozeti", anahtar, lambda: tm.terim_ozeti(belgeler, belirsiz), 1)
    toplam = int(ozet_t["Sıklık"].sum()) if len(ozet_t) else 0
    kapsam = sum(bool(b.terimler) for b in belgeler)
    k = st.columns(4)
    k[0].metric("Bulunan terim geçişi", _sayi(toplam))
    k[1].metric("Farklı terim", _sayi(len(ozet_t)))
    k[2].metric("Terim içeren belge", f"{_sayi(kapsam)} / {_sayi(len(belgeler))}")
    k[3].metric("Ekli geçiş oranı", f"%{(ozet_t['Ekli geçiş oranı'] * ozet_t['Sıklık']).sum() / toplam * 100:.0f}"
                if toplam else "—")
    if len(terimci) == 0:
        st.warning("Etkin sözlük boş: hiçbir alan seçilmemiş ve kullanıcı terimi yok.")
    elif not toplam:
        st.warning("Seçilen sözlükteki terimlerin hiçbiri metinlerde bulunamadı.")
    bel = ozet_t[ozet_t["Başka alanda daha uzun terimin parçası"] != ""] if len(ozet_t) else ozet_t
    if len(bel):
        st.warning("Şu terimler başka bir alandaki daha uzun bir terimin parçasıdır; anlamları alana göre değişebilir: "
                   + "; ".join(f"**{_md(r['Standart terim'])}** ({_md(r['Alan'])}, {r['Sıklık']} kez)"
                               for _, r in bel.head(6).iterrows())
                   + ". Örneğin gerçek mühendislik özetlerinde eğitim alanındaki “derin öğrenme” terimi hep makine "
                     "öğrenmesi anlamında geçti. Gerekirse alan seçimini daraltın ya da terimi devre dışı bırakın.")

    alt = st.tabs(["Bulunan terimler", "Metin üzerinde", "Eşleşme ayrıntıları", "Sözlük açık / kapalı",
                   "Sözlüğü düzenle", "Yöntem ve doğruluk"])
    with alt[0]:
        if len(ozet_t):
            st.plotly_chart(yatay_cubuk(ozet_t, "Standart terim", "Sıklık", "En sık geçen terimler", bg["ilk_n"]),
                            width="stretch")
            _sinirli_tablo(ozet_t)
            st.caption("'Ekli geçiş oranı': terimin metinde ek almış biçimde (ör. uzaktan eğitim**e**) geçme oranı.")
    with alt[1]:
        n = min(len(belgeler), ONIZLEME_BELGE_SINIRI)
        _tablo_goster(tm.temizlenmis_metinler(belgeler[:n], metinler[:n], adlar[:n], bg["satir_nolari"][:n]), 420)
        if len(belgeler) > n:
            st.caption(f"İlk {_sayi(n)} belge gösteriliyor; tamamı İndir sekmesinde (TXT, CSV, Excel).")
        iceren = [i for i, b in enumerate(belgeler) if b.terimler]
        if iceren:
            sec = st.selectbox("Terimleri metin üzerinde göster", iceren[:500], key="terim_belge",
                               format_func=lambda i: f"{adlar[i]} ({len(belgeler[i].terimler)} terim)")
            st.markdown(_vurgulu_metin(str(metinler[sec]), belgeler[sec].terimler))
            st.caption("Kalın: metindeki biçim; ⟨ ⟩: bağlandığı standart terim. Özgün metin değiştirilmez.")
    with alt[2]:
        es = _hesapla("terim_eslesme", anahtar, lambda: tm.eslesme_tablosu(belgeler, adlar, bg["satir_nolari"]), 1)
        st.markdown(f"**Eşleşmeler** ({_sayi(len(es))})")
        _sinirli_tablo(es)
        st.caption("Başlangıç/Bitiş: özgün metindeki karakter konumları (0'dan başlar). Güven düzeyi kurala dayalı "
                   "sıralı bir değerdir: 1 = sözlükteki biçim, 0,95 = varyant, 0,9 = kesme işaretli ek, 0,85 = ek, "
                   "0,75 = iyeliği düşmüş tamlama, 0,7 = kısaltma. Olasılık değildir.")
        ca = _hesapla("terim_cakisma", anahtar, lambda: tm.cakisma_tablosu(belgeler, adlar), 1)
        st.markdown(f"**Çakışmalar** ({_sayi(len(ca))}) — iç içe ya da kesişen adaylardan seçilmeyenler")
        _sinirli_tablo(ca)
        ey = _hesapla("terim_eslesmeyen", anahtar, lambda: tm.eslesmeyen_tablosu(belgeler, adlar), 1)
        st.markdown(f"**Standartlaştırılmayan benzer biçimler** ({_sayi(len(ey))}) — türetim eki, tanınmayan ek, "
                    "tireli bileşik ya da başlık satırı")
        _sinirli_tablo(ey)
    with alt[3]:
        _terim_karsilastirma(bg)
    with alt[4]:
        _terim_yonetimi(ayar)
        etkin = _etkin_sozluk(ayar)
        st.download_button("Etkin sözlüğü indir (.csv)", lambda: csv_bayt(tm.sozluk_disa_aktar(etkin), ";", True),
                           "etkin_terim_sozlugu.csv", "text/csv", key="terim_sozluk_indir", on_click="ignore",
                           help="Bu dosyayı düzenleyip daha sonra 'Kendi sözlüğünü yükle' ile geri yükleyebilirsiniz.")
    with alt[5]:
        st.markdown(TERIM_DOGRULUK)

    toplam_es = sum(len(b.terimler) for b in belgeler)
    bg["ek_tablolar"].update({
        "Terim özeti": (lambda: ozet_t, len(ozet_t)),
        "Terim eşleşmeleri": (lambda: tm.eslesme_tablosu(belgeler, adlar, bg["satir_nolari"]), toplam_es),
        "Terim çakışmaları": (lambda: tm.cakisma_tablosu(belgeler, adlar),
                              sum(len(b.terim_sonucu.cakismalar) for b in belgeler if b.terim_sonucu)),
        "Eşleşmeyen biçimler": (lambda: tm.eslesmeyen_tablosu(belgeler, adlar),
                                sum(len(b.terim_sonucu.eslesmeyenler) for b in belgeler if b.terim_sonucu)),
        "Etkin sözlük": (lambda: tm.sozluk_disa_aktar(terimci.terimler), len(terimci)),
    })


def _terim_karsilastirma(bg: dict):
    st.markdown("Aynı ön işleme ayarlarıyla, sözlük **kapalı** ve **açık** iken elde edilen sonuçlar yan yana "
                "gösterilir. Açık listedeki alt çizgili öğeler (ör. `uzaktan_eğitim`) korunmuş terimlerdir.")
    ayarlar, metinler, adlar = bg["ayarlar"], bg["metinler"], bg["adlar"]
    kapali_anahtar = ozet(bg["hazirlik_anahtari"], repr(ayarlar), None)

    def kapali_hesapla():
        return derlem_on_isle(metinler, ayarlar)

    kapali = _kapili_hesapla(bg, "terim_kapali", kapali_anahtar, kapali_hesapla, "Karşılaştırmayı hesapla",
                             "Karşılaştırma için metinler sözlüksüz olarak yeniden işlenir.")
    if kapali is None:
        return
    acik = bg["belgeler"]
    n = bg["ilk_n"]
    tf_ayar = bg["tf_ayar"]
    sekme = st.tabs(["Sözcük sıklığı", f"{bg['n_deger']}-gram", "TF-IDF", "Konu modeli (NMF)"])
    ciftler = [
        ("Sözcük sıklığı", lambda b: sozcuk_sikligi(b), "Sözcük", "Sıklık"),
        (f"{bg['n_deger']}-gram", lambda b: ngram_sikligi(b, bg["n_deger"], bg["min_siklik"]), "N-gram", "Sıklık"),
    ]
    for i, (ad, fn, sutun, deger) in enumerate(ciftler):
        with sekme[i]:
            ka = _hesapla(f"terim_kars_kapali_{i}", kapali_anahtar + ad, lambda: fn(kapali), 1)
            ac = _hesapla(f"terim_kars_acik_{i}", bg["on_isleme_anahtari"] + ad, lambda: fn(acik), 1)
            tablo, olcut = tm.ilk_n_karsilastir(ka, ac, sutun, deger, n)
            st.caption(f"İlk {n} listede ortak öğe: {olcut['ortak']} · Jaccard: {olcut['jaccard']:.2f} · açık listede "
                       f"korunmuş terim: {olcut['acik_listede_terim_birimi']}".replace(".", ","))
            _tablo_goster(tablo, 420)
    with sekme[2]:
        if bg["tfidf"] is None:
            st.warning(bg["tfidf_hata"] or "TF-IDF hesaplanamadı.")
        else:
            try:
                ka = _hesapla("terim_kars_tf", kapali_anahtar + repr(tf_ayar),
                              lambda: tfidf_hesapla(kapali, list(adlar), *tf_ayar), 1).derlem_tablosu
                tablo, olcut = tm.ilk_n_karsilastir(ka, bg["tfidf"].derlem_tablosu, "Terim", "Ortalama TF-IDF", n)
                st.caption(f"İlk {n} listede ortak öğe: {olcut['ortak']} · Jaccard: {olcut['jaccard']:.2f}"
                           .replace(".", ","))
                _tablo_goster(tablo, 420)
            except TfidfHatasi as e:
                st.warning(str(e))
    with sekme[3]:
        kk = st.number_input("Konu sayısı", 2, 30, 8, key="terim_konu_k")
        konu_anahtari = ozet(kapali_anahtar, bg["on_isleme_anahtari"], int(kk))
        sonuc = _bul("terim_konu", konu_anahtari)
        if st.button("NMF konu modelini iki durumda çalıştır", key="terim_konu_calistir"):
            def calistir():
                ayar_ = km.KonuAyarlari("nmf", int(kk))
                return {d: km.konu_modeli(b, ayar_) for d, b in (("kapalı", kapali), ("açık", acik))}
            try:
                with st.spinner("Konu modelleri hesaplanıyor…"):
                    sonuc = _hesapla("terim_konu", konu_anahtari, calistir, 1)
            except ValueError as e:
                st.warning(str(e))
        if sonuc:
            c = st.columns(2)
            for (d, m), col in zip(sonuc.items(), c):
                with col:
                    st.markdown(f"**Sözlük {d}**")
                    st.caption(f"NPMI tutarlılığı: {m.olcutler.get('Ortalama NPMI tutarlılığı')} · Çeşitlilik: "
                               f"{m.olcutler.get('Konu çeşitliliği (ilk 25 sözcük)')}".replace(".", ","))
                    for j, u in enumerate(m.ust_sozcukler(8), 1):
                        st.markdown(f"{j}. " + _md(", ".join(u)))
    with st.expander("Bu karşılaştırma nasıl yorumlanmalı?"):
        st.markdown(
            "- Sözlük, terimleri **okunabilir ve tutarlı** kılar: \"uzaktan eğitime\", \"uzaktan eğitimin\" tek öğede "
            "toplanır. Gerçek akademik özetlerde terim geçişlerinin sözlük kapalıyken yalnızca %59'u tek bir n-gramda "
            "toplanabildi; açıkken %99,8'i tek standart birim oldu.\n"
            "- Sözlük analizleri **kendiliğinden iyileştirmez.** 1.397 mühendislik özetinde LDA konu tutarlılığı arttı "
            "(NPMI 0,04 → 0,13), NMF tutarlılığı değişmedi (0,25), NMF kararlılığı (0,68 → 0,60) ve belge benzerliğinin "
            "yazar anahtar sözcükleriyle uyumu (AUC 0,871 → 0,851) biraz düştü. Nedeni: terim tek birim olunca "
            "\"yapay sinir ağı\" ile \"sinir ağı\" geçen belgeler ortak \"sinir\" sözcüğünü artık paylaşmaz.\n"
            "- Bu yüzden sonuçlarınızı her iki durumda da inceleyin ve hangisini raporladığınızı yönteminizde belirtin.")


try:
    ana()
except Exception as _hata:  # st.stop()/rerun BaseException olduğu için burada yakalanmaz
    _kayit = guvenli_hata_kaydi(_hata)
    st.error("Beklenmeyen bir hata oluştu. Verileriniz günlüğe yazılmadı. Sayfayı yenileyip tekrar deneyin; "
             "sorun sürerse ayarları varsayılana döndürün ya da verinizi daha küçük parçalara bölün.")
    st.caption(f"Teknik ayrıntı (metin içermez): {_kayit}")
