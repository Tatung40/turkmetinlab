# 📚 TürkMetinLab

**Türkçe akademik metinler için ücretsiz, kod bilgisi gerektirmeyen metin madenciliği uygulaması.**
Sürüm 1.1.0 · Değişiklikler: [DEGISIKLIKLER.md](DEGISIKLIKLER.md) · Bilimsel değerlendirme:
[docs/bilimsel_degerlendirme_1.1.md](docs/bilimsel_degerlendirme_1.1.md)

TürkMetinLab; makale özetleri, görüşme dökümleri, açık uçlu anket yanıtları gibi Türkçe metinlerde
sözcük sıklığı, n-gram ve TF-IDF analizleri yapar. Sonuçları etkileşimli grafikler, kelime bulutu ve
indirilebilir Excel/CSV tabloları olarak sunar. Tamamen ücretsiz ve açık kaynak araçlarla
çalışır; ücretli hizmet, veritabanı veya abonelik gerektirmez.

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| Dosya yükleme | CSV, Excel (.xlsx) ve TXT. Türkçe karakter kodlaması (UTF-8, Windows-1254, ISO-8859-9) otomatik tanınır. |
| Türkçe ön işleme | Türkçeye uygun küçük harf dönüşümü (I→ı, İ→i), kesme işareti, durak sözcükler, sayılar, web adresleri |
| Lemmatizasyon (isteğe bağlı) | Sözlük tabanlı (simplemma); Türkçe altın standart verilerde ölçülen doğruluk %94,4–94,5; dönüşüm tablosu |
| Veri denetimi | Boş/yer tutucu hücreler, yinelenen kimlik ve metinler, formüller, bozuk veya yanlış türde dosyalar bildirilir |
| Sözcük sıklığı | Ham sıklık, göreli sıklık (%), bin sözcükteki sıklık, belge sıklığı |
| N-gram | 2, 3 ve 4 sözcüklü öbekler; seçilebilir bölütleme (varsayılan: cümle sonu, virgül, ; ve : aşılmaz) |
| TF-IDF | Derlem geneli ve belge bazında en ayırt edici terimler, ısı haritası |
| Görselleştirme | Etkileşimli Plotly grafikleri, Türkçe karakterli kelime bulutu |
| Dışa aktarım | Tüm tablolar + kullanılan ayarlar tek Excel dosyasında; ayrı CSV dosyaları |
| Örnek veri | 16 kurgusal Türkçe akademik özet (yalnızca deneme amaçlıdır) |

## Gizlilik

- Yüklenen dosyalar ve sonuçlar yalnızca **sizin oturumunuzun belleğinde** tutulur. Başka kullanıcılar
  erişemez; uygulama hiçbir veriyi diske, veritabanına ya da GitHub'a yazmaz.
- **🗑️ Verilerimi şimdi temizle** düğmesi her şeyi hemen siler. Sekmeyi kapatırsanız veriler yaklaşık
  2 dakika sonra sunucu belleğinden silinir.
- Hata durumunda sunucu günlüğüne metinleriniz yazılmaz.
- Çevrimiçi sürüm Streamlit Community Cloud (Snowflake) sunucularında çalışır. **Kişisel veya gizli
  veriler** (KVKK kapsamındaki bilgiler, görüşme dökümleri) için verileri anonimleştirin ya da
  uygulamayı kendi bilgisayarınızda çalıştırın (aşağıda *Yerel çalıştırma*).
- `.gitignore` dosyası, proje klasörüne yanlışlıkla kopyalanan veri dosyalarının GitHub'a
  gönderilmesini de engeller.

## Veri boyutu sınırları

| Veri boyutu | Ücretsiz bulut sürümünde ne olur? |
|---|---|
| 500 bin sözcüğe kadar | Normal çalışır (300 bin sözcük ~4 sn) |
| 500 bin – 1,5 milyon sözcük | Uyarı verilir; analiz bir dakikayı bulabilir |
| 1,5 milyon sözcük üzeri | Analiz yapılmaz (uygulamayı herkes için durdurabilir). Veriyi bölün ya da yerelde çalıştırın. |

Arayüzdeki tablolar ilk 5.000 satırı gösterir; tamamı *İndir* sekmesinden alınır. Excel'de sayfa başına
en çok 50.000 satır yazılır; daha uzun tabloların tamamı CSV ile indirilir. Yükleme sınırı 30 MB'tır.

---

## Yayımlama (kod bilgisi gerekmez)

Uygulamayı internette ücretsiz yayımlamak için iki ücretsiz hesap gerekir: **GitHub** (dosyaları
saklar) ve **Streamlit Community Cloud** (uygulamayı çalıştırır).

### Adım 1 — GitHub'a yükleme

1. <https://github.com/signup> adresinden ücretsiz bir hesap açın (hesabınız varsa giriş yapın).
2. Sağ üstteki **+** simgesine tıklayın → **New repository**.
3. *Repository name* alanına `turkmetinlab` yazın, **Public** seçili kalsın, **Create repository**'ye tıklayın.
4. Açılan sayfada **uploading an existing file** bağlantısına tıklayın.
5. Bilgisayarınızdaki `turkmetinlab` klasörünün **içindeki her şeyi** seçip tarayıcı penceresine
   sürükleyin (Chrome veya Edge önerilir; klasörler de sürüklenebilir).
   - **Mac kullanıyorsanız:** `.streamlit`, `.github` ve `.gitignore` gizli dosyalardır. Finder'da
     klasörü açıp **Cmd + Shift + .** (nokta) tuşlarına basarak görünür yapın, sonra hepsini seçin.
6. Yükleme bitince sayfanın altındaki **Commit changes** düğmesine tıklayın.

### Adım 2 — Streamlit Community Cloud'da yayımlama

1. <https://share.streamlit.io> adresine gidin ve **Continue with GitHub** ile giriş yapın;
   istenen izinleri onaylayın.
2. **Create app** → **Deploy a public app from GitHub** seçin.
3. Alanları doldurun:
   - **Repository:** `kullanıcı-adınız/turkmetinlab`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Advanced settings** → **Python version** olarak **3.12** seçin → **Save**.
5. **Deploy** düğmesine tıklayın. İlk kurulum birkaç dakika sürer.
6. Uygulamanın adresi (`https://....streamlit.app`) oluşur; bu bağlantıyı paylaşabilirsiniz.

> Uygulama bir süre kullanılmazsa uyku moduna geçer. Açtığınızda "wake up" düğmesine
> basmanız yeterlidir; yaklaşık bir dakikada yeniden açılır.

---

## Kullanım

1. **Veri seçin:** Sol menüden örnek veriyle başlayın ya da kendi dosyanızı yükleyin.
   - **CSV / Excel:** Her satır bir belgedir. Metin sütununu ve (isteğe bağlı) belge adı sütununu seçin.
   - **TXT:** Metni dosya, paragraf veya satır düzeyinde belgelere ayırabilirsiniz.
2. **Ön işleme ayarlarını** gözden geçirin. *Ön İşleme* sekmesinde her belgenin özgün ve işlenmiş
   halini yan yana görebilirsiniz.
3. **Sekmelerde** sonuçları inceleyin.
4. **İndir** sekmesinden sonuçları alın. Excel dosyasının *Ayarlar* sayfası, analizin hangi
   ayarlarla yapıldığını kaydeder (yöntem bölümünüzü yazarken ve tekrarlanabilirlik için).

### Dosya hazırlama ipuçları

- Excel'den CSV kaydederken **CSV UTF-8** biçimini seçin (Windows-1254 de desteklenir).
- Metin içinde virgül/noktalı virgül varsa hücre tırnak içinde olmalıdır (Excel bunu kendiliğinden yapar).
- Eski `.xls` dosyalarını Excel'de **Farklı Kaydet → Excel Çalışma Kitabı (.xlsx)** ile dönüştürün.
- En büyük dosya boyutu: 50 MB.

---

## Bilimsel notlar

**Türkçe küçük harf dönüşümü.** Python'un standart `lower()` işlevi Türkçede hatalıdır
(`"IŞIK".lower()` → `"işik"`). Uygulama `I → ı` ve `İ → i` dönüşümünü açıkça yapar.

**N-gramlar yapay olarak oluşturulmaz.** Durak sözcük, sayı gibi çıkarılan sözcüklerin yeri boş
bırakılır. Örneğin "eğitim **ve** öğretim" ifadesinden "eğitim öğretim" ikilisi üretilmez. N-gramlar
varsayılan olarak cümle sonu, virgül, noktalı virgül, iki nokta, parantez ve paragraf sınırlarını aşmaz
(PDF'ten kopyalanmış metinler nedeniyle tek satır sonu varsayılan olarak sınır değildir; seçilebilir).

**TF-IDF formülü** (scikit-learn `TfidfVectorizer`):

- `tf` = terimin belgedeki ham sayısı (isteğe bağlı: `1 + ln(tf)`)
- `idf = ln((1 + N) / (1 + df)) + 1` (yumuşatılmış; isteğe bağlı: `ln(N / df) + 1`)
- Her belge vektörü L2 normuyla (uzunluğu 1 olacak biçimde) normalleştirilir.
- *Ortalama TF-IDF*, terimin tüm belgelerdeki değerlerinin ortalamasıdır (geçmediği belgelerde 0).

Bu değerler testlerde, kütüphaneden bağımsız olarak elle yazılmış bir formülle karşılaştırılarak
doğrulanmıştır.

**Belirlenimcilik.** Hiçbir analiz rastgelelik içermez; aynı veri ve ayarlar her zaman aynı
sonucu verir. Kelime bulutu yerleşimi de sabit tohum değeriyle üretilir.

**Sıralama.** Eşit sıklıktaki sözcükler Türk alfabesine göre (a, b, c, ç, d, …, ı, i, …) sıralanır.

**N-gram sınırları.** UD Türkçe ağaç bankalarında noktalamasız yan yana sözcüklerin %60,2'si aynı
sözdizim öbeğindeyken bu oran virgülün iki yanında %3,4'tür. Bu nedenle varsayılan bölütleme virgülü
de sınır sayar. Ayrıntı: `docs/bilimsel_degerlendirme_1.1.md`.

### Sınırlılıklar

- Varsayılan olarak sözcükler köklerine indirgenmez: "eğitim", "eğitimi" ve "eğitimde" ayrı sayılır.
  İsteğe bağlı lemmatizasyon bağlama bakmaz (~%94 doğruluk), eylem adlarını fiil köküne indirir
  (*çalışma → çalış*) ve kalıp terimleri değiştirebilir (*uzaktan eğitim → uzak eğitim*).
- Türkçe kurala göre "COVID" → "covıd" olur. Gerekirse *Büyük harfli kısaltmaları koru*
  seçeneğini kullanın (bu durumda büyük harfle yazılmış başlıklar da ayrı sayılır).
- Durak sözcük listesi genel amaçlıdır; alanınıza özgü sözcükleri arayüzden ekleyebilirsiniz.
- Örnek veri seti kurgusaldır; bilimsel kaynak olarak kullanılmamalıdır.

---

## Geliştiriciler için

### Yerel çalıştırma (kendi bilgisayarınızda)

Python 3.12 veya 3.13 kurulu olmalıdır.

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Yerelde veri sınırını yükseltmek için (ör. 5 milyon sözcük):

```bash
TML_EN_FAZLA_SOZCUK=5000000 streamlit run app.py
```

### Testler

```bash
python -m pytest -q
```

231 otomatik test. Kapsadıkları konular:

- Türkçe harf dönüşümü, kodlama tanıma, CSV/Excel/TXT okuma
- Bölütleme ve lemmatizasyon (UD Turkish-BOUN altın standart verisiyle)
- Gizlilik (paylaşılan önbellek, diske yazma, hata günlüğü)
- Dosya güvenliği
- TF-IDF'in bağımsız bir uygulama ve Türkçe referans veriyle karşılaştırılması
- Performans ve bellek
- Uygulamanın uçtan uca çalışması

Testler GitHub'a her yüklemede **Actions** sekmesinde Python 3.12 ve 3.13 ile otomatik çalışır.
Uzun süren büyük veri ölçümü için: `TML_BUYUK_TEST=1 python -m pytest tests/test_performans.py`.

### Proje yapısı

```
turkmetinlab/
├── app.py                    # Streamlit arayüzü
├── tmlab/
│   ├── preprocess.py         # Türkçe ön işleme ve bölütleme
│   ├── lemma.py              # İsteğe bağlı lemmatizasyon
│   ├── stopwords.py          # Durak sözcük listeleri
│   ├── loader.py             # CSV / Excel / TXT okuma ve dosya güvenliği
│   ├── kalite.py             # Veri kalitesi denetimi
│   ├── analysis.py           # Sıklık, n-gram, TF-IDF
│   ├── visualize.py          # Grafikler ve kelime bulutu
│   ├── export.py             # CSV / Excel dışa aktarım (formül enjeksiyonu koruması)
│   ├── oturum.py             # Oturuma özel bellek, veri temizliği, güvenli hata kaydı
│   └── performans.py         # Veri boyutu sınırları
├── tests/                    # Otomatik testler (pytest)
│   └── veri/                 # UD Turkish-BOUN örneği (CC BY-SA 4.0) ve beklenen sonuçlar
├── degerlendirme/            # Lemmatizasyon, bölütleme ve performans ölçüm betikleri
├── docs/                     # Bilimsel değerlendirme raporu
├── ornek_veri/               # Örnek veri seti
├── .streamlit/config.toml    # Uygulama ayarları
├── requirements.txt          # Uygulama bağımlılıkları (sürümleri sabit)
└── requirements-dev.txt      # Test bağımlılıkları
```

### Kullanılan açık kaynak kütüphaneler

Streamlit, pandas, NumPy, scikit-learn, Plotly, wordcloud, Matplotlib (DejaVu Sans fontu), openpyxl,
XlsxWriter, simplemma. Test verisi: UD Turkish-BOUN (CC BY-SA 4.0; atıf bilgisi `tests/veri/README.md`).
