# 📚 TürkMetinLab

**Türkçe akademik metinler için ücretsiz, kod bilgisi gerektirmeyen metin madenciliği uygulaması.**
Sürüm 1.0.0 — Aşama 1

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
| Sözcük sıklığı | Ham sıklık, göreli sıklık (%), bin sözcükteki sıklık, belge sıklığı |
| N-gram | 2, 3 ve 4 sözcüklü öbekler; cümle sınırlarını aşmaz |
| TF-IDF | Derlem geneli ve belge bazında en ayırt edici terimler, ısı haritası |
| Görselleştirme | Etkileşimli Plotly grafikleri, Türkçe karakterli kelime bulutu |
| Dışa aktarım | Tüm tablolar + kullanılan ayarlar tek Excel dosyasında; ayrı CSV dosyaları |
| Örnek veri | 16 kurgusal Türkçe akademik özet (yalnızca deneme amaçlıdır) |

## Gizlilik

- Yüklediğiniz dosyalar **yalnızca bellekte** işlenir; diske, veritabanına ya da GitHub'a kaydedilmez.
- Oturumu kapattığınızda veriler silinir.
- `.gitignore` dosyası, proje klasörüne yanlışlıkla kopyalanan veri dosyalarının GitHub'a
  gönderilmesini de engeller.

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
nokta, virgül, noktalı virgül, parantez ve satır sonu gibi sınırları aşmaz.

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

### Sınırlılıklar (Aşama 1)

- Sözcükler köklerine indirgenmez: "eğitim", "eğitimi" ve "eğitimde" ayrı sayılır.
- Türkçe kurala göre "COVID" → "covıd" olur. Gerekirse *Büyük harfli kısaltmaları koru*
  seçeneğini kullanın (bu durumda büyük harfle yazılmış başlıklar da ayrı sayılır).
- Durak sözcük listesi genel amaçlıdır; alanınıza özgü sözcükleri arayüzden ekleyebilirsiniz.
- Örnek veri seti kurgusaldır; bilimsel kaynak olarak kullanılmamalıdır.

---

## Geliştiriciler için

### Yerel çalıştırma

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

### Testler

```bash
python -m pytest -q
```

122 otomatik test; Türkçe harf dönüşümü, kodlama tanıma, CSV/Excel/TXT okuma, n-gram sınırları,
TF-IDF formül doğrulaması, dışa aktarım ve uygulamanın uçtan uca çalışmasını kapsar. Testler
GitHub'a her yüklemede **Actions** sekmesinde otomatik çalışır.

### Proje yapısı

```
turkmetinlab/
├── app.py                    # Streamlit arayüzü
├── tmlab/
│   ├── preprocess.py         # Türkçe ön işleme
│   ├── stopwords.py          # Durak sözcük listeleri
│   ├── loader.py             # CSV / Excel / TXT okuma
│   ├── analysis.py           # Sıklık, n-gram, TF-IDF
│   ├── visualize.py          # Grafikler ve kelime bulutu
│   └── export.py             # CSV / Excel dışa aktarım
├── tests/                    # Otomatik testler (pytest)
├── ornek_veri/               # Örnek veri seti
├── .streamlit/config.toml    # Uygulama ayarları
├── requirements.txt          # Uygulama bağımlılıkları (sürümleri sabit)
└── requirements-dev.txt      # Test bağımlılıkları
```

### Kullanılan açık kaynak kütüphaneler

Streamlit, pandas, NumPy, scikit-learn, Plotly, wordcloud, Matplotlib (DejaVu Sans fontu), openpyxl.
