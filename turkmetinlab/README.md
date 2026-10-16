# 📚 TürkMetinLab

**Türkçe akademik metinler için ücretsiz, kod bilgisi gerektirmeyen metin madenciliği uygulaması.**
Sürüm 2.1.0 · Değişiklikler: [DEGISIKLIKLER.md](DEGISIKLIKLER.md) · Bilimsel değerlendirme:
[2.1](docs/bilimsel_degerlendirme_2.1.md) · [2.0](docs/bilimsel_degerlendirme_2.0.md) ·
[1.1](docs/bilimsel_degerlendirme_1.1.md)

TürkMetinLab; makale özetleri, görüşme dökümleri, açık uçlu anket yanıtları gibi Türkçe metinlerde
sözcük sıklığı, n-gram ve TF-IDF analizleri; belge benzerliği, sözcük birliktelik ağları, konu
modellemesi ve grup karşılaştırması yapar. İsteğe bağlı akademik terim sözlüğüyle "uzaktan eğitime",
"yapay zekânın" gibi ek almış çok sözcüklü terimleri tanır ve tek kavram olarak sayar. Sonuçları etkileşimli grafikler, kelime bulutu ve
indirilebilir Excel/CSV tabloları olarak sunar. Tamamen ücretsiz ve açık kaynak araçlarla
çalışır; ücretli hizmet, veritabanı veya abonelik gerektirmez.

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| Dosya yükleme | CSV, Excel (.xlsx) ve TXT. Türkçe karakter kodlaması (UTF-8, Windows-1254, ISO-8859-9) otomatik tanınır. |
| Türkçe ön işleme | Türkçeye uygun küçük harf dönüşümü (I→ı, İ→i), kesme işareti, durak sözcükler, sayılar, web adresleri |
| Lemmatizasyon (isteğe bağlı) | Sözlük tabanlı (simplemma); Türkçe altın standart verilerde ölçülen doğruluk %94,4–94,5; dönüşüm tablosu |
| **Akademik terim sözlüğü** (2.1) | İsteğe bağlı, 637 aday terim, 6 alan. Ek almış çok sözcüklü terimleri tanır (uzaktan eğitime → uzaktan eğitim), türetilmiş biçimleri ayırır, en uzun terimi seçer, terimi analizlerde tek birim yapar ya da yalnızca işaretler. Kendi terimlerinizi ekleyebilir, sözlük yükleyebilir, terimleri devre dışı bırakabilirsiniz. Sözlük açık/kapalı karşılaştırması |
| Veri denetimi | Boş/yer tutucu hücreler, yinelenen kimlik ve metinler, formüller, bozuk veya yanlış türde dosyalar, İngilizce içeren belgeler (2.0) bildirilir |
| Sözcük sıklığı | Ham sıklık, göreli sıklık (%), bin sözcükteki sıklık, belge sıklığı |
| N-gram | 2, 3 ve 4 sözcüklü öbekler; seçilebilir bölütleme (varsayılan: cümle sonu, virgül, ; ve : aşılmaz) |
| TF-IDF | Derlem geneli ve belge bazında en ayırt edici terimler, ısı haritası |
| **Belge benzerliği** (2.0) | TF-IDF kosinüs benzerliği, en benzer çiftler, belge sıralaması, ısı haritası, hiyerarşik kümeleme + dendrogram + siluet; büyük derlemlerde bellek dostu en yakın komşu |
| **Sözcük birliktelik ağı** (2.0) | ±k sözcük penceresi / cümle / belge bağlamı, cümle sınırı seçeneği, ham sıklık, Jaccard, PPMI, derece ve ağırlıklı derece, etkileşimli ağ, düğüm/kenar tabloları |
| **Konu modellemesi** (2.0) | NMF (önerilen) ve LDA, sabit tohum, NPMI/UMass tutarlılığı, konu çeşitliliği, alt örneklem kararlılığı, konu sayısı karşılaştırması, araştırmacının konu adlandırması, tipik belgeler |
| **Grup karşılaştırması** (2.0) | Kategorik sütuna göre ham ve bin sözcükte sıklıklar, Dirichlet önselli log-odds (Monroe vd., 2008), belge düzeyinde permütasyon eşiği, küçük/dengesiz grup uyarıları |
| Görselleştirme | Etkileşimli Plotly grafikleri, Türkçe karakterli kelime bulutu |
| Dışa aktarım | Tüm tablolar (yeni modüller dahil) + tüm yöntemsel ayarlar ve sözlük sürümü tek Excel dosyasında; ayrı CSV dosyaları; temizlenmiş metinler TXT/CSV/Excel |
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

### Yeni sürümü güvenle yükleme (canlı uygulamayı bozmadan)

Yeni bir sürümü doğrudan `main` dalına yüklemeyin. Canlı uygulama `main` dalından çalışır. Aşağıdaki
yolla yeni sürüm önce ayrı bir dalda test edilir ve ayrı bir deneme uygulamasında denenir; canlı
uygulama ancak siz onayladıktan sonra güncellenir.

1. **Yeni dala yükleyin.** GitHub'da deponuzu açın, ardından **Add file → Upload files**'a tıklayın.
   Yeni sürüm klasörünün içindeki her şeyi sürükleyin (Mac'te gizli dosyalar için **Cmd + Shift + .**).
   Sayfanın altında **"Create a new branch for this commit and start a pull request"** seçeneğini
   işaretleyin. Dal adı olarak yeni sürümün adını yazın (ör. `gelistirme-2.1`), **Propose changes**'a, sonra **Create pull
   request**'e tıklayın. **Henüz birleştirmeyin.**
2. **Testleri bekleyin.** Pull request sayfasında **Testler** kontrolleri (Python 3.12 ve 3.13) birkaç
   dakika içinde tamamlanır. İkisinde de yeşil ✓ görmelisiniz. Kırmızı ✗ varsa birleştirmeyin.
3. **Deneme uygulaması kurun.** <https://share.streamlit.io>'da **Create app → Deploy a public app from
   GitHub**'ı seçin. Aynı depoyu seçin, **Branch** alanına dal adını (ör. `gelistirme-2.1`), **Main file path** alanına
   `app.py` yazın. **App URL** alanına farklı bir ad verin (ör. `turkmetinlab-deneme`). **Advanced
   settings**'te Python **3.12**'yi seçip **Deploy**'a tıklayın. Canlı uygulamanız bu adımdan etkilenmez.
4. **Deneyin.** Deneme uygulamasında örnek veriyle ve kendi (anonimleştirilmiş) verinizle yeni
   sekmeleri deneyin.
5. **Onaylarsanız birleştirin.** Pull request sayfasında **Merge pull request → Confirm merge**'e
   tıklayın. Canlı uygulama birkaç dakika içinde yeni sürüme geçer. Ardından deneme uygulamasını
   share.streamlit.io'da uygulamanın **⋮** menüsünden **Delete** ile silebilirsiniz.
6. **Onaylamazsanız:** Pull request'i **Close pull request** ile kapatın. Canlı uygulama eski sürümde
   kalır.

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

### Akademik terim sözlüğü (2.1, isteğe bağlı)

1. Sol menüde **📚 Akademik terim sözlüğü** bölümünü açın ve **Çok sözcüklü akademik terimleri tanı ve
   koru** kutusunu işaretleyin.
2. **Yalnızca çalışmanızla ilgili alanları seçin.** Gerçek mühendislik özetlerinde, eğitim alanındaki
   "derin öğrenme" terimi hep makine öğrenmesi anlamında geçti; ilgisiz alanları kaldırmak bu tür anlam
   hatalarını belirgin biçimde azalttı (kesinlik %90 → %99,5).
3. **Temsili seçin:**
   - *Tek birim*: terim analizlerde `uzaktan_eğitim` olarak sayılır.
   - *Yalnızca işaretle*: analizler değişmez, terimler yalnızca listelenir.
4. **Terimler** sekmesinde şunları denetleyin:
   - bulunan terimler;
   - metin üzerindeki eşleşmeler;
   - çakışmalar ve standartlaştırılmayan biçimler (ör. *uzaktan eğitimci*).

   Aynı sekmede kendi terimlerinizi ekleyebilir, CSV/Excel sözlüğü yükleyebilir ya da bir terimi devre dışı
   bırakabilirsiniz.
5. **Sözlük açık / kapalı** sekmesinde iki durumun sonuçlarını karşılaştırın. Raporunuzda hangi temsili
   kullandığınızı belirtin; sözlük sürümü Excel'in *Ayarlar* sayfasına yazılır.

Sözlük dosyası biçimi: ilk sütun ya da `standart_terim` sütunu terimleri içerir. `yazim_varyantlari`
(birden çoksa `;` ile ayrılır), `alan` ve `terim_id` sütunları isteğe bağlıdır. Etkin sözlüğü *Terimler →
Sözlüğü düzenle* bölümünden indirip düzenleyerek yeniden yükleyebilirsiniz. Eklediğiniz terimler yalnızca
oturumunuzda tutulur.

### Dosya hazırlama ipuçları

- Excel'den CSV kaydederken **CSV UTF-8** biçimini seçin (Windows-1254 de desteklenir).
- Metin içinde virgül/noktalı virgül varsa hücre tırnak içinde olmalıdır (Excel bunu kendiliğinden yapar).
- Eski `.xls` dosyalarını Excel'de **Farklı Kaydet → Excel Çalışma Kitabı (.xlsx)** ile dönüştürün.
- En büyük dosya boyutu: 30 MB.

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

**2.0 modülleri.** Her yeni modül elle hesaplanmış örneklerle, bağımsız uygulamalarla (scikit-learn,
scipy, networkx, jmhessel/FightingWords) ve 1.417 gerçek DergiPark makale özetiyle sınanmıştır. Öne
çıkan bulgular:
- Başlıkla aramada makalenin kendi özeti %84,8 oranında ilk sırada geldi.
- Yazar anahtar terimlerini ayırt etmede en iyi ölçüt Jaccard oldu.
- Kısa akademik özetlerde NMF, LDA'dan daha tutarlı ve kararlı konular verdi.
- Permütasyon eşiği, anlamsız gruplamalarda hedeflenen %5 düzeyinde yanlış pozitif verdi.

Ayrıntı: `docs/bilimsel_degerlendirme_2.0.md`.

**2.1 terim tanıma.** Kurallar Türkçe çekim eklerini (ünlü uyumu ve kaynaştırma denetimiyle) kabul eder,
türetim eklerini reddeder; en uzun terim seçilir; özgün metin değişmez. Sonuçlar:
- **Gerçek akademik metin** (699 DergiPark özeti, eşleştirici etiketlemeden önce donduruldu): sözlük dizisini
  bulmada kesinlik %99,7, duyarlılık %99,2. Terimin sözlükteki anlamıyla geçmesi açısından kesinlik %90,1;
  ilgisiz alanlar kaldırılınca %99,5.
- **Genel Türkçe metin** (400 cümle): yanlış alarm yok.
- **Analizlere etkisi:** Sözlük terimleri tutarlı kılar: geçişlerin tek n-gramda toplanma oranı %59'dan
  %99,8'e çıktı. Ama analizleri kendiliğinden iyileştirmez: LDA tutarlılığı arttı, NMF kararlılığı ve
  benzerlik uyumu biraz düştü.

Ayrıntı: `docs/bilimsel_degerlendirme_2.1.md`.

### Sınırlılıklar

- Benzerlik konuyu ölçer; anlamsal eşdeğerlik, intihal ya da metin kökeni tespiti değildir.
- Birlikte görülme nedensellik ya da doğrudan anlamsal ilişki değildir.
- Konular algoritmiktir; araştırmacı okuyup adlandırmadıkça doğrulanmış nitel tema değildir.
- Grup farkları neden-sonuç açıklamaz.
- Gerçek veri sınaması ağırlıklı olarak mühendislik özetleriyle yapılmıştır.
- Varsayılan olarak sözcükler köklerine indirgenmez: "eğitim", "eğitimi" ve "eğitimde" ayrı sayılır.
  İsteğe bağlı lemmatizasyon bağlama bakmaz (~%94 doğruluk), eylem adlarını fiil köküne indirir
  (*çalışma → çalış*) ve kalıp terimleri değiştirebilir (*uzaktan eğitim → uzak eğitim*).
- Türkçe kurala göre "COVID" → "covıd" olur. Gerekirse *Büyük harfli kısaltmaları koru*
  seçeneğini kullanın (bu durumda büyük harfle yazılmış başlıklar da ayrı sayılır).
- Durak sözcük listesi genel amaçlıdır; alanınıza özgü sözcükleri arayüzden ekleyebilirsiniz.
- Örnek veri seti kurgusaldır; bilimsel kaynak olarak kullanılmamalıdır.
- **Terim sözlüğü (2.1):**
  - Sözlük uzman doğrulamasından geçmemiş aday terimlerden oluşur.
  - Anlam ayrımı yapılmaz: aynı dizi başka anlamda geçebilir.
  - Gerçek veri sınaması ağırlıklı olarak mühendislik özetleriyle yapılmıştır. Diğer alanlarda öğe sayısı
    azdır; etiketleme uzman etiketlemesi değildir.
  - Yazım hataları ve türetilmiş biçimler (*tip 2 diyabetli*) tanınmaz.
  - Terimleri tek birim yapmak belgeler arasındaki kısmi sözcük örtüşmesini azaltabilir.

---

## Geliştiriciler için

### Yerel çalıştırma (kendi bilgisayarınızda)

Python 3.12 veya 3.13 kurulu olmalıdır.

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Yerelde veri sınırlarını yükseltmek için (ör. 5 milyon sözcük, LDA için 3 milyon sözcük):

```bash
TML_EN_FAZLA_SOZCUK=5000000 TML_LDA_SOZCUK=3000000 streamlit run app.py
```

### Testler

```bash
python -m pytest -q
```

461 otomatik test (2.0'ın 314 testi ve 1.1'in 232 testi değiştirilmeden korunmuştur; 3 büyük veri testi isteğe bağlıdır). Kapsadıkları konular:

- Türkçe harf dönüşümü, kodlama tanıma, CSV/Excel/TXT okuma
- Bölütleme ve lemmatizasyon (UD Turkish-BOUN altın standart verisiyle)
- Gizlilik (paylaşılan önbellek, diske yazma, hata günlüğü)
- Dosya güvenliği
- TF-IDF'in bağımsız bir uygulama ve Türkçe referans veriyle karşılaştırılması
- Benzerlik, birliktelik ağı, konu modeli ve grup karşılaştırması: elle hesaplanmış örnekler, bağımsız
  uygulamalar, ekilmiş konular, permütasyon kalibrasyonu
- Terim tanıma (2.1): elle yazılmış çekim/türetim listeleri, konumların özgün metne geri eşlenmesi,
  iç içe ve kesişen terimler, kısaltmalar, kullanıcı sözlüğü, oturum yalıtımı, dondurulmuş yapay test
  kümesinde alt sınır
- Performans ve bellek
- Uygulamanın uçtan uca çalışması

Testler GitHub'a her yüklemede **Actions** sekmesinde Python 3.12 ve 3.13 ile otomatik çalışır.
Uzun süren büyük veri ölçümü için: `TML_BUYUK_TEST=1 python -m pytest tests/test_performans.py`.
Gerçek akademik veriyle validasyon (internet gerekir): `python degerlendirme/v2_dogrulama.py`,
`python degerlendirme/v21_terim_dogrulama.py hepsi`.

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
│   ├── benzerlik.py          # 2.0: belge benzerliği ve kümeleme
│   ├── birliktelik.py        # 2.0: sözcük birliktelik ağları
│   ├── konu.py               # 2.0: konu modellemesi (NMF, LDA)
│   ├── grup.py               # 2.0: grup karşılaştırması
│   ├── terimler.py           # 2.1: akademik terim sözlüğü ve çok sözcüklü terim tanıma
│   ├── sozluk/               # 2.1: çekirdek terim sözlüğü (637 aday terim) ve entegrasyon notları
│   ├── visualize.py          # Grafikler ve kelime bulutu
│   ├── export.py             # CSV / Excel dışa aktarım (formül enjeksiyonu koruması)
│   ├── oturum.py             # Oturuma özel bellek, veri temizliği, güvenli hata kaydı
│   └── performans.py         # Veri boyutu sınırları
├── tests/                    # Otomatik testler (pytest)
│   └── veri/                 # UD Turkish-BOUN örneği (CC BY-SA 4.0) ve beklenen sonuçlar
├── degerlendirme/            # Lemmatizasyon, bölütleme, performans, 2.0 ve 2.1 validasyon betikleri
│   └── veri/                 # 2.1: yapay terim test kümesi, DergiPark altın standardı (yalnızca konumlar)
├── docs/                     # Bilimsel değerlendirme, sözlük denetimi ve test raporları
├── ornek_veri/               # Örnek veri seti
├── .streamlit/config.toml    # Uygulama ayarları
├── requirements.txt          # Uygulama bağımlılıkları (sürümleri sabit)
└── requirements-dev.txt      # Test bağımlılıkları
```

### Kullanılan açık kaynak kütüphaneler

Streamlit, pandas, NumPy, SciPy, scikit-learn, NetworkX, Plotly, wordcloud, Matplotlib (DejaVu Sans fontu), openpyxl,
XlsxWriter, simplemma. Test verisi: UD Turkish-BOUN (CC BY-SA 4.0; atıf bilgisi `tests/veri/README.md`).
