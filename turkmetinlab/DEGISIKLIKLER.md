# Değişiklikler

## 3.0.0rc1 — Nitel analiz pilotu ve bilimsel standartlaştırma (sürüm adayı)

Ayrıntılı kararlar: `docs/dongu_1_envanter.md` … `docs/dongu_4_envanter.md`, yayın kapıları:
[`docs/yayin_kontrol_listesi.md`](docs/yayin_kontrol_listesi.md).

- **Kütüphaneye taşınan hesaplar:**
  - TF-IDF benzerlik iç çarpımı → scikit-learn `linear_kernel`.
  - Ağ derece/strength → NetworkX.
  - NMF yakınsama → `n_iter_`.
  - ELECTRA kırpma → `tokenizers`; softmax → SciPy.
  - Eski hesaplar testlerde dondurulmuş oracle olarak tutulur; eşdeğerlik toleranslı sınanır.
- **Düzeltmeler:**
  - ELECTRA uzun metinlerde kapanış `[SEP]` belirtecini düşürüyordu; kırpılan metinlerde tahmin değişebilir.
  - Nitel ekranda YZ hiyerarşi önerisi onaysız olarak onaylı kayıtlara yazılıyordu; artık onay kuyruğundan geçer.
  - Qwen pilot betiği belgelendiği gibi çalıştırılınca `tmlab` bulunamıyordu.
- **Raporlama:**
  - Nitel Excel raporuna yöntem/sınırlılık sayfası, sürüm ve oluşturma zamanı eklendi.
  - LDA perplexity "örneklem içi" olarak adlandırıldı.
  - NPMI/UMass ve deneysel modüller arayüzde sınırlılıklarıyla etiketlendi.
- **Açık:** Gerçek Qwen3.5 kabul testi ile ELECTRA sürüm/lisans doğrulaması yapılmadı (bkz. kontrol listesi).

## 2.1.0 — Akademik terim sözlüğü ve çok sözcüklü terim tanıma

Ayrıntılı validasyon ve performans raporu: [`docs/bilimsel_degerlendirme_2.1.md`](docs/bilimsel_degerlendirme_2.1.md).
Sözlük denetimi: [`docs/sozluk_denetimi_2.1.md`](docs/sozluk_denetimi_2.1.md). 2.0'ın bütün özellikleri ve 314 testi
(2'si isteğe bağlı) değiştirilmeden korunmuştur.

### Yeni: akademik terim sözlüğü (isteğe bağlı, varsayılan olarak kapalı)
- **Sözlük:** 637 aday terimlik çekirdek sözlük, 6 alan ayrı ayrı seçilebilir. Terimler uzman doğrulaması
  bekleyen **aday terimler** olarak sunulur.
- **Ek tanıma:** "uzaktan eğitime → uzaktan eğitim", "yapay zekânın → yapay zekâ", "ölçme ve değerlendirmede",
  "yapısal eşitlik modellemesinin", "madde tepki kuramına" gibi ek almış biçimler tanınır.
  - Ünlü uyumu ve kaynaştırma denetlenir; ünsüz yumuşaması ve çoğul iyelik biçimleri (kuramları) kabul edilir.
  - Zincirleme tamlamada iyelik düşmesi (makine öğrenme yöntemleri) daha düşük güvenle kabul edilir.
  - Kesme işareti, tire ve şapka (â/a) yazım farkları tanınır.
- **Yanlış pozitif önlemleri:**
  - Türetim ekli biçimler (eğitimci, zekâsız, değerlendirmeci) eşleşmez ve ayrı tabloda gösterilir.
  - Noktalama terimi böler; tireli bileşiğin parçası eşleşmez (log-normal dağılım).
  - Kısaltmalar isteğe bağlıdır, büyük/küçük harfe duyarlıdır ve başlık satırlarında reddedilir.
- **Çakışmalar:** En uzun terim önceliklidir. Ardından kullanıcı terimi > alan sözlüğü > genel liste.
  İç içe ve kesişen adaylar çakışma tablosuna yazılır.
- **Özgün metin değişmez:** Her eşleşme terim kimliği, alan, metindeki biçim, standart terim, eşleşme türü,
  güven düzeyi ve özgün metindeki karakter konumlarıyla saklanır.
- **Temsil:**
  - "Tek birim" (`uzaktan_eğitim`): sıklık, n-gram, TF-IDF, ağ, konu ve benzerlik analizlerinde tek öğe; lemma
    ve durak sözcük süzgeçlerinden geçmez.
  - "Yalnızca işaretle": analizler değişmez.
- **Kullanıcı sözlüğü:** terim ekleme ve silme, çekirdek terimi devre dışı bırakma, CSV/Excel sözlük
  yükleme, etkin sözlüğü indirme. Hepsi yalnızca oturumda tutulur; "Verilerimi temizle" bunları da siler.
- **Yeni "📚 Terimler" sekmesi:**
  - bulunan terimler, metin üzerinde vurgulama;
  - özgün metin, temizlenmiş metin ve normalize terimler yan yana;
  - eşleşme, çakışma ve eşleşmeyen biçim tabloları;
  - alanlar arası belirsiz terim uyarısı;
  - sözlük açık/kapalı karşılaştırması (sıklık, n-gram, TF-IDF, NMF);
  - yöntem ve ölçülen doğruluk.

### Dışa aktarım
- Temizlenmiş metinler TXT olarak (yeni) ve CSV/Excel olarak ("Normalize terimler" sütunuyla) indirilebilir.
- Excel'e yeni sayfalar: Terim özeti, Terim eşleşmeleri, Terim çakışmaları, Eşleşmeyen biçimler, Etkin sözlük.
- *Ayarlar* sayfasına eklenenler:
  - sözlük sürümü (SHA-256) ve seçilen alanlar;
  - temsil ve kısaltma seçenekleri;
  - eklenen, yüklenen ve devre dışı bırakılan terimler;
  - eşleştiricinin parmak izi.

### Validasyon (özet)
- **Gerçek akademik metin** (699 DergiPark özeti, test yarısı, eşleştirici etiketlemeden önce donduruldu):
  - Sözlük dizisinin bulunması: kesinlik %99,7, duyarlılık %99,2.
  - Terimin sözlükteki anlamıyla geçmesi: kesinlik %90,1. Alan seçimiyle %99,5.
- Genel Türkçe metinde (400 cümle) yanlış alarm yok.
- Sözlük analizleri kendiliğinden iyileştirmedi: LDA tutarlılığı arttı; NMF kararlılığı ve benzerlik AUC
  değeri biraz düştü.

### Testler
- Yeni: `test_terimler.py` (135), `test_terimler_arayuz.py` (10), performans testine 2 ek (1'i isteğe bağlı).
- Eşleştirici yazılmadan önce dondurulmuş yapay test kümesi: `degerlendirme/veri/terim_test_yapay.tsv`.
- DergiPark altın standardı (yalnızca konum ve etiket): `degerlendirme/veri/terim_altin_dergipark.tsv`.
- Validasyon betiği: `degerlendirme/v21_terim_dogrulama.py`.

### Düzeltmeler ve notlar
- DergiPark verisindeki kimlik sütunu benzersiz değil (414 kimlik 929 özette tekrar ediyor) ve 20 özet
  yineleniyor. 2.1 değerlendirmesi satır sırasını kimlik olarak kullanır ve yinelenen metinleri çıkarır. 2.0
  benzerlik sonuçları yinelenenler çıkarılarak yeniden hesaplandı; fark önemsiz çıktı (AUC 0,865 → 0,869).
- `.gitignore`: sözlük CSV'si, kullanıcı verisini dışlayan `*.csv` kuralının istisnasına alındı.

### Bağımlılıklar
- Yeni bağımlılık yok.

## 2.0.0 — Analiz modülleri

Ayrıntılı validasyon ve performans raporu: [`docs/bilimsel_degerlendirme_2.0.md`](docs/bilimsel_degerlendirme_2.0.md).
1.1'in bütün özellikleri ve 232 testi (1'i isteğe bağlı) değiştirilmeden korunmuştur.

### Yeni modüller (yeni sekmeler)
- **🧭 Benzerlik:**
  - TF-IDF kosinüs benzerliği, en benzer belge çiftleri, seçilen belgeye en benzer belgeler, ısı haritası.
  - İsteğe bağlı hiyerarşik kümeleme: ortalama bağlantı veya Ward; dendrogram ve siluet değerleri.
  - 2.000 belgeden büyük derlemlerde bellek dostu, sonucu tam matrisle aynı en yakın komşu hesabı.
- **🕸️ Sözcük Ağı:**
  - Üç bağlam türü: ±k sözcük penceresi (cümle sınırına uyma seçeneğiyle), aynı cümle, aynı belge.
  - Ölçütler: ham sıklık, Jaccard (varsayılan, gerçek veride en iyisi), PPMI.
  - Derece ve ağırlıklı derece; etkileşimli ağ; Gephi'ye aktarılabilen düğüm ve kenar tabloları;
    düğüm ve kenar sınırları.
- **🧩 Konular:**
  - NMF (varsayılan; gerçek akademik özetlerde daha tutarlı, kararlı ve hızlı) ve LDA; sabit `random_state`.
  - Ölçütler: NPMI ve UMass tutarlılığı, konu çeşitliliği, alt örneklem kararlılığı.
  - Farklı konu sayılarının karşılaştırılması.
  - Küçük derlem, kısa belge ve yüksek konu sayısı uyarıları.
  - Araştırmacının konu adlandırması (indirilen dosyalara yansır) ve her konunun tipik belgeleri.
  - LDA'ya derlem büyüklüğü sınırı ve tahmini süre gösterimi.
- **👥 Gruplar:**
  - Kategorik sütuna göre grup özeti; ham ve bin sözcükte sıklıklar.
  - Dirichlet önselli log-odds (Monroe vd., 2008) ve belge düzeyinde permütasyon eşiği.
  - Belge oranı ve tek belge baskınlığı göstergesi.
  - Küçük grup, dengesiz grup, kimlik sütunu ve boş grup uyarıları.
  - İki grupta saçılım grafiği.

### Veri kalitesi ve ön işleme
- İngilizce metin içeren belgeler satır numarasıyla bildiriliyor.
- Yeni "İngilizce durak sözcükleri de çıkar" seçeneği.
- Akademik kalıp sözcük listesine gerçek veride sık görülen 9 kalıp fiil eklendi (seçenek varsayılan olarak kapalı).

### Dışa aktarım
- Yeni modüllerin bütün tabloları Excel'e ve CSV'ye ekleniyor.
- Tüm yöntemsel ayarlar (bağlam, ölçüt, önsel, tohum, permütasyon sayısı…) Excel'deki *Ayarlar*
  sayfasına yazılıyor.

### Performans ve güvenlik
- Uzun analizler düğmeyle başlıyor; 300 bin sözcüğün üstündeki derlemlerde ağ ve grup analizleri de.
  Sonuçlar yalnızca kullanıcının oturumunda saklanıyor; paylaşılan önbellek ve disk kullanılmıyor.
- Eşzamanlı 4 kullanıcıyla oturum yalıtımı, bellek ve süre sınandı.

### Testler ve validasyon
- Yeni testler: `test_benzerlik.py`, `test_birliktelik.py`, `test_konu.py`, `test_grup.py`; performans
  ve kalite testlerine ekler.
- Gerçek akademik veriyle (1.417 DergiPark özeti) yöntemsel validasyon betiği: `degerlendirme/v2_dogrulama.py`.

### Yeni bağımlılıklar
- `scipy==1.18.1` (önceden scikit-learn ile dolaylı olarak geliyordu; artık doğrudan kullanılıyor).
- `networkx==3.6.1`: ağ yerleşimi; BSD lisanslı.

## 1.1.0 — Bilimsel ve teknik iyileştirme

Ayrıntılı gerekçe ve ölçümler: [`docs/bilimsel_degerlendirme_1.1.md`](docs/bilimsel_degerlendirme_1.1.md)

### Gizlilik
- Yüklenen veriler ve sonuçlar artık tüm kullanıcılarca paylaşılan `st.cache_data` önbelleğinde
  değil, **yalnızca ilgili oturumun belleğinde** tutuluyor.
- Yeni **"Verilerimi şimdi temizle"** düğmesi: yüklenen dosyaları ve sonuçları hemen siler.
- Excel dosyaları tamamen bellekte oluşturuluyor. 1.0'da xlsxwriter geçici dosyayı diske yazıyordu.
- Beklenmeyen hatalarda sunucu günlüğüne kullanıcı metni yazılmıyor.
- Gizlilik açıklaması gerçek davranışa göre yeniden yazıldı; bulut altyapısı ve kişisel veriler
  için öneriler eklendi.

### Türkçe ön işleme
- **İsteğe bağlı lemmatizasyon** (simplemma). Türkçe altın standart verilerde %94,4–94,5 doğruluk
  ölçüldü. Anlamı tersine çeviren -sız/-siz hatalarına karşı koruma kuralı var. Dönüşüm tablosu
  gösteriliyor ve indirilebiliyor.
- **N-gram bölütleme seçenekleri:** Katı (varsayılan), Orta ve Yalnızca cümle sonu. Seçim UD
  ağaç bankası ölçümüne dayanıyor.
- "19. yüzyıl", "Prof. Dr.", "vb." gibi noktalar artık cümle sonu sayılmıyor. Saat biçimi
  (10:30) tek sayı olarak işleniyor.
- Tek satır sonu varsayılan olarak sınır değil, paragraf her zaman sınır; seçenek olarak
  değiştirilebiliyor.
- "vb, vs, vd, bkz, bk, örn, krş" durak sözcük listesine eklendi.

### Dosya güvenliği ve veri kalitesi
- Uzantısı değiştirilmiş dosyalar (PDF, görsel, Word, eski .xls), ikili veri ve sıkıştırma
  bombası reddediliyor.
- Excel formülleri çalıştırılmıyor; formül içeren hücreler bildiriliyor.
- Dışa aktarımda formül enjeksiyonu koruması:
  - Excel'de hiçbir hücre formül veya bağlantıya dönüşmüyor.
  - CSV'de "=, +, -, @" ile başlayan hücrelere ' ekleniyor (isteğe bağlı).
- Boş hücre, "NA/null/#YOK" ve Excel hata değerleri satır numarasıyla bildiriliyor. Excel'de "NA"
  artık sessizce boşa çevrilmiyor.
- Yinelenen belge kimlikleri ve yinelenen metinler bildiriliyor. İstenirse yinelenen metinler bir
  kez sayılıyor.
- Kapanmamış tırnak ve eksik sütunlu satır uyarıları eklendi.
- "İlk satır başlık" seçeneği eklendi.

### Performans
- Büyük derlemde (2,9 milyon sözcük) süre 87 sn'den 36 sn'ye, tepe bellek 1,5 GB'tan 0,8 GB'a indi.
- 500 bin sözcük üzerinde uyarı veriliyor, 1,5 milyon sözcük üzerinde bulutta analiz yapılmıyor.
  Yerel kurulumda sınır ortam değişkeniyle değiştirilebiliyor.
- Excel ve CSV dosyaları yalnızca indirme düğmesine basılınca üretiliyor. Excel'de sayfa başına
  50.000 satır sınırı var; tablonun tamamı CSV ile alınabiliyor.
- Yükleme sınırı 50 MB'tan 30 MB'a indirildi.

### Testler
- 122 testten 231 teste çıkıldı (artı isteğe bağlı büyük ölçüm). 1.0 testleri korundu.
- Yeni test dosyaları: gizlilik, bölütleme, lemmatizasyon, veri kalitesi, dosya güvenliği, TF-IDF
  referans karşılaştırmaları (UD Turkish-BOUN) ve performans.
- GitHub Actions testleri Python 3.12 ve 3.13 ile çalıştırıyor.

### Yeni bağımlılıklar
- `simplemma==2.0.0`: lemmatizasyon, MIT lisanslı.
- `xlsxwriter==3.2.9`: güvenli Excel yazımı, BSD lisanslı.

## 1.0.0 — İlk sürüm
CSV/Excel/TXT yükleme, Türkçe ön işleme, sözcük sıklığı, n-gram, TF-IDF, grafikler, kelime bulutu,
CSV/Excel indirme, örnek veri seti.
