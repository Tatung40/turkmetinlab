# TürkMetinLab 2.1 — Bilimsel ve Teknik Değerlendirme Raporu

Konu: Akademik terim sözlüğü ve çok sözcüklü terim tanıma · Tarih: 16 Ekim 2026 · Dal: `gelistirme-2.1`

Bütün sayılar `degerlendirme/v21_terim_dogrulama.py` betiğiyle üretilmiş ve
`degerlendirme/v21_terim_sonuclari.json` dosyasına yazılmıştır.

## 1. Kısa özet

| Soru | Yanıt |
|---|---|
| Terimler ek almış biçimleriyle tanınıyor mu? | Evet. Gerçek akademik metinde ekli geçişlerin %98,4'ü (dondurulmuş sürüm), eksiz geçişlerin %99,5'i bulundu. |
| Yanlış eşleşme çok mu? | Biçim düzeyinde çok az: 717 eşleşmede 2. Ancak eşleşmelerin %9,6'sında sözlükteki dizi **başka bir anlamda** geçiyordu; bunların neredeyse tamamı eğitim alanındaki "derin öğrenme" teriminin makine öğrenmesi anlamında kullanılmasıydı. **Alan seçimi** bu sorunu büyük ölçüde gideriyor (kavram düzeyi kesinlik %90,1 → %99,5). |
| Genel Türkçe metinde yanlış alarm var mı? | 400 cümlelik akademik olmayan metinde 0. |
| Sözlük analizleri iyileştiriyor mu? | **Kendiliğinden hayır.** Terimleri okunur ve tutarlı kılıyor, ama konu modeli ve benzerlik ölçütleri karışık yönde değişti (bölüm 7). |
| Bulgular genellenebilir mi? | Sınırlı olarak. Gerçek veri büyük ölçüde mühendislik özetlerinden oluşuyor ve etiketlemeyi geliştirici yaptı (uzman değil). AkademikDerlem bu ortamdan erişilemediği için kullanılamadı. |

## 2. Yöntem

### 2.1 Tanıma kuralları (`tmlab/terimler.py`)

1. **Özgün metin değiştirilmez.** Eşleştirme, karakter konumlarını özgün metne geri eşleyen
   normalleştirilmiş bir görünümde yapılır. Bu görünüm şunları uygular: Türkçe küçük harf, şapka katlama
   (â→a), kesme ve tire varyantlarının birleştirilmesi, NFKC, yumuşak tire ve satır sonu tiresi. Her
   eşleşme `metin[başlangıç:bitiş]` = metindeki biçim olacak şekilde saklanır; bu, testlerde denetlenir.
2. **Sözcük sınırları.** Terimin son sözcüğü dışındaki sözcükler birebir eşleşir.
   - Sözcükler arasında boşluk, tek satır sonu ya da tek tire olabilir; noktalama terimi böler.
   - Tireyle önceki sözcüğe bağlı terim eşleşmez: "log-normal dağılım" içinde "normal dağılım" yoktur.
     Tek harfli önekler bu kuralın dışındadır: "k-en yakın komşu".
3. **Çekim ekleri.** Son sözcükte yalnızca çekim ekleri kabul edilir.
   - Kabul edilen ekler: çoğul, iyelik, altı hâl eki, -ki, ek-eylem.
   - Kabul edilen ek dizileri ünlü uyumuna göre üretilen sonlu bir kümedir. Kaynaştırma (y, n, s) ve D/t,
     C/ç değişimi denetlenir.
   - Ünsüz yumuşaması kabul edilir.
   - Ünlü düşmesi ve ikizleşme yalnızca kısa bir listede kabul edilir (kayıt → kaydı, tıp → tıbbı). Genel
     kural "eğitim → eğitme", "ölçüm → ölçme" gibi gerçek ama farklı sözcükler üretirdi.
4. **Belirtisiz isim tamlaması.** Son sözcük iyelik ekliyse (kuramı) çoğul yol da kabul edilir (kuramları).
   Ünlü uyumu iyelik ünlüsünden alınır; böylece "kontrolü → kontrolleri" doğru işlenir.
5. **Zincirleme tamlamada iyelik düşmesi.** "makine öğrenme yöntemleri" ve "Pearson korelasyon katsayısı"
   gibi biçimler ayrı bir eşleşme türü ("iyeliksiz") olarak, daha düşük güven düzeyiyle kabul edilir.
6. **Türetim ekleri eşleşmez.** "uzaktan eğitimci", "yapay zekâsız", "değerlendirmeci", "açık erişimli"
   gibi biçimler standartlaştırılmaz; nedeniyle birlikte ayrı bir tabloya yazılır.
7. **Kısaltmalar** (isteğe bağlı, varsayılan olarak kapalı):
   - büyük/küçük harfe duyarlıdır ("YEM" eşleşir, "yem" eşleşmez);
   - ek yalnızca kesme işaretiyle kabul edilir;
   - tamamı büyük harfli başlık satırlarında reddedilir.
8. **Çakışmalar** şu öncelik sırasıyla çözülür. Reddedilen her aday çakışma tablosuna yazılır.
   1. En uzun terim (üretken yapay zekâ ⊃ yapay zekâ).
   2. Standart biçim > varyant > kısaltma.
   3. Eksiz > kesme işaretli > çekimli > iyeliksiz.
   4. Kullanıcı terimi > alan sözlüğü > Genel araştırma ve istatistik.
   5. Metinde önce gelen.
9. **Temsil.**
   - "Tek birim" (varsayılan): terim tek belirteç olur (`uzaktan_eğitim`). Bu belirteç lemmatizasyon,
     durak sözcük, sayı ve uzunluk süzgeçlerinden geçmez. Bütün analizler aynı belirteç listesini
     kullandığı için n-gram, TF-IDF, ağ, konu ve benzerlik analizlerinde temsil aynıdır.
   - "Yalnızca işaretle": analizler sözlük kapalıyken olduğu gibi kalır; bu, testle doğrulanmıştır.

### 2.2 Değerlendirme verileri ve geliştirme/test ayrımı

| Veri | Rolü | Nasıl hazırlandı |
|---|---|---|
| Entegrasyon notlarındaki örnekler, birim test cümleleri | Geliştirme | `tests/test_terimler.py` |
| **Yapay test kümesi** (158 satır: 142 beklenen terim, 19 negatif cümle) | Test | Her alandan 12 terim sabit tohumla seçildi. Cümleler, **eşleştirici yazılmadan önce** elle yazılıp git'e kaydedildi (`degerlendirme/veri/terim_test_yapay.tsv`). |
| **DergiPark** (1.397 benzersiz Türkçe özet, mühendislik ağırlıklı) | Yarısı geliştirme, yarısı test | Belgeler sabit tohumla (2101) 698/699 ikiye ayrıldı. Geliştirme yarısında hatalar arandı ve kurallar düzeltildi. Sonra eşleştirici donduruldu (commit `ab6b3dd`) ve test yarısı etiketlendi. |
| UD Turkish-BOUN (400 cümle, akademik olmayan genel metin) | Yanlış alarm denetimi | `tests/veri` |

**DergiPark altın standardı** (`degerlendirme/veri/terim_altin_dergipark.tsv`; yalnızca konum ve etiket,
metin yok):

- **Aday bölgeler:** Eşleştiriciden bağımsız, gevşek (yüksek duyarlıklı) bir arama yapıldı: ilk sözcükler
  birebir, son sözcük önekle aranıyor; tire, şapka ve büyük/küçük harf farkı yok sayılıyor. Buna kısa
  önekli ek arama ve eşleştiricinin çıktısı eklendi (havuzlama). Ek bölgeler etiketleyiciye karışık sırayla
  verildi.
- **Etiketler:** 751 bölge, eşleştiricinin kararları görülmeden etiketlendi:

  | Etiket | Anlamı | Sayı |
  |---|---|---|
  | T | Gerçek terim | 662 |
  | A | Sözlükteki dizi geçiyor ama anlam ya da alan farklı | 69 |
  | D | Türetilmiş biçim | 5 |
  | V | Sözlükte olmayan varyant ("bilgisayarlı görme") | 4 |
  | N | Terim değil | 11 |

- **Sınırlılıklar:**
  - Etiketlemeyi **geliştirici (Claude) yaptı; bu bir uzman etiketlemesi değildir.**
  - Havuzdaki aramaların hiçbirinin yakalamadığı geçişler ölçülemez; duyarlılık bu yüzden üst sınır
    tahminidir.
  - Etiketleme sırasında iki bölgede bir satırlık kayma fark edilip düzeltildi; bütün etiketler bölge
    metniyle tek tek karşılaştırılarak denetlendi.

**Fark edilen veri sorunu:** DergiPark dosyasındaki kimlik sütunu benzersiz değildir (414 kimlik 929 farklı
özette tekrar eder) ve 20 özet birebir yinelenir. 2.1'de kimlik olarak satır sırası kullanıldı ve yinelenen
metinler çıkarıldı. 2.0 benzerlik validasyonu yinelenenler çıkarılarak yeniden hesaplandı; fark önemsiz çıktı:
AUC 0,865 → 0,869, başlıkla ilk sırada bulma %84,8 → %84,8.

**AkademikDerlem:** Kullanıcı yalnızca küçük, tekrarlanabilir bir örneklemle kullanmaya izin verdi. Ancak
bu çalışma ortamının ağ politikası Hugging Face'e erişimi engelliyor (403). Bu yüzden **hiçbir kayıt
indirilmedi ve bu veriyle değerlendirme yapılamadı.** Erişimin olduğu bir ortamda izlenecek yöntem
`DEVAM_NOTLARI.md`'de yazılıdır.

## 3. Yapay test kümesi

Bu kümedeki tam başarı, **gerçek metinde kanıtlanmış doğruluk değildir.** Cümleleri, kuralları bilen
geliştirici yazdı; küme yalnızca hedeflenen dilbilgisi durumlarının kapsandığını gösterir.

| Kırılım | Doğru | Yanlış pozitif | Kaçırılan |
|---|---|---|---|
| Genel | 142 | 0 | 0 |
| Ekli biçimler | 99 | 0 | 0 |
| Eksiz biçimler | 38 | 0 | 0 |
| Kısaltmalar (seçenek açık) | 5 | 0 | 0 |
| İç içe terimler (en uzununun bulunması) | 9 | 0 | 0 |
| Negatif cümleler (19 cümle: türetim, noktalama, "yem", büyük harfli başlık…) | — | 0 | — |
| Her alan (6 alan, 20–24'er öğe) | hepsi doğru | 0 | 0 |

## 4. Gerçek akademik metin (DergiPark test yarısı, 699 özet)

### 4.1 Asıl sonuç: dondurulmuş eşleştirici

Bu sonuçlar test etiketlemesinden önce dondurulmuş sürüme (commit `ab6b3dd`) aittir.

**Yüzey düzeyi** ölçüm, sözlükteki dizinin doğru konum ve kimlikle bulunup bulunmadığını sorar; A
etiketliler doğru sayılır. **Kavram düzeyi** ölçüm, terimin sözlükteki anlamıyla geçip geçmediğini sorar;
A etiketliler yanlış pozitif sayılır.

| Senaryo | Düzey | Doğru | Yanlış pozitif | Kaçırılan | Kesinlik | Duyarlılık | F1 |
|---|---|---|---|---|---|---|---|
| 6 alan, kısaltmalar kapalı (varsayılan) | yüzey | 715 | 2 | 6 | **0,997** | **0,992** | 0,994 |
| | kavram | 646 | 71 | 6 | **0,901** | 0,991 | 0,944 |
| 6 alan, kısaltmalar açık | yüzey | 725 | 2 | 6 | 0,997 | 0,992 | 0,995 |
| | kavram | 656 | 71 | 6 | 0,902 | 0,991 | 0,945 |
| Yalnızca mühendislik + genel araştırma alanları | yüzey | 547 | 2 | 5 | 0,996 | 0,991 | 0,994 |
| | kavram | 546 | 3 | 5 | **0,995** | 0,991 | 0,993 |

**Kırılımlar** (dondurulmuş sürüm, varsayılan senaryo; kısaltma öğeleri hariç 652 terim geçişi):

| Grup | Geçiş | Bulunan | Duyarlılık |
|---|---|---|---|
| Eksiz biçimler | 409 | 407 | 0,995 |
| Ekli biçimler | 243 | 239 | 0,984 |
| İç içe terimler (en uzununun bulunması) | 83 | 81 | 0,976 |

Eşleşme türüne göre kesinlik (kavram düzeyi):

| Eşleşme türü | Kesinlik |
|---|---|
| Çekimli | 0,982 (214/218) |
| Kesme işaretli | 1,000 (3/3) |
| İyeliksiz | 0,957 (22/23) |
| Eksiz | 0,861 (407/473) |

Eksiz biçimlerde kesinliğin düşük olmasının nedeni, "derin öğrenme" teriminin 62 kez eksiz geçmesidir.

**Hataların dökümü:**

- **Kavram düzeyinde 71 yanlış pozitif:**

  | Neden | Sayı |
  |---|---|
  | "derin öğrenme" (eğitim alanında kayıtlı) makine öğrenmesi anlamında geçmiş | 62 |
  | "performans değerlendirmesi" (eğitim alanında, öğrenci performansını değerlendirme anlamında kayıtlı) model, denetleyici ya da sanallaştırma performansının değerlendirilmesi anlamında geçmiş | 6 |
  | "faktör analizi", vinçlerde dinamik yük faktörü anlamında | 1 |
  | "açık veri": "erişime açık veri seti" ve "halka açık veri tabanları" | 2 (yalnızca bu ikisi biçim hatasıdır) |

  Alan bazında, kavram düzeyinde eğitim alanının kesinliği 0,218'dir (19 doğru, 68 yanlış). Diğer
  alanlarda kesinlik 0,96–1,00 arasındadır.
- **6 kaçırılan:**
  - "Evrişimli Sinir Ağları’dır" (kesmeden önce çekim eki);
  - "k-En yakın komşu algoritması" (tek harfli önek);
  - "yaşam kalitemizi" ve "makine öğrenmenin" (iyeliği düşmüş gövdeye ek);
  - "5-katmanlı-çapraz-doğrulama" (tireli zincir);
  - "Yapay Sinir Ağlari" (yazım hatası: "ları" yerine "lari").

### 4.2 Test sonrası düzeltmeler (bağımsız değil)

Kaçırılan ilk dört durum için kurallar genişletildi:
- kesme işaretinden önce çekimli gövde;
- tek harfli önek;
- iyeliği düşmüş gövdeye çekim eki.

Bu değişiklikler test verisi görüldükten sonra yapıldığı için **aşağıdaki sayılar bağımsız bir ölçüm
değildir.** Yeni kuralların yeni yanlış pozitif üretmediğini göstermek için verilmiştir.

| Senaryo (düzeltilmiş sürüm) | Kesinlik | Duyarlılık |
|---|---|---|
| Yüzey, 6 alan | 0,997 | 0,997 |
| Kavram, 6 alan | 0,901 | 0,997 |
| Kavram, mühendislik + genel | 0,995 | 0,996 |

Kırılım: eksiz 0,998, ekli 0,996, iç içe 0,988. Yapay kümede ve 135 geliştirme testinde gerileme olmadı.

### 4.3 Geliştirme yarısında bulunan ve düzeltilen sorunlar

Bunlar test etiketlemesinden önce yapıldı.

1. **Zincirleme tamlamada iyelik düşmesi** ("makine öğrenme yöntemleri", "Pearson korelasyon katsayısı",
   "saldırı tespit sistemi"): yaklaşık 20 kaçırılan geçiş vardı. "İyeliksiz" eşleşme türü eklendi.
2. **"IOT":** IoT kısaltması büyük harfle yazıldığında da tanınıyor; küçük harfli "yem" hâlâ tanınmıyor.
3. **"log-normal dağılım"** içinde "normal dağılım" eşleşiyordu; tireli bileşik kuralı eklendi.
4. Kimliklerin benzersiz olmaması, değerlendirme betiğinde eşleşmelerin yanlış belgeye atanmasına yol
   açıyordu; satır sırası kimliğine geçildi.

## 5. Genel Türkçe metinde yanlış alarm

Akademik olmayan UD Turkish-BOUN örneğinde (400 cümle, 4.033 sözcük; haber, biyografi, deneme…) kısaltmalar
açıkken **1 eşleşme** bulundu: "sivil toplum kuruluşları". Bu gerçek bir terimdir; **yanlış alarm 0**.

## 6. Ön işlemenin terimlere etkisi

652 gerçek terim geçişi üzerinde, her ayar için şu oranlar ölçüldü:
- **İçine boşluk giren:** çıkarılan bir sözcük terimi bölüyor mu? Bölünen terim n-gram olamaz.
- **Standart biçimde kalan:** işlenmiş dizi, terimin standart biçimiyle aynı mı?
- **Tutarlı temsil:** terimin bütün geçişleri tek bir n-gramda toplanabiliyor mu?

| Ayar | İçine boşluk giren | Standart biçimde kalan | Tutarlı temsil | Sözlük açık: tek ve standart birim |
|---|---|---|---|---|
| Hiçbiri | %0,0 | %62,6 | %59,2 | **%99,8** |
| Durak sözcük (varsayılan) | %0,2 | %62,3 | %59,1 | %99,8 |
| Durak + akademik kalıp | %0,2 | %60,7 | %59,1 | %99,8 |
| Lemmatizasyon | %0,0 | %21,8 | %94,9 | %99,8 |
| Lemma + durak | %0,2 | %21,8 | %94,8 | %99,8 |

**Yorum:**
- Sözlük kapalıyken çekim ekleri aynı terimi farklı n-gramlara dağıtıyor: geçişlerin yalnızca %59'u tek bir
  n-gramda toplanabiliyor.
- Lemmatizasyon tutarlılığı %95'e çıkarıyor, ama terimlerin %78'ini standart dışı bir biçime dönüştürüyor:
  "uzaktan eğitim → uzak eğitim", "makine öğrenmesi → makine öğren".
- Sözlük her iki sorunu birlikte çözüyor. Kalan %0,2 tek geçiş, yazım hatalı "Ağlari"dır.
- Durak sözcük temizliği bu derlemde terimleri nadiren bölüyor (yalnızca "ölçme **ve** değerlendirme" türü
  terimler).

## 7. Sözlüğün analizlere etkisi (1.397 DergiPark özeti)

Ayarlar: durak sözcük (Türkçe ve İngilizce), lemma kapalı. Sözlük açıkken 6 alan seçili ve "tek birim"
temsili kullanıldı.

| Ölçüt | Sözlük kapalı | Sözlük açık |
|---|---|---|
| Farklı sözcük sayısı | 30.809 | 30.902 |
| Belge benzerliği: ortak yazar anahtar sözcüklü çiftleri ayırma (AUC) | **0,871** | 0,851 |
| NMF (10 konu): NPMI tutarlılığı | 0,251 | 0,254 |
| NMF: konu çeşitliliği | 0,876 | 0,872 |
| NMF: alt örneklem kararlılığı (ortalama Jaccard) | **0,683** | 0,598 |
| NMF: konu sözcükleri ile yazar anahtar sözcüklerinin örtüşmesi (rastgele düzey) | 0,430 (0,095) | 0,374 (0,075) |
| LDA (10 konu): NPMI tutarlılığı | 0,043 | **0,127** |
| LDA: anahtar sözcük örtüşmesi (rastgele düzey) | 0,324 (0,098) | 0,227 (0,064) |
| TF-IDF ilk 30: ortak öğe | — | 29/30 |
| İkili (bigram) ilk 30: ortak öğe | — | 23/30 |
| Ön işleme süresi | 0,9 sn | 2,0 sn |

**Yorum:**
- Sözlük, **analitik başarımı kendiliğinden artırmıyor.** LDA tutarlılığı arttı; NMF kararlılığı,
  benzerlik AUC değeri ve anahtar sözcük örtüşmesi azaldı.
- **Olası açıklama:** Terim tek birim olunca "yapay sinir ağı" ile "sinir ağı" ya da "makine öğrenmesi" ile
  "makine öğrenme" geçen belgeler, ortak "sinir" ya da "makine" sözcüğünü artık paylaşmıyor; kısmi sözcük
  örtüşmesi kayboluyor.
- **Anahtar sözcük örtüşmesi ölçütünün sınırı:** Bu ölçüt, küçük ve çok sayıda belirteci kayırır. Tek sözcük
  temsilinde bir anahtar sözcüğün herhangi bir parçasının eşleşmesi yetiyor.
- **Pratik öneri:** Sonuçları her iki durumda da inceleyin; "Sözlük açık / kapalı" sekmesi bunun için
  vardır. Raporda hangi temsili kullandığınızı belirtin. Terimleri yalnızca listelemek istiyorsanız
  "Yalnızca işaretle" seçeneği analizleri değiştirmez.
- **Bulgunun kapsamı:** Bu sonuçlar tek bir derleme (mühendislik özetleri) aittir; başka alanlarda ve
  uzun metinlerde farklı çıkabilir.

## 8. Performans ve bellek

Sentetik derlem kullanıldı: Zipf dağılımlı ve her ~50 sözcükte bir sözlük terimi içeriyor. Bu yoğunluk
gerçek özetlerdekinin yaklaşık iki katıdır. Pandas her iki durumda da yüklüdür.

| Derlem | Sözlük | Ön işleme | Tepe bellek | Eşleşme |
|---|---|---|---|---|
| 295 bin sözcük | yok | 1,4 sn | 133 MB | — |
| 295 bin sözcük | tek birim | 2,4 sn | 147 MB | 3.310 |
| 2,95 milyon sözcük | yok | 14,4 sn | 304 MB | — |
| 2,95 milyon sözcük | tek birim | 23,4 sn | 357 MB | 33.227 |

- **Sabit maliyet:** Sözlüğün derlenmesi 0,01 sn ve yaklaşık 10 MB bellek tutuyor (ek kümeleri dahil).
- **Kısaltma seçeneği:** Süreye etkisi ölçülebilir değil.
- **Veri sınırları:** Mevcut sınırlar korunmuştur (500 bin sözcükte uyarı; 1,5 milyon sözcük üstünde bulutta
  analiz yok).
- **Testler:** Bu ölçümler `tests/test_performans.py` içinde otomatik testtir.

## 9. Güvenlik, gizlilik ve tekrarlanabilirlik

- **Kullanıcı terimleri:** Elle eklenen terimler, yüklenen sözlük ve devre dışı bırakılan terimler yalnızca
  `st.session_state` içinde (`_tml_` önekiyle) tutulur. "Verilerimi temizle" bunları da siler (testle
  doğrulandı).
- **Önbellek:** Derlenmiş eşleştirici oturum belleğindedir. Modül düzeyindeki tek önbellek, ek kümelerini
  ses profiline göre tutar; sözcük ya da terim içermez. Bunu tarayan bir test vardır.
- **Oturum yalıtımı:** İki ayrı oturumda biri terim ekleyince öteki etkilenmiyor (AppTest ile doğrulandı).
- **Sözlük dosyası yükleme:** Var olan güvenli okuyucu kullanılır (dosya türü, sıkıştırma bombası,
  formüller). Formül biçimli bir terim düz metin olarak kalır; Excel çıktısında hiçbir hücre formül değildir
  (tarayıcı testi).
- **Tekrarlanabilirlik:** Excel'deki *Ayarlar* sayfasına şunlar yazılır:
  - sözlük sürümü (SHA-256 özeti) ve seçilen alanlar;
  - temsil ve kısaltma seçenekleri;
  - eklenen, yüklenen ve devre dışı bırakılan terimler;
  - eşleştiricinin parmak izi.

## 10. Kalan sınırlılıklar

1. **Sözlük uzman doğrulamasından geçmemiştir.** Terimler aday terimdir. Sözlükteki eş anlamlı kayıtlar,
   eksik varyantlar ve belirsiz terimler `docs/sozluk_denetimi_2.1.md`'de listelenmiştir; hiçbiri sessizce
   değiştirilmemiştir.
2. **Anlam ayrımı yapılmaz.** Aynı dizi farklı anlamda geçebilir ("derin öğrenme", "performans
   değerlendirmesi"). Alan seçimi, "alanlar arası belirsiz" uyarısı ve terimi devre dışı bırakma bu sorunu
   azaltır ama tamamen çözmez.
3. **Gerçek veri kanıtı tek alanda güçlüdür.** Test yarısındaki 652 terim geçişinin 480'i mühendislik
   terimidir. Eğitim (19), sağlık (21), psikoloji (31) ve ekonomi (30) alanlarında öğe sayısı azdır. Bu
   alanlar için doğruluk iddiası yapılmamalıdır.
4. **Etiketleme uzman etiketlemesi değildir;** tek etiketleyici olduğu için etiketleyiciler arası uyum
   ölçülemedi.
5. **AkademikDerlem ile değerlendirme yapılamadı** (ağ erişimi yok).
6. **Bilinen kural sınırları:**
   - Ünlü düşmesi yalnızca listedeki sözcüklerde tanınır.
   - Alıntı sözcüklerin ünlü uyumu istisnaları kısmen ele alınır.
   - Yazım hataları ("Ağlari", "yapay zekatekniklerinden") tanınmaz.
   - Türetilmiş biçimler bilerek tanınmaz. Bu, "tip 2 diyabetli hastalar" gibi kavramı içeren ifadelerde
     duyarlılığı düşürür; bu biçimler "eşleşmeyen biçimler" tablosunda görünür.
7. **Terim birimleri benzerliği azaltabilir;** kısmi sözcük örtüşmesi kaybolur (bölüm 7).
8. **Sözlükteki standart biçim çoğulsa** (ör. "öğretmen adayları"), tekil biçim ("öğretmen adayı")
   tanınmaz; kullanıcı varyant olarak ekleyebilir.
