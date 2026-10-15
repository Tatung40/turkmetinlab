# TürkMetinLab 1.1 — Bilimsel ve Teknik Değerlendirme

Bu belge 1.1 sürümündeki kararların hangi ölçümlere dayandığını özetler. Tüm sayılar bu sürüm
geliştirilirken gerçekten çalıştırılan ölçümlerden alınmıştır. Yeniden üretmek için kullanılan
betikler `degerlendirme/` klasöründedir.

Kullanılan altın standart veriler: **UD Turkish-BOUN** ve **UD Turkish-IMST** ağaç bankaları
(Universal Dependencies). Bu veri setlerinde dilbilimciler her sözcüğün ayrımını, lemmasını ve
sözdizim ilişkisini elle işaretlemiştir. Lisans: BOUN CC BY-SA 4.0. IMST yalnızca ölçüm için
kullanıldı, depoya eklenmedi.

---

## 1. Gizlilik: paylaşılan önbellek riski

**Bulgu.** 1.0 sürümü yüklenen metinleri ve sonuçları `st.cache_data` ile önbelleğe alıyordu.
Streamlit belgelerine göre bu önbellekteki değerler *uygulamanın tüm kullanıcılarına açıktır* ve
oturum kapandıktan sonra da sunucu belleğinde kalır. Başka bir kullanıcı aynı dosyayı yüklemedikçe
doğrudan erişemez. Yine de kullanıcı metinleri süresiz olarak paylaşılan bellekte tutuluyordu. Bu,
1.0'daki "hiçbir yere kaydedilmez" ifadesiyle uyumlu değildi.

İncelemede üç sorun daha bulundu:

| Sorun | 1.1'de ne yapıldı |
|---|---|
| Excel yazıcısı (xlsxwriter) varsayılan olarak çalışma sayfalarını geçici dosya olarak **diske** yazıyor (`tempfile.mkstemp`). | `in_memory=True` ayarı açıldı. Test, geçici dosya oluşturulmaya çalışılırsa başarısız olur. |
| Streamlit yakalanmamış hataları tam mesajıyla sunucu günlüğüne yazıyor. Mesaj (ör. ayrıştırma hatası) kullanıcı metninden parça içerebilir ve uygulama sahibinin günlüğünde görünebilir. | Uygulama gövdesi güvenli bir sarmalayıcıyla çalışıyor. Günlüğe yalnızca hata türü ve kod konumu yazılıyor; test bunu doğruluyor. |
| İndirme düğmeleri dosya içeriğini her etkileşimde sunucu belleğine kopyalıyordu. | Dosyalar artık yalnızca düğmeye basılınca üretiliyor (tembel indirme). |

**Yeni mimari.**
- Kullanıcı verisinden türeyen her hesaplama yalnızca ilgili oturumun `st.session_state` alanında
  tutulur (`tmlab/oturum.py`). Kayıt sayısı sınırlıdır.
- `st.cache_data` yalnızca herkese açık örnek veri için kullanılır. Bu kural, kaynak kodunu
  tarayan bir testle güvence altındadır.
- **"Verilerimi şimdi temizle"** düğmesi oturumdaki tüm sonuçları ve Streamlit'in bellekte tuttuğu
  yüklenmiş dosyaları hemen siler.
- Düğme kullanılmazsa Streamlit, bağlantı koptuktan sonra en geç `server.disconnectedSessionTTL`
  süresi (varsayılan 120 sn) dolunca oturumu ve dosyaları siler.

**Gizlilik metni** bu davranışa göre yeniden yazıldı. Metin, verinin Streamlit Community Cloud
(Snowflake) sunucularında işlendiğini de açıkça belirtir. Kişisel ya da gizli veriler için
anonimleştirmeyi veya uygulamanın yerel olarak çalıştırılmasını önerir.

**Kalan sınır.** Python, silinen nesnelerin belleğini sıfırlamaz. Bellek işletim sistemine hemen
geri verilmeyebilir. Bulut sağlayıcısının altyapısı (ağ, sunucu) bu uygulamanın denetiminde değildir.

---

## 2. Türkçe lemmatizasyon

**Değerlendirilen yöntemler** (BOUN test: 9.711 sözcük, IMST test: 7.625 sözcük; noktalama ve sayılar hariç):

| Yöntem | Tür | BOUN | IMST | Hız | Ek bellek |
|---|---|---|---|---|---|
| Hiç işlem yapmamak | — | %45,8 | %43,0 | — | — |
| Snowball | kural tabanlı kök bulucu | %57,0 | %54,3 | 0,05 ms/sözcük | 9 MB |
| zeyrek 0.1.3 | Zemberek uyarlaması, bağlamsız | %83,6 | — | 1,1 ms/sözcük | 126 MB |
| zemberek-python 0.2.3 | Zemberek uyarlaması, bağlamlı | %92,3 | %93,3 | 4,3 ms/sözcük | 116 MB |
| **simplemma 2.0.0** | sözlük tabanlı, bağlamsız | **%94,5** | **%94,4** | 0,08 ms/sözcük | ~110 MB* |

\*Sözlük ilk kullanımda belleğe açılır.

Karşılaştırma yapılırken Türkçe küçük harf kuralları uygulandı ve şapkalı harfler sadeleştirildi.
Zemberek'in fiil biçimi ("gelmek") UD biçimine ("gel") çevrildi.

**Neden Zemberek değil?**
- **Java sürümü:** Ücretsiz sunucuda ek kurulum (Java) ve bu ortamdan indirilemeyen bağımlılıklar
  gerektiriyor.
- **Python uyarlaması (zemberek-python):**
  - Bakımı durmuş bir bağımlılık kullanıyor (ANTLR 4.8). Bu bağımlılık Python 3.13'te kaldırılan
    `typing.io` modülüne dayandığı için çalışmıyor.
  - Ayrıca `pkg_resources` modülüne ihtiyaç duyuyor; bunun için setuptools'un eski bir sürümüne
    sabitlenmesi gerekiyor.
  - Çok yavaş: 300 bin sözcük yaklaşık 22 dakika sürer.
  - Doğruluğu simplemma'dan yüksek değil.

**Neden zeyrek değil?** Bağlam çözümlemesi yok. Aday lemmaları `set()` ile birleştirdiği için hangi
lemmanın seçileceği çalıştırmadan çalıştırmaya değişebilir; bu, sonuçların tekrarlanabilirliğini
bozar. NLTK verisini ayrıca indirmeyi de gerektiriyor.

**Veri sızıntısı denetimi.** simplemma'nın belgelerine göre sözlüğü UD *eğitim* bölümlerinden
yararlanır. Değerlendirmede kullanılan *test* bölümleri dışarıda tutulmuştur. Sonuç, birbirinden
bağımsız iki ağaç bankasında tutarlı çıkmıştır (%94,5 ve %94,4).

**Hata çözümlemesi** (BOUN, içerik sözcükleri):
- %93,5 doğru.
- %3,9 türetim ekleri fazla soyuldu (*politikacı → politika*).
- %2,3 türetim ekleri eksik soyuldu.
- Olumsuzluk eki **-sız/-siz** içeren sözcüklerde hata oranı %43 (58 sözcükte 25). Bu ekin atılması
  anlamı tersine çevirir (*anlamsız → anlam*).

**Koruma kuralı.** Lemma + (sız|siz|suz|süz) sözcüğün başıysa lemmaya bu ek geri eklenir. Altın
veride sınandı:

| Veri | Kural yokken doğruluk | Kuralla doğruluk | Kuralın değiştirdiği sözcükler (doğru: önce → sonra) |
|---|---|---|---|
| BOUN test | %93,50 | %93,68 | 21 sözcük: 3 → 17 |
| BOUN dev | %93,16 | %93,26 | 14 sözcük: 2 → 10 |
| IMST test | %94,00 | %94,00 | 15 sözcük: 7 → 7 |

Kural hiçbir veride doğruluğu düşürmediği için uygulanır.

**Karar.** Lemmatizasyon **isteğe bağlı ve varsayılan olarak kapalı** sunulur. Arayüzde ölçülen
doğruluk ve sınırlılıklar gösterilir. Değişen her sözcük bir tabloda listelenir ve dışa aktarılır.

**Kullanıcıya bildirilen sınırlılıklar:**
- Bağlama bakılmaz.
- Eylem adları fiil köküne iner (*çalışma → çalış*). Bu, UD altın verisinin kendi kuralıdır.
- Belirteçler köke inebilir; bu yüzden kalıp terimler değişir (*uzaktan eğitim → uzak eğitim*).
- Yaklaşık 20 sözcükte 1 hata beklenir.

---

## 3. N-gram bölütleme

**Soru.** Virgül, noktalı virgül, iki nokta ve tire n-gram sınırı sayılmalı mı?

**Yöntem.** UD BOUN (eğitim + geliştirme + test) ve IMST (eğitim + test) bir arada incelendi.
Aralarında hiç ya da tek bir noktalama işareti bulunan ardışık iki sözcüğün, aynı sözdizim
öbeğinde doğrudan bağlı olup olmadığı ölçüldü. Bağlı sayılmayan ilişkiler: eşgüdüm (conj),
yan yana cümle (parataxis) ve kök (root).

| Aradaki işaret | Sözcük çifti | Öbek içi doğrudan bağlı |
|---|---|---|
| Noktalama yok | 115.430 | **%60,2** |
| Virgül | 9.682 | **%3,4** |
| Noktalı virgül | 313 | %1,3 |
| İki nokta | 327 | %2,4 |
| Bitişik tire (*sosyo-ekonomik*) | 345 | %57,4 |
| Boşluklu tire (*a – b*) | 47 | %19,1 |

**Yorum.**
- **Virgül:** İki yanındaki sözcükler noktalama olmayan durumdan 18 kat daha az bağlıdır.
  Bağlı oldukları durumlar da çoğunlukla öbek değil, yan cümle düzeyindeki ilişkilerdir (ccomp,
  obj, nsubj). Virgülü aşan bir ikili ("ekonomi, sağlık" → "ekonomi sağlık") büyük olasılıkla
  yapaydır.
- **Noktalı virgül, iki nokta ve boşluklu tire:** Bunlar da güçlü sınırdır.
- **Bitişik tire:** Sözcük içi bir işarettir; bu yüzden tireli sözcük tek sözcük sayılmaya devam eder.

**Uygulanan seçenekler:**

| Düzey | Sınır sayılan işaretler |
|---|---|
| **Katı** (varsayılan) | cümle sonu, ; : , parantez, tırnak, eğik çizgi, boşluklu tire |
| **Orta** | Katı ile aynı, ama virgül sınır değil |
| **Yalnızca cümle sonu** | . ! ? … ve paragraf |

**Ek kurallar:**
- Sıra sayısındaki nokta cümle sonu sayılmaz (*19. yüzyıl*). Noktadan sonra küçük harf geliyorsa
  sıra sayısı kabul edilir.
- Yaygın kısaltmalardaki nokta cümle sonu sayılmaz (*Prof. Dr., vb., bkz.*).
- Sayı içindeki ayırıcılar bölmez (*3,5 · 1.250 · 10:30*).
- Paragraf (boş satır) her zaman sınırdır.
- Tek satır sonu varsayılan olarak sınır **değildir**. PDF'ten kopyalanan metinlerde satırlar cümle
  ortasında biter (1.0'da her satır sonu sınırdı). İstenirse arayüzden açılabilir.
- Çıkarılan web adresleri sınır bırakır.

---

## 4. Sözcük ayrımı ve TF-IDF doğrulaması

- **Sözcük ayrımı:** Altın standarda karşı 400 cümle ve 4.031 sözcükte **F1 = %99,74**.
  Farklılıklar sınır durumlarda: tırnağın kesme işareti yerine kullanılması ("P"ye") ve sıra
  sayısındaki noktayı sözcüğe dahil eden işaretleme.
- **TF-IDF, altın sözcüklerle:** Gerçek Türkçe veride (5 tür, 40 belge, 1.500'den fazla terim)
  scikit-learn sonuçları, bağımsız olarak yazılmış saf Python uygulamasıyla dört ayar bileşiminin
  hepsinde birebir aynıdır (fark < 10⁻⁶). Ayar bileşimleri: yumuşatılmış/yumuşatılmamış IDF ×
  ham/logaritmik TF.
- **TF-IDF, uçtan uca:** Ham metinden kendi sözcük ayrımımızla hesaplanan en ayırt edici 10 terim,
  altın sözcük ayrımıyla hesaplananlarla ortalama %95'in üzerinde örtüşür.
- **Matematiksel özellikler:** Belge sırası sonucu değiştirmez. Bir belgeyi kendisiyle birleştirmek
  L2 normalleştirilmiş değerleri değiştirmez.
- **Kayıtlı beklenen sonuçlar:** Örnek veri setinin TF-IDF sonuçları bağımsız uygulamayla üretilip
  `tests/veri/ornek_tfidf_beklenen.json` dosyasına kaydedildi. İstenmeyen değişiklikleri yakalar.
- **1.0 testleri:** Hepsi korunmuştur.

---

## 5. Dosya güvenliği ve veri kalitesi

| Risk | Önlem |
|---|---|
| Satır kayması (fazla sütunlu satır) | 1.0'dan beri atlanır ve satır numarasıyla bildirilir |
| Kapanmamış tırnak (satırların tek hücreye yutulması) | Sezgisel tespit ve uyarı |
| Eksik sütunlu satır | Boş kabul edilir ve bildirilir |
| Boş hücre, "NA/null/#YOK" yer tutucuları, Excel hata değerleri | Analize alınmaz, satır numarasıyla bildirilir. Excel'de "NA" artık sessizce boşa çevrilmez. |
| Yinelenen belge kimliği | (2), (3) eklenir ve bildirilir |
| Yinelenen metin | Bildirilir. İstenirse bir kez sayılır. |
| Formül içeren Excel hücreleri | Formüller hiçbir zaman çalıştırılmaz; sayılır ve bildirilir |
| Formül enjeksiyonu (dışa aktarımda) | Excel'de tüm metinler "metin" türünde yazılır, formül ya da bağlantı oluşmaz. CSV'de "=, +, -, @" ile başlayan metin hücrelerine ' eklenir (isteğe bağlı). |
| Uzantısı değiştirilmiş dosya (PDF, görsel, arşiv, Word, eski .xls) | Dosya imzası denetlenir ve anlaşılır bir hatayla reddedilir |
| İkili veri, NUL baytlar | Reddedilir |
| Sıkıştırma bombası (.xlsx) | Açılmış boyut 200 MB'ı aşarsa dosya açılmadan reddedilir |
| Başlık satırı olmayan dosya | "İlk satır başlık" seçeneği (1.0'da ilk belge sessizce kayboluyordu) |
| Excel hücre sınırı (32.767 karakter) | Uzun metin işaretlenerek kısaltılır ve bildirilir. CSV'de tam metin bulunur. |
| Dosya adıyla Markdown enjeksiyonu | Arayüzde kaçış uygulanır |

---

## 6. Performans ve bellek

Ölçüm `degerlendirme/performans_olcumu.py` ile, 2 çekirdekli bir makinede yapıldı. Her ölçüm ayrı
bir süreçte çalıştı. Tepe bellek için `VmHWM` kullanıldı; Linux'ta `ru_maxrss` üst süreçten
devralındığı için yanıltıcıdır.

Derlemler gerçekçi sözcük dağarcıklıdır (Zipf dağılımı):

| Derlem | Sözcük | Farklı sözcük | Analiz süresi | Tepe bellek (analiz) | Excel dahil |
|---|---|---|---|---|---|
| Küçük | 12 bin | 925 | 1,0 sn | 233 MB | 233 MB |
| Orta | 288 bin | 60 bin | 3,8 sn | 287 MB | 458 MB |
| Orta + lemmatizasyon | 288 bin | 60 bin | 5,3 sn | 404 MB | 579 MB |
| Büyük | 2,88 milyon | 175 bin | 22,1 sn | 698 MB | 806 MB |
| Büyük + lemmatizasyon | 2,88 milyon | 175 bin | 28,1 sn | 848 MB | 927 MB |

1.1'deki iyileştirmelerle büyük derlemde toplam süre 87 sn'den 36 sn'ye, tepe bellek
1.501 MB'tan 806 MB'a indi. İyileştirmeler:
- Vektörel Türkçe sıralama.
- TF-IDF'te belge başına NumPy sıralaması.
- Sözcük türü önbelleği ve bellekte tek kopya (`sys.intern`).
- Excel'de sayfa başına 50.000 satır sınırı; tablonun tamamı CSV ile alınır.

Streamlit Community Cloud bir uygulamaya yaklaşık 690 MB – 2,7 GB bellek verir. Tüm kullanıcılar
aynı süreci paylaşır; Streamlit'in kendisi de boşta yaklaşık 300 MB kullanır. Bu nedenle:
- **500 bin sözcüğün üzerinde** kullanıcı uyarılır.
- **1,5 milyon sözcüğün üzerinde** bulutta analiz yapılmaz. Kullanıcıya veriyi bölmesi ya da
  uygulamayı kendi bilgisayarında çalıştırması önerilir. Yerelde sınır `TML_EN_FAZLA_SOZCUK` ortam
  değişkeniyle yükseltilebilir.
- Arayüz tabloları ilk 5.000 satırı, ön işleme önizlemesi ilk 500 belgeyi gösterir. Tamamı indirilebilir.
- Yükleme sınırı 30 MB'tır.

Gerçek tarayıcıyla yapılan denemede:
- 288 bin sözcüklük CSV 3,4 sn'de analiz edildi.
- 648 bin sözcük uyarıyla birlikte 9,3 sn'de analiz edildi.
- 1,58 milyon sözcük 4,4 sn içinde gerekçesiyle reddedildi.
- Sunucu günlüğünde kullanıcı metni bulunmadı.

---

## 7. Kalan sınırlılıklar

- Lemmatizasyon bağlamdan bağımsızdır (~%94 doğruluk). Bağlamlı bir çözümleyici (ör. Zemberek)
  ücretsiz bulut ortamında güvenilir biçimde çalıştırılamadı.
- Bölütleme kuralları kural tabanlıdır. Kısaltma listesi sınırlıdır; tamamı küçük harfle yazılmış
  metinlerde sıra sayısı kuralı cümle sonunu kaçırabilir.
- Kapanmamış tırnak tespiti sezgiseldir; her durumu yakalamaz.
- Performans sınırları sentetik (Zipf) derlemlerle ölçülmüştür. Gerçek veriler biraz farklı
  davranabilir. Bulut sunucusu bu ölçüm makinesinden yavaş olabilir.
- Bellek temizliği Python ve Streamlit'in davranışına bağlıdır. Silinen veriler bellekten hemen
  fiziksel olarak kaldırılmayabilir.
