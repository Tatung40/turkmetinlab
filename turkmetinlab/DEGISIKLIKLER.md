# Değişiklikler

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
