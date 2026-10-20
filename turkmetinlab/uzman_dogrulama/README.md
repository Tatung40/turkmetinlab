# TürkMetinLab 2.5 — Uzman doğrulaması (TF-IDF / E5 / hibrit)

TF-IDF, E5 ve hibrit yöntemlerini iki bağımsız uzmanın ilgililik yargılarıyla karşılaştırmak için hazırlanmıştır.
**Henüz hiçbir uzman puanı yoktur. Bu klasör bir sonuç değil, bir ölçüm aracıdır.** Analiz planı ve karar ölçütü
puanlar görülmeden yazılmıştır: `ortak.py` başındaki açıklama.

## 1. Havuzun üretilmesi (GitHub Actions, ücretsiz)

1. **Parola:** GitHub → *Settings → Secrets and variables → Actions → Secrets* bölümüne `TML_DOGRULAMA_PAROLA`
   adıyla güçlü bir parola ekleyin.
   - Depo herkese açık olduğu için metin içeren dosyalar **yalnızca bu parolayla şifrelenmiş** olarak saklanır.
   - Parola yoksa iş durur.
2. **Çalıştırma:** *Actions → ONNX 2.5 (E5-small) → Run workflow* yolunu izleyin. Dal: `gelistirme-2.5-staging`,
   kip: `uzman`.
   - **Önce aşama `pilot` (5 sorgu):** Yönergeyi ve süreyi denemek içindir. Pilot sorgular asıl çalışmaya hiç
     girmez: asıl 50 sorgu önce çekilir, pilot kalan belgelerden seçilir.
     - Pilot dosyalarında ayrı işaretler vardır: kimlikler `P-` ile başlar, havuz kimliği ayrıdır ve dosyada
       "Aşama: PİLOT" yazar. Dosya adları `pilot_` ile başlar.
     - Analiz, aşaması uymayan dosyayı reddeder. Pilot için karar üretilmez.
   - **Pilot sonrası:** Yönergede değişiklik gerekirse yalnızca `YONERGE.md`'yi düzeltin. Ölçütler ve +0,030
     eşiği değişmez.
   - **Ardından aşama `asil` (50 sorgu).**
3. **Çıktılar** (artifact `onnx-2.5-sonuclar`):
   - `uzman_havuz_ozeti_<aşama>.json`: metin içermez. Sorgu, küme ve çift sayıları ile dışlama sayılarını gösterir.
   - `uzman_paketi_<aşama>.7z`: şifrelidir. Windows'ta 7-Zip ile açılır. İçinde iki klasör vardır:
     - `uzmanlara_gonderin/`: `uzman_1.xlsx`, `uzman_2.xlsx`, `YONERGE.md`. Her uzmana **yalnızca kendi
       dosyasını** ve yönergeyi gönderin.
     - `yalnizca_arastirmaci/anahtar.json`: çift ↔ yöntem/sıra eşleşmesi. **Uzmanlara göndermeyin.**

Seçim kuralları, tohumlar (asıl 2510, pilot 2511) ve dışlamalar: `havuz_olustur.py` başındaki açıklama.

**Şifreleme** (`paketle.sh`):
- 7z yoksa Ubuntu deposundan ücretsiz `p7zip-full` kurulur.
- Paket AES-256 ile şifrelenir; dosya adları da şifrelidir.
- Sonra paket denetlenir:
  - doğru parolayla bütünlük;
  - yanlış parolayla listelenememe;
  - eksiksiz içerik.
- Bir denetim başarısız olursa paket silinir ve iş durur. Şifresiz dosyalar geçici klasörde kalır ve iş sonunda
  silinir.
- Parola günlüğe yazılmaz.

## Kaynak ve lisans durumu (bilinmiyor)

- **Lisans:** DergiParkEn deposunda (humakilicunlu/DergiParkEn) lisans dosyası ya da yeniden dağıtım izni yoktur.
  README yalnızca atıf ister: Kılıç Ünlü, H. ve Çetin, A. (2023). *Neural Computing and Applications*, 35,
  3413–3422.
- **Telif:** Özetlerin telif hakları yazarlara ya da dergilere aittir. **Yeniden dağıtım koşulları bilinmiyor.**
- **Bu nedenle:**
  - Metinler depoya ve açık artifact'e konmaz.
  - Yalnızca şifreli olarak ve iki uzmanın iç değerlendirmesi için kullanılır.
- **Yayında paylaşılabilecekler:** Metin içermeyen bilgiler; örneğin kaynak, veri SHA-256'sı, sıra numaraları ve
  puanlar. Daha geniş paylaşım için kaynaktan izin alınmalıdır.

Tahmini iş yükü: yaklaşık 50 × 20 = 1.000 çift (gerçek sayı özet dosyasında). Çift başına 20–30 saniyeyle her uzman
için 6–8 saat sürer. Birkaç oturuma bölünmelidir.

## 2. Uzman değerlendirmesi

- **Bağımsız puanlama:** Uzmanlar `YONERGE.md`'ye göre birbirinden habersiz puanlar.
  - Puanlar: 0 = ilgisiz, 1 = kısmen ilgili, 2 = ilgili.
  - Uzman yalnızca "Puan" ve "Açıklama" hücrelerini doldurabilir; öteki hücreler kilitlidir.
- **Körleme:** Dosyalarda yöntem adı, sıra ve benzerlik puanı yoktur.
  - Çift kimlikleri rastgeledir.
  - Adayların sırası her uzman için ayrı karıştırılmıştır.
- **Eşleştirme güvencesi:** Her satırda, metinlerden üretilen bir denetim kodu vardır. Analiz şunları denetler ve
  sorun varsa durur:
  - satır kaydırma ya da metin düzenleme;
  - yinelenen ya da eksik kimlik;
  - aynı dosyanın iki kez verilmesi;
  - başka bir havuzun dosyası.

## 3. Analiz (yerel; ağ gerekmez)

```bash
# a) İki dosya geldikten sonra: uyum + ön analiz + uzlaşma dosyası
python uzman_dogrulama/analiz.py --anahtar anahtar.json --uzman1 uzman_1.xlsx --uzman2 uzman_2.xlsx --cikti sonuc
# b) Uzlaşı toplantısında sonuc/uzlasma.xlsx doldurulduktan sonra: birincil analiz ve karar
python uzman_dogrulama/analiz.py ... --uzlasma sonuc/uzlasma.xlsx --cikti sonuc2
```

### Uyum

- **Birincil:** Doğrusal ağırlıklı Cohen κ ve Gwet AC2.
- **İkincil:**
  - karesel ağırlıklı κ ve AC2;
  - ağırlıksız κ ve AC1;
  - ham uyum.
- **Güven aralığı:** Sorgular üzerinden küme bootstrap ile.
- **Ayrıca raporlananlar:**
  - çapraz tablo ve sınıf dağılımları;
  - uyuşmazlık yönü ve büyüklüğü (0↔2);
  - adayın hangi yöntemden geldiğine göre uyuşmazlık;
  - en çok uyuşmazlık olan sorgular.

### Uzlaşı

Uzlaşı, **iki dosya da teslim edildikten sonra** yapılır. `uzlasma.xlsx` yalnızca uyuşmazlıkları içerir: iki puanı
ve açıklamaları gösterir, yöntem bilgisi vermez. Uzmanlar birlikte bir uzlaşı puanı ve gerekçe yazar.

### Erişim ölçütleri

- **Ölçütler:** nDCG@5 ve nDCG@10 (kazanç 2^ilgililik − 1). Precision@5 ve @10 iki eşikle verilir: katı (= 2) ve
  gevşek (≥ 1).
- **Etiket kümeleri:**
  - uzlaşı (birincil);
  - her uzman ayrı ayrı ve iki uzmanın ortalaması (duyarlılık analizi).
- **Karşılaştırma:** Sorgu bazında eşleştirilmiş bootstrap (10.000 yeniden örnekleme, %95 güven aralığı).

### Kapsam

- **Havuz:** Değerlendirme yalnızca üç yöntemin ilk 10 sonucundan oluşan **havuz içindedir.** Bu nedenle ilk 10
  sonucun tümü yargılanmıştır. Yargısız belge ilgisiz sayılmaz; böyle bir belge varsa analiz hata verir.
- **Ölçülemeyen:** Gerçek geri çağırma (recall) ölçülemez.
- **Havuz yanlılığı göstergeleri:** "havuz içi geri çağırma" ve yalnızca tek yöntemden gelen aday sayıları.

### Karar (önceden belirlendi, değiştirilmez)

Hibrit − TF-IDF nDCG@5 farkı **≥ +0,030** olmalı ve güven aralığının alt sınırı **0'dan büyük** olmalıdır.
Uzlaşı yoksa karar verilmez.

## 4. Sınırlılıklar

- 50 sorgu ve iki uzman: güven aralıkları geniş olacaktır.
- Derlem çoğunlukla mühendislik özetlerinden oluşur.
  - "Konu yayılımı" yazar anahtar sözcüklerinden çıkarılan kümelerle sağlanır; resmî alan sınıflaması değildir.
  - Sonuçlar başka alanlara genellenmemelidir.
- Üç düzeyli kısmi ilgililik özneldir. Yönerge ve uzlaşı süreci bu öznelliği azaltır, ama tamamen ortadan kaldırmaz.
- **Görev:** Yalnızca belge → belge benzerliği değerlendirilir. Kısa sorgu ile arama görevi kapsam dışıdır.