# TürkMetinLab 2.5 — Anlamsal model fizibilite raporu

Tarih: 19 Ekim 2026 · Dal: `fizibilite-2.5` (2.1'den açıldı; uygulama kodu değiştirilmedi)

## 1. Sonuç

**Henüz hiçbir model önerilemez ve hiçbir model uygulamaya entegre edilmemelidir.**

- **Modeller çalıştırılamadı.** İki gömme modeli bu çalışma ortamında indirilemedi: ağ politikası Hugging
  Face'i engelliyor (`403 Forbidden`; tek deneme yapıldı ve sonuç dosyasına kaydedildi). Bu yüzden MiniLM ve E5
  için **erişim başarısı, süre ve bellek ölçümü yoktur.** Tahmin de üretilmemiştir.
- **Ölçülebilen her şey ölçüldü:**
  - TF-IDF referansının gerçek verideki başarısı;
  - gömme altyapısının çalışma zamanı belleği;
  - token kesilmesi için kesin bir alt sınır.

  Bu ölçümler iki önemli risk gösteriyor:
  1. **Bellek:** `torch` + `sentence-transformers`, model yüklenmeden 765 MB bellek kullanıyor. Bu, ücretsiz
     sunucu için tek başına aşırı yüksek. Aynı iş için ONNX Runtime çalışma zamanı 53 MB.
  2. **Kesilme:** MiniLM'in 128 token sınırı özetlerin **en az %61'ini** kesiyor.
- **Ek hazırlık (19 Ekim):** Ölçümleri ücretsiz GitHub Actions'ta yapacak, elle başlatılan bir iş akışı hazırlandı
  (README). Henüz çalıştırılmadı; model sonuçları bekleniyor.
- **Yapılması gereken:** Karşılaştırma betiği hazır ve uçtan uca sınandı. Modeller kullanıcının kendi
  bilgisayarında bir komutla ölçülmeli (README). Karar, önceden yazılmış ölçütlere göre verilmelidir.

## 2. TF-IDF referansı (gerçek veri, ölçüldü)

Veri ve örneklem:
- DergiPark, 500 özet (1.395 uygun benzersiz özetten; tohum 2501), ortanca 139,5 sözcük.
- Görev A: 500 sorgu. Görev B: 269 sorgu (ortalama 6 ilgili belge).
- Değerler: ortalama [%95 güven aralığı].

| Yöntem | A: İsabet@1 | A: MRR | A: nDCG@5 | B: İsabet@5 | B: MRR | B: nDCG@5 |
|---|---|---|---|---|---|---|
| Özgün metin (`tfidf_ham`) | 0,914 | 0,943 | 0,952 [0,935–0,966] | 0,606 | 0,471 | 0,322 [0,280–0,364] |
| **TürkMetinLab (`tfidf_tml`, referans)** | 0,904 | 0,937 | 0,947 [0,931–0,962] | 0,602 | 0,468 | 0,317 [0,277–0,359] |
| + terim sözlüğü | 0,896 | 0,932 | 0,944 | 0,599 | 0,469 | 0,316 |
| + lemmatizasyon | 0,930 | 0,954 | 0,962 | 0,591 | 0,459 | 0,303 |

**Ön işlemenin etkisi:**
- Referansa göre farklar **pratik eşiğin (0,03) altında.** Tek istatistiksel olarak ayırt edilebilir fark,
  lemmatizasyonun görev A'daki etkisi (nDCG@5 +0,014 [+0,006, +0,023]). Bu fark da pratik açıdan küçüktür ve
  üstünlük olarak yorumlanmamalıdır.
- Durak sözcük temizliği ve terim birleştirme erişimi ne artırdı ne azalttı. Gömme modellerinde bu seçeneklerin
  etkisi **ayrıca** ölçülecek (`_ozgun` ve `_temiz` karşılaştırması). Etkileri olumlu varsayılmamalıdır.

**Görevlerin yorumu:**
- **Görev A tavana yakın:** TF-IDF %90 oranında doğru özeti ilk sırada buluyor. Bu görev, anlamsal modellerin
  katkısını göstermeye uygun değildir.
- **Asıl ayırt edici görev B:** nDCG@5 ≈ 0,32 olduğundan iyileşme payı geniş. Ancak etiketi dolaylı ve
  gürültülüdür (ortak yazar anahtar sözcüğü).

## 3. Kaynak ölçümleri (bu makinede, CPU)

| Ölçüm | Değer | Durum |
|---|---|---|
| TF-IDF, 500 özet: süre ve tepe bellek | 0,4 sn, 155 MB (lemma ile 1,8 sn, 270 MB) | Ölçüldü |
| `import torch` (PyPI 2.14.1+cu130) | 478 MB | Ölçüldü |
| `import torch` + `sentence-transformers` | **765 MB** (model yok) | Ölçüldü |
| torch kurulum boyutu (diskte) | 1,2 GB; PyPI'den kurulumda ek CUDA paketleri | Ölçüldü |
| `onnxruntime` + `tokenizers` | **53 MB**, diskte 63 MB | Ölçüldü |
| Model ağırlıkları (her biri ~118 milyon parametre, fp32 ≈ 0,47 GB) | — | Ölçülmedi (indirilemedi) |
| MiniLM: kesilen özet oranı | **≥ %61**: 126 sözcükten uzun özetler; her sözcük en az 1 token | Kesin alt sınır |
| E5: kesilen özet oranı | ≥ %0: en uzun özet 441 sözcük, sınır 512 token | Alt sınır; gerçek oran ölçülmeli |
| Model yükleme, 100/500 belge süresi, tepe RAM, eşzamanlı kullanım | — | **Ölçülmedi** |

**Not:** CPU'ya özel torch sürümü daha küçüktür. Bu ortamdan PyTorch paket sunucusuna erişilemediği için ölçülemedi.

## 4. Ücretsiz Streamlit dağıtımı için teknik riskler

1. **Bellek:** 2.1 uygulaması orta boyutlu veride zaten 360 MB'a, 4 eşzamanlı kullanıcıyla 566 MB'a çıkıyor
   (2.0 ölçümü). torch tabanlı bir model, ağırlıklarıyla birlikte büyük olasılıkla **1,2 GB'ı aşar.** Bu,
   ücretsiz plan için kabul edilemez bir risktir.
2. **Kurulum:** `requirements.txt` içinde PyPI'den gelen torch Linux'ta CUDA paketlerini de kurar (birkaç GB).
   CPU sürümü için ek paket dizini gerekir; derleme süresi ve disk sınırı açısından risklidir.
3. **İlk açılış:** Model, uygulama her uyandığında yeniden indirilir (~0,5 GB). Bu, açılışı yavaşlatır ve
   Hugging Face erişimine bağımlılık yaratır.
4. **Süre:** Paylaşılan sunucuda kodlama süresi bu makineden yavaş olacaktır. Yerel ölçümler sunucu ölçümü
   değildir.
5. **Gizlilik:** Model önbelleği (`st.cache_resource`) yalnızca model ağırlıklarını tutmalıdır. Gömme vektörleri
   kullanıcı verisidir ve 2.1'deki gibi yalnızca oturum belleğinde kalmalıdır.

## 5. Gerekçeli mimari önerisi (koşullu)

Ölçümler yapılana kadar uygulamaya entegrasyon **önerilmez.** Ölçüm sonunda bir model karar ölçütlerini
(README) sağlarsa güvenli yol şudur:

1. **Çalışma zamanı:** torch değil, **ONNX Runtime + tokenizers** kullanılmalı (ölçülen ek yük 53 MB'a karşı
   765 MB). Model ONNX'e dönüştürülmeli; mümkünse 8 bit nicemlenmeli. Nicemlemenin başarıya etkisi aynı betikle
   yeniden ölçülmelidir.
2. **Model seçimi önceliği:** Uzun akademik metinler için **E5-small** daha uygun görünüyor (512 token). MiniLM
   özetlerin en az %61'ini 128 token'da kesiyor. Bu bir kesinti gerekçesidir, başarı iddiası değildir.
3. **Kademeli kullanım:** Gömme isteğe bağlı ayrı bir sekme olmalı, sınırlı belge sayısıyla (ör. ≤ 2.000)
   çalışmalı ve düğmeyle başlamalı. TF-IDF varsayılan kalmalı.
4. **Hibrit sıralama:** Betik, TF-IDF + gömme skor birleştirmesini de ölçüyor. Yalnızca ölçülürse
   değerlendirilmeli.
5. **Hiçbir model ölçütleri sağlamazsa** 2.5'te anlamsal model olmamalıdır. TF-IDF + terim sözlüğü + 2.0
   modülleri yeterli kabul edilmelidir.

## 6. Bilimsel sınırlılıklar

- İnsan tarafından ilgililik yargısı verilmiş Türkçe akademik bir erişim veri seti bulunamadı.
  - Görev A'nın etiketi gerçek ama görev sözcüksel olarak kolay.
  - Görev B'nin etiketi dolaylı (yazar anahtar sözcükleri).
- Veri büyük ölçüde mühendislik özetleridir; diğer alanlara genellenemez.
- Bootstrap güven aralıkları sorgu örneklemesindeki belirsizliği yansıtır; etiket gürültüsünü yansıtmaz.
- Betik sahte (rastgele ağırlıklı, küçük) bir modelle uçtan uca sınandı. O çalıştırmanın sayıları **model
  sonucu değildir** ve hiçbir yerde raporlanmamıştır.
