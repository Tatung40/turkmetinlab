# TürkMetinLab 2.5 — E5-small ONNX optimizasyonu (Aşama 2)

Tarih: 19 Ekim 2026 · Dal: `gelistirme-2.5-fizibilite` · Uygulama kodu ve `main` değiştirilmedi.

## 1. Durum

- **Gerçek E5 ONNX ölçümü bu ortamda yapılamadı.** Hugging Face buradan erişilemiyor (403; tek deneme). Bu
  raporda E5 ONNX için **hiçbir sayı yoktur**. Ölçümler elle başlatılan `.github/workflows/onnx-2.5.yml` ile
  GitHub Actions'ta yapılacaktır.
- **Hazır olan:** dönüştürme, torch'suz kodlayıcı, 20 örnek uyum denetimi ve karşılaştırma betiği. Hepsi küçük,
  rastgele ağırlıklı bir modelle uçtan uca sınandı (çıkış kodu 0). Bu sınamanın sayıları **E5 sonucu değildir**,
  raporlanmamıştır.
- Sınama sırasında bir hata bulundu ve düzeltildi: ONNX tarafında token uzunluğu dolgu dahil sayılıyordu. Bu,
  kesilme oranını yanlış (fazla) gösterirdi.

## 2. Referans (kullanıcının GitHub Actions n=500 çalıştırması; JSON elimizde yok)

| Ölçüt (görev B, n=500) | Değer |
|---|---|
| TF-IDF (`tfidf_tml`) nDCG@5 | 0,317 |
| E5 PyTorch, özgün metin | 0,304 |
| TF-IDF + E5 hibrit | 0,342 (TF-IDF'ye göre **+0,025**) |
| E5 PyTorch tepe RAM / 500 belge | ~2.631 MB / 63,84 sn |

Bu değerler yeniden üretilmedi. Güven aralıkları bildirilmediği için burada yorumlanmadı. ONNX iş akışı TF-IDF ve
PyTorch E5'i **aynı makinede yeniden ölçer**; süre/RAM karşılaştırması yalnızca o çalıştırmanın içinden yapılmalıdır.

## 3. Yöntem

- **Model ve lisans:** `intfloat/multilingual-e5-small`, MIT. Ağırlıklar depoya, ZIP'e veya artifact'e konmaz;
  yalnızca Actions önbelleğinde tutulur.
- **Dönüştürme** (`e5_onnx_donustur.py`, bir kez; sonuç Actions önbelleğinde):
  - `torch.onnx.export` (TorchScript, opset 17; olmazsa bir kez dynamo); dinamik toplu iş ve uzunluk.
  - Çıktı: `last_hidden_state`. `token_type_ids` sıfır kabul edilir (tek bölüt).
  - INT8: `onnxruntime.quantization.quantize_dynamic`, ağırlıklar QInt8.
  - Künye (`onnx_kunye.json`): kaynak sürüm (commit), torch sürümü, dosya boyutları ve SHA-256.
  - **INT8 hatası:** Nicemleme ya da INT8 dosyasının açılması başarısız olursa hata künyeye yazılır ve yalnızca
    INT8 elenir; FP32 deneyleri sürer. FP32 dışa aktarılamaz ya da açılamazsa iş durur.
  - **Önbellek bütünlüğü:** Önbellekten devam ederken FP32, INT8 (hazırsa) ve `tokenizer.json` künyedeki boyut ve
    SHA-256 ile, künye de kaynak model sürümüyle denetlenir. Eksik, bozuk ya da uyumsuzsa yeniden oluşturulur.
    Yeni dosyalar geçici klasörde üretilir, künye en son yazılır ve klasör ancak tamamlanınca yerine taşınır.
- **Çıkarım** (`e5_onnx_kodlayici.py`): yalnızca `onnxruntime` + `tokenizers` + `numpy`; **torch yüklenmez**
  (sınandı ve sonuçta `torch_yuklendi_mi` olarak kaydedilir).
  - Modelin kendi `tokenizer.json`'ı kullanılır; kesme sınırı 512 token.
  - Dikkat maskeli ortalama havuzlama ve L2 normalizasyonu uygulanır (sentence-transformers ile aynı).
  - Önekler değişmedi: başlık `query: `, özet `passage: `, özet→özet `query: `.
- **Uyum denetimi** (`e5_onnx_uyum.py`, 20 örnek; eşikler önceden yazıldı):
  - FP32: en küçük kosinüs ≥ 0,999 ve ilk-1 sıralama %100 aynı. Geçemezse iş durur.
  - INT8: en küçük kosinüs ≥ 0,98 ve ilk-5 örtüşmesi ≥ 0,80. Geçemezse 100/500 deneyine alınmaz.
- **Karşılaştırma** (`anlamsal_karsilastirma.py`; aynı örneklem, tohum 2501):
  - Yöntemler: TF-IDF, E5 PyTorch, E5 ONNX FP32, E5 ONNX INT8 ve her birinin TF-IDF hibriti.
  - Ölçütler: nDCG@5, İsabet@5 (= Recall@5 görev A'da), MRR; TF-IDF'ye ve PyTorch E5'e göre eşleştirilmiş fark
    (%95 bootstrap GA).
  - Uyum: ilk-5 örtüşmesi ve ilk-1 aynılığı.
  - Kaynak: yükleme süresi, 100 ve 500 belge süresi, başlangıç (yükleme sonrası) ve tepe RAM (`VmHWM`), dosya
    boyutu, 3 eşzamanlı oturum.
- **Etiket uyarısı:** Görev B'nin ilgililiği yazar anahtar sözcüklerinden türetilir. Bu, uzman yargısı değildir.

## 4. Önceden belirlenen karar ölçütleri

**ONNX entegrasyon adayı sayılır, ancak:**

1. **Kayıp yok:** ONNX − PyTorch nDCG@5 farkının %95 GA alt sınırı görev A ve B'de **> −0,01** olmalıdır. FP32
   için ayrıca 20 örnek uyum koşulu geçilmiş olmalıdır.
2. **Kaynak:** Aynı çalıştırmada, PyTorch E5'e göre tepe RAM'de **en az %30 azalma** olmalıdır; 500 belge
   süresi de PyTorch'tan uzun olmamalıdır.
3. **Streamlit sınırı:** 3 eşzamanlı oturumda toplam bellek, 2.1 uygulamasının tepe kullanımıyla birlikte
   **≤ 690 MB** olmalıdır (README, ölçüt 3). Streamlit'te denenmedikçe bu yalnızca **tahmini uyum** sayılır.
4. **Bilimsel katkı (ayrı ölçüt):** Hibritin görev B'de TF-IDF'ye göre kazancı **≥ +0,03** olmalı ve GA 0'ı
   içermemelidir.
5. **Gizlilik ve ücret:** Çıkarım yerel CPU'da yapılmalı, dış hizmet kullanılmamalı ve ek ücret gerekmemelidir.

## 5. Entegrasyon kararı (bugünkü kanıtla)

- **ONNX:** Karar verilemez; ölçüm yok. 1–3. ölçütler iş akışı çalıştırıldıktan sonra değerlendirilecek.
- **Hibrit:** Kullanıcının referansına göre kazanç **+0,025**. Bu, önceden belirlenen **0,03 eşiğinin altındadır**.
  ONNX en iyi durumda PyTorch ile aynı sonucu verir; doğruluğu artırmaz. Dolayısıyla **ONNX ne kadar hafif olursa
  olsun, mevcut kanıtla hibritin 2.5'e alınması önerilmez.** E5 tek başına (0,304) TF-IDF'nin (0,317) altındadır.
- **Bu durumda ONNX çalışmasının değeri:** Hibrit ileride (ör. insan etiketli bir değerlendirmeyle) eşiği geçerse,
  ücretsiz sunucuda çalışabilecek bir yolun hazır ve ölçülmüş olmasıdır.
- 500 belgelik ONNX sonucu ölçütleri karşılarsa öneri yalnızca şöyle değişir: "ayrı dalda, isteğe bağlı,
  **deneysel** sekme olarak ve Streamlit'te bellek denemesinden sonra". Varsayılan yöntem TF-IDF kalır.

## 6. Kalan riskler

- **Streamlit üzerinde test edilmedi.** GitHub makinesi (açık depoda 4 çekirdek/16 GB, özel depoda 2 çekirdek/7 GB) Streamlit sunucusundan güçlüdür.
- Model dosyasının (boyutu iş akışında ölçülecek) her uyanışta indirilmesi gerekir. Bu, açılış süresi ve
  Hugging Face erişimine bağımlılık yaratır.
- INT8 nicemleme işlemciye bağlıdır (AVX2/VNNI); farklı sunucuda süre ve uyum farklı olabilir.
- Görev B etiketleri dolaylıdır; veri çoğunlukla mühendislik özetidir.
- Kullanıcının referans değerlerinin JSON'u görülmedi; GA'ları bilinmiyor.

## 7. GitHub'da yapılması gerekenler

1. Bu daldaki `.github/workflows/onnx-2.5.yml` dosyasını **`main`'e de** ekleyin. `workflow_dispatch` düğmesi
   yalnızca varsayılan dalda bulunan iş akışlarında görünür. Dosya yalnızca elle başlar; uygulamayı değiştirmez.
2. Actions → **ONNX 2.5 (E5-small)** → *Run workflow* → dal: `gelistirme-2.5-fizibilite`.
3. Artifact `onnx-2.5-sonuclar` içinde şu dosyalar olmalı:
   - `onnx_kunye.json`, `onnx_uyum_20.json`;
   - `sonuclar_n100_onnx.*`, `sonuclar_n500_onnx.*`.
4. `sonuclar_n500_onnx.json` ile 4. bölümdeki ölçütler doldurulmalı ve bu rapor güncellenmelidir.
