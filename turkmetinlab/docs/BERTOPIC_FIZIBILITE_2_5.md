# BERTopic 2.5 — Gerçek entegrasyon fizibilitesi

**Durum:** İzole, isteğe bağlı prototip; kullanıcı arayüzünde etkinleştirilmemiştir.

## Önceki engel
TürkMetinLab'ın ücretsiz Streamlit ortamında Torch/Sentence-Transformers bağımlılıkları yüksek RAM ve disk kullanır. Geçmişte ağ erişimi yüzünden modeller çalıştırılamamış ve modelden sonuç üreten sahte veya boş geri dönüş kabul edilmemiştir. E5-small ONNX FP32 altyapısı şu anda deneysel dalda çalışır. BERTopic'in kendisi E5 ONNX kodlayıcısı değildir; mevcut belge vektörlerini BERTopic'e doğrudan vermek resmi API tarafından desteklenir.

## Prototip yaklaşımı
`tmlab.bertopic_adapter.run(docs, embeddings, n_topics)` mevcut E5 vektörlerini alır. BERTopic gerçek kurulumunu içeri alır; KMeans kümeleme ile sabit konu sayısı kullanır. Böylece HDBSCAN yerine KMeans seçildiği raporda açıkça belirtilir. Konu terimleri c-TF-IDF ile hesaplanır. BERTopic'in varsayılan UMAP davranışı bağımlılıklara ve sürümlere göre değişebilir: tam olarak hangi boyut indirgeme algoritmasının kullanıldığı gerçek ortam testinde doğrulanmadan açıklanamaz.

Bu, **klasik UMAP+HDBSCAN BERTopic ile aynı işlem hattı değildir**. Haricî ücretli API gerektirmez. Streamlit'ın temel requirements dosyasına bertopic, torch veya sentence-transformers eklenmedi.

## Başarı ölçütleri
Gerçek ortamda bertopic içe aktarımı, en az 30 Türkçe metnin gerçek E5 embedding ile işlenmesi, anlamlı konu anahtar sözcükleri, belge/konu satırı eşitliği, deterministik KMeans ve makul tepe bellek tüketimi ayrıca doğrulanmalıdır. Bu koşullar tamamlanmadan 2.5 kullanıcı arayüzü etkinleştirilmez.

## Sınırlılıklar ve raporlama
KMeans sabit konu sayısı ister; HDBSCAN'deki gürültü konu -1 davranışı burada zorunlu değildir. Kümeleme yapılarının tutarlılık, konu çeşitliliği ve insan yorumlarıyla değerlendirilmesi gerekir. NMF ve LDA korunur. E5 vektörleri anlam yakınlığı sağlasa da Türkçe konu etiketleri otomatik olarak nitel tema sayılmaz.

Kaynak: https://github.com/MaartenGr/BERTopic, resmi embedding dokümanı: https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html
