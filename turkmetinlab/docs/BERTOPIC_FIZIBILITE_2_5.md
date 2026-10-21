# TürkMetinLab 2.5 — BERTopic Konu Modelleme Kılavuzu

**Durum:** Yalnızca 2.5 staging dalında deneysel modül. NMF/LDA ve canlı 2.1 değiştirilmedi.

## 1. Yöntem ne yapar?

BERTopic, belgeleri anlamsal vektörlerine göre kümeleyip kümelerin ayırt edici kelimelerini c-TF-IDF ile ortaya çıkarır. TürkMetinLab'ın uyguladığı hat, BERTopic'in standart UMAP+HDBSCAN yapılandırması değildir: E5-small ONNX + KMeans + c-TF-IDF kullanılır. Ücretli API veya yeni bir uzaktan YZ hizmeti gerekmez.

Kaynak: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794. Resmî depo: https://github.com/MaartenGr/BERTopic

## 2. Analiz birimi ve veri hazırlığı

CSV/Excel'de her satır bir belge, tez özeti veya yorumdur. İsteğe bağlı kimlik sütununu seçin. Deneysel modül **10–500 belge** arasında çalışır. Boş kayıt, yinelenen kayıt ve dil karışımı varsa araştırma amacına uygun biçimde önceden gözden geçirin. Konu sayısı seçimi her zaman araştırmacıya aittir.

## 3. Hesaplama basamakları ve varsayılanlar

1. **Anlamsal temsil:** Belgelerin özgün metni `passage: ` önekiyle `intfloat/multilingual-e5-small` ONNX FP32 modeline verilir. Maskeli ortalama havuzlama ve L2-normalizasyon sonrasında her belgenin vektörü elde edilir.
2. **Uzunluk kontrolü:** E5'in 512 belirteç sınırı aşılırsa sessiz kırpma yapılmaz. Önce uzun belgeleri kısaltmanız gerektiği belirtilir. Bu özellik veri kaybına karşı korumadır.
3. **Kümeleme:** `KMeans(n_clusters=k, random_state=42, n_init=10)` çalıştırılır. Kullanıcı konu sayısı k'yı seçer; her belge tam olarak bir kümeye atanır. HDBSCAN'in -1 gürültü etiketi bu yapılandırmada üretilmez.
4. **Boyut indirgeme:** `BaseDimensionalityReduction()` kullanılır; UMAP uygulanmaz ve E5 vektörleri doğrudan KMeans'e girer.
5. **Sözcük çıkarma:** Kümelerindeki belgeler bir araya getirilir; Türkçe Unicode sözcükler `CountVectorizer` ile sayılır. Her konu birleştirilmiş tek bir metin olarak işlediği için minimum frekans **1** tutulur; daha yüksek eşik konuya özgü ayırt edici sözcükleri silebilir.
6. **c-TF-IDF:** Konudaki göreli sözcük sıklıkları, sözcüğün tüm konu belgelerindeki yaygınlığıyla ağırlıklandırılır. Yüksek c-TF-IDF ağırlığı, bir sözcüğün o konuyu ayırt etmede katkı sağladığını gösterir; bu sayı olasılık değildir.

## 4. Sonuç dosyaları

- **bertopic_konular.csv:** konu numarası, konu başına belge sayısı, otomatik konu adı, önemli sözcükler ve mevcutsa temsilî belge bilgileri.
- **bertopic_belgeler.csv:** analizdeki sıralaması korunmuş her belgenin kimliği ve atandığı konu numarası.

Bir belgede iki konu birlikte bulunabileceği hâlde bu yapılandırma **tek baskın küme** atar. Temsilî yorumlar kişisel bilgi içerebileceğinden CSV'leri paylaşmadan önce kontrol edin. Konu adları araştırmacı tarafından içerik okunarak doğrulanmalıdır.

## 5. NMF ve LDA ile karşılaştırma

**NMF:** TF-IDF matrisini negatif olmayan ağırlıklara ayırır. **LDA:** Sayım temelli olasılıksal konu dağılımları üretir. **BERTopic:** Anlamsal E5 vektörlerinin kümesini ve c-TF-IDF temsilini kullanır. LDA olasılıkları, NMF ağırlıkları ve c-TF-IDF skorları aynı ölçekte değildir; sayısal olarak doğrudan karşılaştırmayın. Konu etiketlerini, örnek belgeleri, NPMI, çeşitlilik ve kararlılık gibi ayrık ölçütleri değerlendirin. Deneysel BERTopic ekranı bu ölçütlerin tamamını henüz üretmez.

## 6. Hafif açık kaynak entegrasyonu

Resmî BERTopic **0.17.4 MIT** wheel dosyasının yayımlanmış SHA-256 değeri `0dbe7ee22c18fb76efe7f4fbf3a14b51c1e67d9fccc569e3f81e75b8f32646d1` doğrulanır. Kaynak Python koduna dokunulmadan yalnızca zorunlu bağımlılık metadata'sı kaldırılarak (resmî belgelerde tarif edilen hafif kurulum ilkesi), NOTICE ve orijinal lisans korunarak wheel üretilir. Üretici betik: `turkmetinlab/tools/prepare_bertopic_wheel.py`. Wheel dosyası `turkmetinlab/staging/vendor/` altında sürümlüdür ve yalnızca bu ONNX+KMeans kullanım senaryosu için onaylanmıştır. Alternatif BERTopic arka uçlarının bu hafif paketle çalışması garanti edilmez.

Belge: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html

## 7. Doğrulama, bellek ve bilimsel sınırlılıklar

GitHub Actions, BERTopic kaynak paketinin kurulumunu ve gerçek BERTopic konu hesaplamasını; ayrıca gerçek model olmayan **küçük bir ONNX test modeli** ile Streamlit dosya yükleme, E5 kodlama, konu ataması ve CSV çıkış akışını sınar. Bunlar yazılım doğrulamasıdır; Türkçe konuların gerçek dünyadaki anlamsal doğruluğunu kanıtlamaz.

Gerçek E5 ağırlıklarıyla en azından eğitim, iklim ve sağlık gibi farklı alanlardan insan uzmanlarının etiketlediği yorumlarda konu tutarlılığı, kararlılık, temsilî belge uyumu ve süre/bellek ölçümleri ayrıca yapılmalıdır. Kısa tweetler, konu sayısının yanlış seçimi, alan kayması ve tekrarlanan metinler sonuçları etkiler. Ücretsiz Streamlit Cloud bellek performansı GitHub runner'ındaki başarıdan çıkarılamaz.

## 8. Uygulama adımları

Konu modelleme bölümünde BERTopic deneysel panelini açın, 10–500 belgeli verinin kimlik ve metin sütununu seçin, konu sayısını belirleyin, sonuçları çalıştırın. Konu–sözcük ve belge–konu tablolarını inceleyip CSV'leri indirin. Araştırma raporunda kullanılan sürümü, E5 modeli, token sınırı, konu sayısı, tohum, veri kümesi boyutu ve temizleme kararlarını açıkça belirtin.
