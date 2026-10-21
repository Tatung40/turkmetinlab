# TürkMetinLab 2.5 — Duygu Analizi Kılavuzu

**Durum: deneysel, staging.** Yazılım testleri, veri kümesine özgü duygu sınıflandırma geçerliğini kanıtlamaz.

## Analiz birimi ve yükleme

CSV veya Excel'de her satır bir yorum olmalıdır. Metin sütunu seçilir, istenirse belge kimliği belirtilir. Tek TXT paragrafı tek belge sayılabilir. Yinelenen metinleri kaldırmak yorum frekansını değiştirir: araştırma amacınıza göre karar verin.

## Sosyal medya temizliği

Duygu Analizi sekmesinde tweet temizliği isteğe bağlıdır. URL ve @kullanıcı adları kaldırılabilir, hashtag işareti kaldırılarak etiket sözcüğü korunabilir. Emojiler varsayılan olarak tutulur; üç yöntem emoji duygularını sistematik olarak yorumladığını iddia etmez. Olumsuzluk bildiren değil, yok, hiç gibi sözcükleri koruyun. Kaynak veri değiştirilmez. Genel Ön İşleme menüsündeki stopword ayarları duygu analizine doğrudan uygulanmaz.

## SWNetTR++

Kaynak: https://github.com/swnettr/SWNetTR. Kaynak yayınlar: Sağlam, Genç ve Sever (2019 ve 2016). Çıktılar: sözlük eşleşmeleri, olumlu/olumsuz eşleşme sayısı, sözcük kapsaması, ortalama sözlük tonu. Ortalama ton = eşleşen sözcük tonu toplamı / eşleşme sayısı. Eşleşmeyen sözcükleri nötr saymayın. Olumsuzluk, ironi ve çok anlamlılık sonuçları yanıltabilir.

## TES-Lex

Kaynak: https://github.com/HandeAkaUymaz/TES-Lex. Kumova Metin ve Aka Uymaz (2025), IT Professional 27(5), 52–58, DOI 10.1109/MITP.2025.3572550. Kaynak Excel dosyası 20.121 kayıt içerir. Sekiz duygu: öfke, neşe, üzüntü, korku, iğrenme, şaşkınlık, beklenti, güven. Polarite, duygu etiketi ve yoğunluk ayrı kayıtlardır. Sözlükte ayrı duygu sütunları seçilmez, temel etiket sütunu seçilir. Ham eşleşmeler ile işlev sözcükleri filtrelenmiş eşleşmeler ayrı gösterilir. Filtreleme duygu doğruluğunu garanti etmez. Duygu katkısı raporundaki yüzdeler yorum yüzdesi değil, eşleşme yüzdesidir. Kaynak README akademik/araştırma kullanım sınırı belirtir; MIT LICENSE ile farklılık nedeniyle diğer kullanımlarda yazarlardan açıklama alınmalıdır.

## ELECTRA-small

Kaynak: https://huggingface.co/ayse-solmaz/turkish-sentiment-electra-small. Model olumlu, nötr veya olumsuz sınıf ve kendi sınıf skorlarını üretir. Skor doğruluk oranı değildir. ONNX INT8 modelinin CPU çıkarımı gerçek model testiyle doğrulanmıştır. Deneysel uygulamada ilk 64 token işlenir; kırpılan belgelerin tam metinleri hakkında sınıf iddiası kurulmaz. Her seferinde en fazla 100 belge işlenir.

## Otomatik bağlam ve kalite kontrolü

Kullanıcıdan iklim, eğitim, sağlık gibi alanlar için özel sözcük listesi istenmez. TES-Lex'in eşleştirdiği bir sözcük birden fazla belgede sık görünüyorsa (varsayılan en az üç belge ve belge oranı en az %20) sistem bunu *incelemeye aday* gösterir. Belge oranı = sözcüğü içeren belge sayısı / toplam belge sayısı × 100. Bu sözcükler **otomatik silinmez**; yüksek yaygınlık duygusuzluk kanıtı değildir.

Belge düzeyinde ayrıca ELECTRA'nın çalıştırılmaması, 64 token kırpması, %65 altı model skoru (kalibre edilmemiş deneysel eşik), düşük SWNetTR++/TES-Lex kapsaması (kalibre edilmemiş %10 eşiği) uyarı olarak raporlanır. Uyarı yokluğu tahminin doğru olduğunu göstermez. ELECTRA üç sınıflı bağlamsal sınıflandırma yaparken TES-Lex sekiz duygu sözlüğüdür: doğrudan aynı ölçekte değillerdir.

## Yeniden üretilebilir raporlama

Her çalışmada veri kaynağı, belge sayısı, temizleme ve filtre ayarları, sözlük/model kaynak sürümleri, kullanılan yöntemler, eksik model sonuçları ve gözlenen kapsamayı kaydedin. İnsan etiketleri olmadan accuracy, Macro-F1 veya geçerlik iddiası yapılmamalıdır. Kapsama oranı duygu doğruluğu değildir. Ham/filtreli duygu tabloları ve uyarı CSV'leri birlikte raporlanmalıdır.
