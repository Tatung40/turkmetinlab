# TürkMetinLab 2.5 — Analiz Yöntemleri Ana Kılavuzu

Bu belge, kullanıcıların her analizin amacını, işlem mantığını, çıktısını ve bilimsel sınırını anlayabilmesi için hazırlanmıştır. Staging modülleri araştırma amaçlı deneysel geliştirmelerdir. Ön işleme seçimlerini ve yazılım sürümünü her raporda belirtin.

## Veri yükleme ve analiz birimi
CSV ve Excel'de her satır bir belge veya yanıt olarak kabul edilir. Metin sütununu ve tercihen sabit benzersiz kayıt numarasını seçin. Tek TXT paragrafı tek belge sayılabilir. Yinelenen kayıtları bir kez sayma seçeneği yorumların frekansını etkiler. Boş kayıt ve dil kontrollerini inceleyin.

## Ön işleme
Türkçe büyük/küçük harf (İ/i ve I/ı), Unicode normalleştirme, kesme işareti, URL/e-posta, sayılar, tireli birleşikler, N-gram bölütleri, genel/akademik/İngilizce durak sözcükleri ve isteğe bağlı lemmatizasyon ayrı ayarlardır. Her seçenek araştırma sorusuna göre sonuçları değiştirebilir. Duygu modülleri sosyal metin temizliğini kendi sekmelerinde ayrıca uygular; stopword çıkarmak duyguyu otomatik olarak daha doğru yapmaz.

## Akademik terimler
Çok sözcüklü terimleri tek birim olarak eşleştirir, ekli biçimleri normalleştirir. Kapsama, çakışma ve eşleşmeyen biçimler denetlenir. Aday terimlerin tamamı alan uzmanı tarafından doğrulanmış kabul edilmez. Sözlük açık/kapalı karşılaştırması, önemli sözcüklerin yanlış birleşip birleşmediğini anlamayı sağlar.

## Sözcük sıklığı
Terim frekansı = toplam geçiş sayısı. Belge frekansı = terimin bulunduğu farklı belge sayısı. Bin sözcük başına frekans = 1000 × terim frekansı / işlenen toplam sözcük sayısı. Belge sayısı ve uzunluğu farklı derlemleri ham frekansla kıyaslamayın.

## N-gram
N adet bitişik sözcüğün birlikte görülme sıklığıdır. Örneğin ikili, üçlü, dörtlü sözcük dizileri; noktalama ve çıkarılan sözcüklerin arasından hatalı ikili üretilmemesi için bölütleme kontrol edilir. Yüksek frekans nedensel veya anlamsal önem kanıtı değildir.

## TF-IDF
Belge içi sıklık ile derlemde yaygınlık arasındaki ağırlığı gösterir. Farklı belgeleri ayırt eden terimleri bulmaya yarar; duygu, doğruluk veya nedensellik ölçmez. Tek belgeyle kıyaslamalı IDF anlamlı değildir. Uygulamadaki normalleştirme ve min/max belge sıklığı ayarları raporlanmalıdır.

## Belge benzerliği ve anlamsal arama
TF-IDF kosinüs benzerliği sözcük ağırlıklarının yönünü, E5 embedding modeli semantik vektör yakınlığını kullanır. Hibrit yöntem iki farklı sıralama sinyalini birleştirir. Yüksek benzerlik metinlerin aynı görüşü savunduğunu veya bilimsel olarak aynı sonucu verdiğini kanıtlamaz. E5/hibrit deneysel sürümdedir ve değerlendirme için insan ilgililik etiketleri gerektirir.

## Sözcük ağı
Belgelerde birlikte görünme üzerinden terimler arası bağlantılar kurulabilir. Jaccard ortak görülen belge sayısının en az birini içeren belge sayısına oranıdır. Grafikte merkezîlik ilişkinin istatistiksel önemini, nedenselliği veya duygusal yakınlığı garanti etmez.

## Konu modelleme
NMF ve LDA terim–belge örüntülerinden örtük konu yapıları tahmin eder. Konu sayısı ve ön işleme seçimleri sonucu belirgin etkiler. Konu etiketleri araştırmacı tarafından yorumlanır. NPMI gibi tutarlılık istatistikleri konu geçerliği için tek başına yeterli değildir; kısa yorumlar veya çok küçük veri kümelerinde kararsızlık görülebilir.

## Grup karşılaştırmaları
Önceden mevcut gruplar arasında sözcük/konu göstergelerinin farklılaşması incelenir. Grup büyüklüğü, temel frekanslar, çoklu karşılaştırma ve örneklemin temsiliyeti raporlanır. Gruplarda önemli terim farklılığı, sebep-sonuç ilişkisi göstermez.

## Kelime bulutu
Sıklık veya TF-IDF'nin görsel özetidir. Buluttaki konum ve renk niceliksel bir ölçü değildir. Grafik yerine tablolardaki mutlak sayıları ve ölçekleri raporlamak gerekir.

## Duygu analizi
Üç yöntem ayrı değerlendirilir: SWNetTR++ sözcük tabanlı polarite/ton; TES-Lex sekiz duyguyla ilişkili sözlük kanıtı; ELECTRA-small olumlu/nötr/olumsuz kısa metin sınıflandırması. Ham/filtreli TES-Lex ve derlemde yaygın kanıt sözcükleri ayrı raporlanır. Yaygın sözcükler otomatik silinmez. Bağlamsal model için 64 token kesilme, model skorları ve kapsama uyarıları gösterilir. Ayrıntılı formüller, kullanım koşulları, raporlama ve sınırlar için [Duygu Analizi Kılavuzu](DUYGU_ANALIZI_KILAVUZU_2_5.md).

## Çıktılar, gizlilik ve yeniden üretilebilirlik
Excel/CSV dışa aktarmalarda veri ve ayarları eşleyin. Genel analizlerin bazıları temizlenmiş metin üzerinden, duygu modülleri özgün/sosyal medya temizlenmiş metin üzerinden çalışır; bunları yöntem bölümünde ayırın. Model çıktılarından kullanıcıların psikolojik durumu, gerçek niyeti veya insan duygu yoğunluğu hakkında doğrulanmamış çıkarımlar yapmayın. Deneysel metrik eşikleri kalibre edilmemiştir. Farklı araştırma alanlarına genelleme için bağımsız etiketli test gerekir.
