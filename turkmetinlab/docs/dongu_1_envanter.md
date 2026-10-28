# Döngü 1 — kompakt envanter ve karar (28 Ekim 2026)

## Güncel kaynak

GitHub üzerinden yeniden doğrulandı: main `dfcce491814d91399f12d0ca11939d67b91073b7`;
gelistirme-2.5-staging `77182b3055dc95433d5381be3f6abc7f83f193fc`;
gelistirme-3-0-nitel-mvp `c8aa7149fd6ec748c931a96b20f96dea89d8d7ae`.
PR #6 açık (BERTopic kalite karşılaştırması, 3 dosya); PR #7 açık (nitel çekirdek,
29 dosya); ikisinin de hedefi **gelistirme-2.5-staging**. Hiçbiri birleştirilmedi.
Bu küçük PR'ın tabanı güncel nitel dalıdır; staging/main'e geri taşıma yapılmaz.
Yerel Git work/dfcce49 ve önceki scratch kaynakları güncel dal sayılmadı:
değişiklikler sabitlenen GitHub kaynaklarından üretildi. Bu ağaçta AGENTS.md yok.
Benzerlik/birliktelik çekirdekleri üç dalda aynı blob; nitel modüller yalnız nitel dalında.

## Dosya/fonksiyon kararı

Yollar `turkmetinlab/` altındadır. Karar bir metodoloji onayı değildir.

| Alan / mevcut API | Karar | Gerekçe / kütüphane karşılığı |
| --- | --- | --- |
| `tmlab/preprocess.py: turkce_kucuk_harf, bolutle, on_isle` | KORU | Türkçe I/İ, kesme, bölüt ve çıkarılan sözcük sınırı veri sözleşmesidir; genel tokenizer aynı değildir. |
| `tmlab/lemma.py: Lemmatizer`; stopwords/terimler | KORU | Mevcut simplemma sözlük API'si; bağlamdan bağımsız lemma ölçümleri korpus dışına genellenmez. |
| `tmlab/kalite.py: belgeleri_hazirla, tekil_adlar`; loader | KORU | Eksik veri, yinelenen kayıt, boyut/güvenlik denetimleri algoritma tekrarı değildir. |
| `tmlab/analysis.py: tfidf_hesapla` | KORU | Zaten sklearn TfidfVectorizer; IDF, L2, alt-doğrusal TF ve terim sırası korunur. |
| `analysis.py: belge_ngramlari, _siklik_tablosu, derlem_ozeti` | KORU | Counter; n-gramlar bölüt/None sınırını aşmaz. Genel CountVectorizer'a doğrudan geçiş eşdeğer değil. |
| `tmlab/benzerlik.py: kosinus_matrisi, en_yakin_komsular` iç çarpımı | KÜTÜPHANEYLE DEĞİŞTİR — bu PR | sklearn.metrics.pairwise.linear_kernel. L2 kontrolü, bellek sınırı, clipping, sıfır vektör köşegeni ve belirlenimci top-k adaptörü korunur. |
| Aynı iki fonksiyondaki `X @ X.T` / blok çarpımı, `XT=tocsc()` | KALDIR — bu PR | İki manuel çekirdek hesabı ve artık kullanılmayan transpoz hazırlığı; eski formül yalnız küçük test oracle'ında kalır. |
| `benzerlik.py: hiyerarsik_kumeleme, siluet_tablosu` | KORU | Zaten scipy linkage/fcluster ve sklearn silhouette_score; average/1−cos ile Ward/Öklid ayrıdır. |
| `tmlab/birliktelik.py: _birim_matrisi, _pencere_sayimi, _pmi, kenar_tablosu` | KORU | Belge/bölüt/pencere sayım birimleri ve PMI/Jaccard paydaları yöntemdir; sıradan graf derecesiyle ikame edilmez. |
| `birliktelik.py: dugum_tablosu, _yerlesim` | KORU / sonraki aday | Yerleşim zaten NetworkX; derece/strength için nx.Graph.degree adayı, fakat tekrar kenar/öz-ilmek ve n−1 evreni önce eşlenmeli. |
| `tmlab/konu.py: _vektorlestir, _model_kur` | KORU | Zaten sklearn NMF/LatentDirichletAllocation; TF-IDF ile sayım uzayı aynı model değildir. |
| `konu.py: tutarlilik, cesitlilik, ortalama_jaccard, konu_eslestir` | DENEYSEL kalite yorumu | NPMI/UMass ve sıralı ortalama Jaccard belgeli özel tanımlardır; eşleştirme zaten scipy linear_sum_assignment. Gensim varsayılan coherence/pencere aynı yöntem değil. |
| `tmlab/bertopic_adapter.py: validate, run` | DENEYSEL | Gerçek BERTopic + mevcut E5 + KMeans, UMAP/HDBSCAN varsayılan BERTopic değildir. Ağır model/yeni paket ana requirements'a eklenmez. |
| `tmlab/grup.py: _z_degerleri, permutasyon_esikleri, ayirt_edici_sozcukler` | DENEYSEL çıkarım / KORU sayımlar | Bilgilendirici/sabit önselli log-odds z; test ve permütasyon evreni korunur. statsmodels standart z testi aynı değildir; bu döngüde yeni paket yok. |
| `tmlab/duygu.py: analyze`; teslex.analyze; duygu_sosyal/duygu_baglam | DENEYSEL | Sözlük sinyalleri/bağlam uyarıları; doğrulanmış duygu etiketi değil. Negasyon/ironi/kapsama ve sözlük lisansı denetlenmeden genel sınıflandırıcı vaadi yok. |
| `tmlab/electra_duygu.py: load_model, infer`; duygu_karsilastirma | DENEYSEL | ONNX/tokenizers ve kaynak model/etiket sınırlamaları; karşılaştırma model doğruluğu kanıtı değil. Model indirilmedi. |
| `tmlab/anlamsal.py: model_hazirla, E5Kodlayici` | KORU — isteğe bağlı | Mevcut ONNX + tokenizers, dosya hash/manifest doğrulaması; Transformer/Torch zorunlu değil. |
| `anlamsal.py: znorm, hibrit, tfidf_sorgu_vektoru` | DENEYSEL / sonraki inceleme | z toplamı kalibre olasılık değil; IDF yeniden kurma fitted vectorizer.transform ile ancak evren eşleşirse eşdeğer. scipy.stats.zscore epsilon/NaN kuralını değiştirebilir. |
| `tmlab/nitel_semantik.py: semantic_merge_candidates`; nitel_ucretsiz.suggest_merges | DENEYSEL öneri | Cosine/SequenceMatcher anlam eşitliği kanıtı değildir; onay zorunlu. sklearn cosine_similarity gelecekte eşik sınırı hassasiyet testiyle değerlendirilebilir. |
| `nitel_cekirdek: validate_codes`; nitel_toplu.validate_records; nitel_hiyerarsi; nitel_local_model | KORU | Birebir kanıt/ofset, benzersiz ID, JSON, loopback/proxy/yönlendirme, Qwen3.5 thinking kontrolü ve araştırmacı onayı değiştirilmedi. Gerçek model sonucu üretilmedi. |
| `tmlab/export.py: formul_guvenli, csv_bayt, excel_bayt`; nitel_rapor.frequency_tables; nitel_kalite.audit_hierarchy | KORU | pandas/openpyxl/xlsxwriter + formül güvenliği, benzersiz katılımcı paydası ve tanısal uyarılar korunur. |

## Kütüphane/ortam kararı

- Seçilen gerçek API: [linear_kernel(X, Y=None, dense_output=True)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html).
  Kurulu/pinli sklearn **1.9.1**, metadata lisansı **BSD-3-Clause**, Python >=3.11;
  3.12/3.13 sınıflandırıcıları mevcut. Güncel 1.9.1 resmî dokümantasyon ve kurulu
  imza kontrol edildi; aktif belgelenen mevcut bağımlılık. CI iki Python'u doğrular.
- Çekirdekte NumPy 2.5.3 (BSD ve paketlenmiş bileşen lisansları), SciPy 1.18.1
  (BSD), NetworkX 3.6.1 (BSD-3-Clause), pandas 3.0.5 (BSD), simplemma 2.0.0 (MIT)
  zaten var. Seçim yeni wheel/kurulum, ücret veya GPU gerektirmez.
- Ücretsiz Streamlit maliyeti: yeni bağımlılık/model yok; tam matris hâlâ O(n²),
  2000 belge sınırı ve parça×n komşu belleği korunur. Hız kazancı ölçülmedi/vaat edilmez.
- BERTopic staging wheel'i (0.17.4, depoda hafifletilmiş paket) MIT iddiasıyla
  tutuluyor; upstream ile paket içerik/lisans eşliği ve model/sözlük lisansları
  ayrı yayın kapısıdır. Bu inceleme bunlara genel lisans onayı vermez.
  ONNX/tokenizers staging/dev'de; canlı requirements'a ağır bağımlılık eklenmedi.
  Bu commit'te ölü olduğu kanıtlanan paket yok; kullanılan paketler silinmedi.

## Eşdeğerlik ve sınırlar

Formül aynı: L2-normalize TF-IDF için Kᵢⱼ = xᵢ·xⱼ = cos(xᵢ,xⱼ).
cosine_similarity girdiyi yeniden normalize edeceği için seçilmedi.
L2 toleransı içindeki hafif sapma dahil eski iç çarpım korunur; komşu aday
seçimi/sıralaması yeniden yazılmaz. Sıfır vektörün kendisine benzerliğini 1
gösteren eski ekran sözleşmesi matematiksel cos(0,0) tanımı değil, korunmuş
sunum kuralıdır. Boş eksenler özel olarak korunur; NaN/Inf tam matris girdisi reddedilir.

14 sabit küçük test geçti, 0 atlandı, 0 başarısız (yerel Python 3.12):
eski seyrek çarpım oracle'ı; Türkçe İ/ı/ç/ğ/ş, aynı/ayrık belge, boş belge,
float32/64, k=0/1/3/20, parça=1/2/9, eşitlikte indis sırası, boş eksen ve tek
sıfır satır, L2 toleransı, eksik/sayısal olmayan NaN/Inf. Tolerans float64 için
1e−12, float32 için 2e−7; komşu indisleri tam eşit. Sayısal null vektör,
eksik ham metnin onaylandığı anlamına gelmez; ham veri kalite katmanı korunur.
Yalnız bu hedefli dosya bir kez çalıştırıldı; düzeltme döngüsü gerekmedi.
Yerel tam regresyon/indirme yapılmadı. Mevcut repo CI politikası ayrıca otomatik
tam test tetikleyebilir; workflow değiştirilmedi ve elle yeniden başlatılmayacak.
Bu PR canlıya alınmaz/birleştirilmez; bütünleşik sürüm yayın kapısı ayrı kalır.

## Döngü 2 — en çok üç öncelik

1. Ağ derece/strength için NetworkX eşdeğerlik oracle'ı; kenar/ilmek/evren kuralları.
2. TF-IDF sorgu dönüşümü ile fitted-vectorizer tutarlılığı; NPMI/UMass/AJ ve grup
   çıkarımında yöntem/örnekleme tanımlarının bağımsız sabit veri doğrulaması.
3. BERTopic wheel + sözlük/model lisans ve sürüm kaynağı denetimi; sonra kullanıcı
   bilgisayarında tek görüş Qwen3.5 kanıt testi (simülasyondan ayrı).
