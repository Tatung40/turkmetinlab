# TürkMetinLab 2.5 — Anlamsal model fizibilite çalışması

Bu klasör TürkMetinLab uygulamasının kodunu **değiştirmez**. Uygulamanın ön işlemesini (`tmlab`) yalnızca okuyarak
kullanır. Sonuç raporu: [`RAPOR.md`](RAPOR.md).

## Karşılaştırılan yöntemler

| Kod adı | Yöntem | Metin |
|---|---|---|
| `tfidf_ham` | TF-IDF + kosinüs | Özgün metin (yalnızca Türkçe küçük harf) |
| `tfidf_tml` | **Referans:** TürkMetinLab varsayılan ön işleme + TF-IDF | Durak sözcükler çıkarılmış |
| `tfidf_tml_terim` | Referans + 637 terimlik sözlük (tek birim) | |
| `tfidf_tml_lemma` | Referans + lemmatizasyon | |
| `minilm_ozgun` / `minilm_temiz` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) | Özgün / TürkMetinLab ile temizlenmiş |
| `e5_ozgun` / `e5_temiz` | `intfloat/multilingual-e5-small` (MIT) | Özgün / temizlenmiş |
| `hibrit_…` | Referans TF-IDF ve gömme skorlarının sorgu başına z-puanı ortalaması | |

Gömme ayarları:
- Yalnızca CPU; vektörler L2 ile normalleştirilir.
- Havuzlama ve azami token sayısı modelin kendi dosyalarından okunur ve sonuç dosyasına yazılır.
- E5 önekleri: başlık sorgusu `query: `, özet `passage: `; özet → özet (simetrik) görevinde iki taraf da `query: `.

## Değerlendirme verisi ve etiketler

- **Kaynak:** [DergiParkEn](https://github.com/humakilicunlu/DergiParkEn) `TRAll.txt`. Türkçe makale özetleri,
  ağırlıklı olarak mühendislik alanında.
  - Lisansı belirtilmemiştir; bu yüzden pakete konmaz, çalışırken indirilir ve diske yazılmaz.
  - 1.395 uygun benzersiz özetten sabit tohumla (2501) 500 (ve 100) özet seçilir.
- **Görev A — başlık → özet:** Her makalenin başlığı sorgudur; tek ilgili belge aynı makalenin özetidir. Etiket,
  yayıncının başlık-özet eşleşmesidir: insan kaynaklıdır, otomatik değildir.
  - **Uyarı:** Başlık ile özet sözcük paylaştığı için bu görev sözcük eşleşmesine dayanan yöntemleri kayırır.
- **Görev B — özet → özet:** İlgililik, iki makalenin yazar anahtar sözcüklerinden ortak olanların sayısıdır
  (0, 1, 2…).
  - Yazarlar tarafından üretilmiştir; ancak doğrudan bir **ilgililik yargısı değildir** (dolaylı ve gürültülü
    etiket).
  - n = 500'de 269 sorgunun en az bir ilgili belgesi var.
- **Aranan ama bulunamayan veri:** İnsan tarafından ilgililik yargısı verilmiş, erişilebilir Türkçe akademik bir
  erişim veri seti bu çalışma ortamında bulunamadı. Hugging Face'teki veri setlerine buradan erişilemiyor (403).

## Ölçütler

- **Görev A:** İsabet@1, İsabet@5, MRR, nDCG@5. Tek ilgili belge olduğu için İsabet@k, Recall@k ile aynıdır.
- **Görev B:** İsabet@1 ve İsabet@5 (ilk k sonuçta en az bir ilgili belge), MRR (ilk ilgili belge), nDCG@5
  (derecelendirilmiş, kazanç 2^ilgililik − 1).
- **Güven aralıkları:** Sorgular üzerinden %95 bootstrap güven aralığı (2.000 yeniden örnekleme). Referansa göre
  farklar eşleştirilmiş olarak hesaplanır.
- **Kaynak ölçümleri:**
  - model yükleme süresi;
  - 100 ve tüm belgeleri kodlama süresi;
  - tepe RAM (her yöntem ayrı süreçte, `VmHWM`);
  - vektör boyutu;
  - disk kullanımı;
  - token kesilme oranları;
  - 3 eşzamanlı oturum benzetimi (paylaşılan model, iş parçacıkları).

## Önceden belirlenen karar ölçütleri

Bu ölçütler model sonuçları görülmeden yazılmıştır. Bir modelin 2.5'e alınabilmesi için **hepsi** sağlanmalıdır:

1. **Bilimsel katkı:** Görev B'de nDCG@5 farkı (model − en iyi TF-IDF) **≥ +0,03** olmalı ve %95 güven aralığı
   0'ı içermemeli.
2. **Kayıp olmaması:** Görev A'da nDCG@5 farkının güven aralığının alt sınırı **> −0,03** olmalı.
3. **Bellek:** Uygulamanın 2.1'deki tepe kullanımına eklenen bellek ile birlikte toplam, 3 eşzamanlı oturumda
   **690 MB'ı** (ücretsiz planda daha önce kullandığımız alt sınır) aşmamalı.
4. **Süre:** 500 özet bu ölçüm makinesinde CPU'da **≤ 60 sn** içinde kodlanmalı. Sunucu daha yavaş olabilir.
5. **Lisans ve kurulum:** Lisans serbest olmalı (Apache-2.0 ya da MIT). GPU paketleri olmadan kurulabilmeli.

## GitHub Actions ile çalıştırma (önerilen; ücretsiz, CPU)

İş akışı: `.github/workflows/fizibilite-2.5.yml`. **Yalnızca elle başlatılır** (`workflow_dispatch`).

Akış:
1. Python 3.12 kurulur.
2. torch, **resmî PyTorch CPU dizininden** kurulur.
3. Ortam denetimi yapılır. CPU'ya özel torch değilse ya da NVIDIA/CUDA paketi varsa iş durur.
4. Veri, işe özel geçici klasöre indirilir.
5. Modeller **tek denemeyle** indirilir ve Actions önbelleğinde saklanır.
6. **Aşama 1 (100 belge)** çalışır. Yalnızca başarılıysa **aşama 2 (500 belge)** çalışır.

Durdurma kuralları:
- Yöntem başına süre ve bellek sınırı vardır (900/1500 sn, 6.000 MB). Sınır aşılırsa ya da model yüklenemezse
  iş 2 çıkış koduyla durur.
- İşin tamamı en fazla 90 dakika sürebilir.
- Yeniden deneme yoktur (yalnızca veri indirmede en çok 2 deneme).

Çıktılar:
- JSON ve Markdown sonuçlar, `fizibilite-2.5-sonuclar` adlı artifact olarak 30 gün saklanır. Model ağırlıkları
  artifact'e girmez.
- Ölçümler GitHub'ın makinesinde yapılır; **Streamlit sunucu ölçümü değildir.**

> **GitHub kısıtı:** GitHub belgelerine göre `workflow_dispatch` ile başlatılan bir iş akışının dosyası
> **varsayılan dalda (`main`) da bulunmalıdır**. Ancak çalıştırırken dal seçilebilir.
> - Dosya yalnızca `gelistirme-2.5-fizibilite` dalında olursa "Run workflow" düğmesi görünmez.
> - Bu dosyayı `main`'e eklemek uygulamayı ve varsayılan testleri değiştirmez, çünkü yalnızca elle başlar.
>   Ama bu, `main`'e yapılan bir değişikliktir ve sizin onayınızı gerektirir.

## Yeniden üretim (kendi bilgisayarınızda)

```bash
pip install -r requirements.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r fizibilite_2.5/requirements-fizibilite.txt
python fizibilite_2.5/anlamsal_karsilastirma.py --n 100     # önce küçük örneklem
python fizibilite_2.5/anlamsal_karsilastirma.py --n 500     # asıl karşılaştırma
```

- Model dosyaları ilk çalıştırmada bir kez `~/.cache/turkmetinlab_modeller` klasörüne indirilir (her model
  yaklaşık 0,5 GB). Bu önbellekte kullanıcı metni bulunmaz.
- Sonuçlar `fizibilite_2.5/sonuclar/` klasörüne JSON ve Markdown olarak yazılır.
- Kendi bilgisayarınızda çalıştırın: kullanıcı metinleri hiçbir dış hizmete gönderilmez.
- Yalnızca TF-IDF ölçümlerini çalıştırmak için: `--yontemler tfidf_ham,tfidf_tml,tfidf_tml_terim,tfidf_tml_lemma`.
- Modeller başka bir yerde indirilmişse: `--model-yolu minilm=/yol,e5=/yol`.
