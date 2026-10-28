# TürkMetinLab 2.5 — staging (deneysel anlamsal analiz)

Bu klasör, canlı uygulamayı değiştirmeden **ayrı bir test uygulaması** çalıştırır.

- `app.py` aynen çalışır. Yalnızca `TML_DENEYSEL_ANLAMSAL=1` iken 🧪 **Anlamsal (deneysel)** sekmesi eklenir.
  `streamlit_app.py` bu bayrağı açar; canlı uygulama bu dosyayı kullanmaz.
- 🧩 Konular bölümünde **BERTopic (E5 ONNX + KMeans + c-TF-IDF)** deneysel yöntemi eklidir; mevcut NMF/LDA korunur.
- BERTopic 0.17.4, doğrulanmış resmi MIT kaynak kodu ve yalnızca düzenlenmiş bağımlılık metadata'sıyla hazırlanan hafif wheel üzerinden kurulur; gereksiz PyTorch, CUDA, UMAP ve HDBSCAN paketleri yüklenmez.
- Model: `intfloat/multilingual-e5-small` (MIT), ONNX FP32, PyTorch yok.
  - INT8 kullanılmaz.
  - Ağırlıklar GitHub'a konmaz. Uygulama, modeli sizin Hugging Face deponuzdan **bir kez** indirir ve SHA-256
    özetiyle doğrular.

## Kurulum (ücretsiz)

1. **Modeli Hugging Face'e yükletin (bir kez):**
   1. Hugging Face → *Settings → Access Tokens*: yazma (*write*) yetkili bir belirteç oluşturun.
   2. GitHub deposu → *Settings → Secrets and variables → Actions*:
      - *Secrets* → `HF_TOKEN` = belirteç.
      - *Variables* → `TML_HF_REPO` = `TahaEser/turkmetinlab-e5-small-onnx`.
      - İsteğe bağlı: `TML_HF_GIZLI` = `true`. Bu durumda depo gizli olur ve uygulamada da `HF_TOKEN` gerekir.
   3. *Actions → ONNX 2.5 (E5-small) → Run workflow*. Dal: `gelistirme-2.5-staging`, kip: `staging`.
   4. İş geçerse artifact'teki `hf_yayin.json` üç değeri içerir. Testler geçmezse yayın yapılmaz.
2. **Streamlit Community Cloud'da ayrı uygulama:**
   1. *Create app* → depo `Tatung40/turkmetinlab`, dal `gelistirme-2.5-staging`.
   2. Giriş dosyası `turkmetinlab/staging/streamlit_app.py`. Adres örneği: `turkmetinlab-staging`.
   3. *Advanced settings*:
      - Python 3.12.
      - *Secrets*:
        ```toml
        TML_E5_ONNX_REPO = "TahaEser/turkmetinlab-e5-small-onnx"
        TML_E5_ONNX_REV = "<hf_yayin.json'daki TML_E5_ONNX_REV>"
        TML_E5_ONNX_SHA256 = "<hf_yayin.json'daki TML_E5_ONNX_SHA256>"
        # HF_TOKEN = "..."   # yalnızca gizli depo için (okuma yetkili belirteç yeterli)
        ```
   4. Bu klasördeki `requirements.txt` kullanılır: 2.1 bağımlılıkları + `onnxruntime` + `tokenizers` + hafif BERTopic.
   5. Canlı uygulama (`main`, `turkmetinlab.streamlit.app`) etkilenmez.

## Kaynak sınırları ve güvenlik

- **Belge sınırı:** En fazla 500 belge (doğrulanan ölçek; `TML_E5_BELGE_SINIRI`).
- **Model ve eşzamanlılık:** Model süreç başına bir kez yüklenir (`st.cache_resource`). Aynı anda tek istek
  çalışır; diğer kullanıcılar sıraya girer.
- **Kodlama:** Toplu iş 8, ilerleme çubuğuyla.
- **Belirteç sınırı:** 512 belirteci aşan belgeler listelenir. Kullanıcı "ilk 512 belirteç" ya da "dışarıda bırak"
  seçeneklerinden birini seçer; sessiz kırpma yapılmaz.
- **Gizlilik:** Vektörler ve sorgular yalnızca oturum belleğinde tutulur. "Verilerimi temizle" bunları da siler.
  Diske yalnızca model dosyaları yazılır.
- **Bellek: Streamlit üzerinde ölçülmedi.** E5-small boyutunda rastgele ağırlıklı bir modelle (gerçek E5 değil),
  2 çekirdekli makinede ölçüldü:

  | Durum (100 belge) | Bellek |
  |---|---|
  | 2.1 uygulaması | ~355 MB |
  | Model yüklendikten sonra | ~1.170 MB |
  | Tepe | ~1.360 MB |

  Ücretsiz planın sınırına yakın ya da üstünde olabilir. Üretime uygunluk ancak staging uygulamasında gözlemle
  karar verilmelidir.

## Testler

- `tests/test_anlamsal.py`, `tests/test_anlamsal_arayuz.py`: küçük sınama modeliyle çalışır; indirme gerektirmez.
- `staging/entegrasyon_testi.py`: gerçek modelle çalışır; GitHub Actions'ta *staging* kipinde otomatik çalışır.
