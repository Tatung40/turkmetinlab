# Yerel nitel analiz mimarisi

## Seçim

Streamlit nitel ekranı ve model sunucusu **aynı kullanıcı bilgisayarında** çalışır.
Akış: tarayıcı → yerel Streamlit → loopback Ollama/llama.cpp → mevcut kanıt doğrulama
→ araştırmacı onayı → frekans/kalite raporu. Mevcut ana uygulama, BERTopic ve E5
değiştirilmez; nitel ekranı `nitel_demo.py` olarak kalır.

| Seçenek | Ücret/kota ve uygulanabilirlik | Karar |
| --- | --- | --- |
| CPU Ollama + yerel Streamlit | Model bir kez indirilir; sonraki çıkarım yerelde, servis sorgu kotası/API bedeli yok. RAM, hız, elektrik ve disk sınırları var. | Başlangıç için temel yol |
| CPU llama.cpp + yerel Streamlit | Yerel Qwen3-4B GGUF ile aynı bağımsızlık; kurulum ve sohbet şablonu daha teknik. | Alternatif |
| Streamlit Community Cloud | Ücretsiz CPU/RAM ve hizmet koşulları değişebilir; 4B çıkarımı veya kotasız kullanım garanti edilemez. Sunucunun localhost'u kullanıcı bilgisayarı değildir. | Yerel modele erişim yolu değil |
| Hugging Face model deposu | Qwen/Qwen3-4B veya güvenilir GGUF ağırlığını bir kez edinmek için kullanılabilir; model lisansı kontrol edilir. | Yalnızca ağırlık kaynağı |
| HF Inference Providers/Spaces/ZeroGPU/PRO | Krediler, kota, kuyruk, süre ve donanım sınırları olabilir. PRO sınırsız ücretsiz API/GPU değildir; yeni abonelik ya da ücret açılmaz. | Temel çıkarım altyapısı değil |

Mevcut çalışma ortamında yetkilendirilmiş ücretsiz Qwen çıkarımı doğrulanamadı.
Önceki socket/proxy/URLError denemeleri tekrarlanmadı. Bu mimari yerel kurulum içindir;
burada gerçek Qwen3 üretimi veya 20 görüş sonucu doğrulanmış değildir.

## Hatalar, güvenilirlik ve gizlilik

- “Yerel modele erişilemedi”: model sunucusu açık mı, adres/port doğru mu,
  Streamlit aynı makinede mi? CPU zaman aşımı dolmuş olabilir. Bulut localhost'u
  için tünel veya herkese açık model portu açmayın.
- HTTP 404: model etiketi veya API yolu yanlış olabilir; mevcut modelleri
  `ollama list` ile kontrol edin. İndirmeyi körlemesine tekrarlamayın.
- Token sınırında kesilen/boş/bozuk yanıt kabul edilmez. /no_think ve JSON isteği
  sunucuya iletilir; bunlar geçerli çıktı garantisi değildir. Düşünme/markdown
  içeren yanıtı sessizce “düzelterek” kanıt saymayız.
- processed denenen, succeeded tüm önerileri doğrulanan (boş liste dahil),
  failed hatalı görüştür. Kısmen geçerli kanıtlar korunur; ilgili görüş failed'dır.
  Hiyerarşi hatası ayrı errors kaydıdır ve CLI başarısız döner.
- CSV birebir quote/start/end içerir; uygulama alıntıyı özgün metinde arar.
  Bu kontrol kanıt–kod anlam uyumunu veya bilimsel geçerliliği garanti etmez.
  Araştırmacı kod kitabını, kategori/tema tutarlılığını, aykırı örnekleri ve
  gerekirse bağımsız kodlayıcı uyumunu incelemelidir. Model adı ve sunucu/model
  sürümü, kuantizasyon, tarih ve ayarlar araştırma günlüğüne kaydedilmelidir.
- JSON'da frekans paydası denenen benzersiz katılımcılardır; başarısız kayıtlar
  sessizce dışlanmaz. Çoklu kodlamada yüzdelerin toplamı %100'ü aşabilir.
  failed/errors varsa tabloyu tamamlanmış 20 görüş sonucu saymayın.
- Loopback adresi zorunlu; uzak/LAN API, yönlendirme ve sistem proxy'si kullanılmaz.
  Ancak yerel sunucunun gerçek ağırlığını veya başka yere veri göndermediğini
  istemci kanıtlayamaz; güvenilir yerel yazılım/model ve işletim sistemi hesabı kullanın.
- Varsayılan kullanım tek kişilik güvenilir bilgisayardır. Ollama genellikle
  uygulama kimlik doğrulaması sağlamaz; loopback diğer yerel süreçlere karşı
  kimlik doğrulama değildir. Portu LAN/Internet'e açmayın. Paylaşılan PC'de
  ayrı işletim sistemi hesapları, dosya izinleri ve ekran kilidi kullanın.
- Streamlit oturumu araştırmacı onayı içerir; kullanıcı kimlik doğrulaması veya
  çok kullanıcılı yetkilendirme sistemi değildir. Merkezi çok kullanıcılı servis
  bu PR'ın kapsamı dışındadır: kimlik doğrulama, kullanıcı/veri izolasyonu,
  kuyruk, kaynak sınırları ve kayıt/saklama politikası olmadan yayımlamayın.
- Görüşleri anonimleştirin. CSV/JSON/Excel çıktıları kişisel veri içerebilir;
  erişimi kısıtlı klasörde saklayın, GitHub'a yüklemeyin. Yerel sunucu günlükleri,
  yedekler ve tarayıcı oturumu da veri barındırabilir. Sunucu sorgu kotası yoktur;
  donanım kapasitesi ve model lisansı geçerlidir.

## Bu PR'da doğrulananlar

33 hedefli test: loopback/proxy/yönlendirme sınırı, güvenli hata mesajları,
kesilmiş/geçersiz yanıt, sayaçlar, ilk kayıt kapısı, hiyerarşi hatasında çıktı
koruma ve Streamlit toplu kodlama/araştırmacı onayı akışı. Tamamı simülasyondur.
Tam regresyon PR CI'ında izlenir; gerçek Qwen3 çıkarımından ayrı değerlendirilir.

## Yinelenen katılımcı kimlikleri

Pilot ve toplu analiz seçili kayıtların tamamında dolu, benzersiz ID ister.
İlk görüş kontrolü açıkken de kontrol bölme ve model çağrısından önce yapılır.
Baş/son boşlukları dışında ID'ler aynen karşılaştırılır. Yinelenen ID altında
aynı metin de farklı metin de olsa tüm işlem reddedilir; hiçbir satır sessizce
silinmez, birleştirilmez veya yeniden numaralandırılmaz. Bu girdi hatası model
başarısızlığı değildir: model çağrısı/sayaç/frekans çıktısı üretilmez. CLI 2 ile
çıkar; giriş CSV'si ve varsa eski çıktı dosyaları değiştirilmez.

Gerçek tekrarları araştırmacı kaynak veriye bakarak çözmelidir. Kazara çoğaltılan
satırların aslı yedekte tutulup karar kayıt altına alınır. Aynı kişinin gerçek,
farklı ifadeleri tek katılımcı kaydında metinleri kaybetmeden bir araya
getirilebilir. Farklı kişilere yanlışlıkla aynı ID verilmişse kimlikler kaynakla
doğrulanarak düzeltilir. Aynı kişiye sırf kontrolü aşmak için farklı ID vermek
katılımcı sayısını/yüzde paydasını bozar; otomatik çözüm uygulanmaz.

Bu düzeltmenin hedefli testleri (pilot + Streamlit toplu akış): 27 geçti.
Sınırın iki tarafındaki ve kalan gruptaki tekrarlar, aynı/farklı metinler,
boşluklu ID, dosyaların korunması ve geçerli 3 kayıtta bir model hatası için
processed/succeeded/failed=3/2/1 ile kod/kategori/tema frekansı 2/3 (%66,67)
doğrulandı. Test yanıtları simülasyondur; gerçek model üretimi değildir.
