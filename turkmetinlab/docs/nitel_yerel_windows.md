# Windows: yerel Qwen3 4B hızlı başlangıç

Streamlit ve Ollama aynı bilgisayarda çalışır. İlk kurulumdan sonra çıkarım için
API bedeli veya servis sorgu kotası yoktur; bilgisayarın RAM/hız sınırları vardır.
[Mimari, alternatifler ve gizlilik](nitel_yerel_mimari.md).

## Windows'ta kısa kurulum

1. [Python 3.12](https://www.python.org/downloads/windows/) ve
   [Ollama Windows](https://ollama.com/download/windows) kurun. Python kurulumunda
   “Add Python to PATH” seçeneğini işaretleyin. Yeni PowerShell açın.
   Bu PR henüz birleşmediyse GitHub'da `fix/nitel-yerel-cpu` dalını; birleşince
   `gelistirme-3-0-nitel-mvp` dalını ZIP olarak indirin ve açın.
2. PowerShell'de **nitel_demo.py ve requirements.txt bulunan** iç
   `turkmetinlab` klasörüne geçin. Bir kez:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ollama pull qwen3:4b
   ```
   Bunlar ilk kurulum indirmeleridir; her çalıştırmada tekrar yapmayın.
   Ollama uygulaması açıksa ayrıca `ollama serve` başlatmayın; çalışmıyorsa
   ayrı PowerShell'de `ollama serve` kullanın. Cloud model adı/API anahtarı kullanmayın.
3. Aynı bilgisayarda ekranı başlatın:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run nitel_demo.py --server.address 127.0.0.1
   ```
   Tarayıcıda `http://127.0.0.1:8501` açılır. Yerel model ayarları:
   adres `http://127.0.0.1:11434/v1/chat/completions`, model `qwen3:4b`.
   CPU yavaşsa zaman aşımını 600 saniyeye yükseltin. Otomatik tekrar yoktur.
4. Önce örnek tek görüşü CLI ile deneyin:
   ```powershell
   .\.venv\Scripts\python.exe -m scripts.nitel_gercek_model_pilot --model qwen3:4b --limit 1 --timeout 600 --output pilot_1.csv
   ```
   `pilot_1.report.json` içindeki succeeded=1, failed=0, boş errors ve CSV'de
   birebir alıntılı anlamlı kodları kontrol edin. Boş kod yanıtı bağlantı/biçim
   başarısı olabilir; anlamlı nitel analiz başarısı değildir.
5. Tek görüşte kanıtlı üretim doğrulanırsa:
   ```powershell
   .\.venv\Scripts\python.exe -m scripts.nitel_gercek_model_pilot --model qwen3:4b --limit 20 --timeout 600 --output pilot_20.csv
   ```
   Bu çalıştırma kendi ilk kaydını da güvenlik kapısı olarak işler; başarısızsa
   kalan 19'a geçmez. Bu çalıştırma içinde ilk yanıt
   korunarak kalan 19 ile birleştirilir; ardından bir hiyerarşi çağrısı yapılır.
   Ayrı pilot_1 çalıştırmasının önbelleği otomatik kullanılmaz. Yalnızca tek
   çalıştırma isteniyorsa doğrudan --limit 20 de aynı ilk kayıt kapısını uygular.

GPU olmayan bilgisayarda Ollama CPU kullanabilir. Yaklaşık 16 GB sistem RAM'i rahat
bir başlangıç hedefidir; kesin gereksinim kuantizasyon, bağlam ve eşzamanlılığa bağlıdır.
Model yüklenemiyorsa daha fazla indirme/tekrar yerine RAM ve sunucu günlüğünü inceleyin.
CPU'yu açıkça zorlamak için Ollama Modelfile'da `FROM qwen3:4b` ve
`PARAMETER num_gpu 0` satırlarıyla yeni bir yerel model oluşturulabilir; arayüzde o adı seçin.

llama.cpp alternatifi: Windows release'inden `llama-server.exe` ve güvenilir,
lisansı uygun Qwen3-4B GGUF ağırlığı zaten edinilmişse:
```powershell
.\llama-server.exe -m C:\modeller\Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 0 --alias qwen3:4b --jinja -c 8192
```
Adres `http://127.0.0.1:8080/v1/chat/completions` olur; CLI'da --endpoint ile
verilir. Sunucu sürümü Qwen3 sohbet şablonunu ve JSON response_format'ını
desteklemelidir. GGUF dosya adı örnektir. Bu alternatif burada çalıştırılmadı.


## Qwen3.5:4b: düşünme ve tek görüş doğrulaması

Zaten kurulu `qwen3.5:4b` kullanılıyorsa yeniden model indirmeyin. Arayüzde model
adını `qwen3.5:4b` olarak girin; adres aynı /v1/chat/completions adresidir.
Yanıt token sınırı başlangıçta 1024'tür; CPU için zaman aşımı ayrı ayardır.

[Ollama OpenAI uyumluluk belgesi](https://docs.ollama.com/api/openai-compatibility)
ve [kaynak kodundaki](https://github.com/ollama/ollama/blob/main/openai/openai.go)
ThinkingFromReasoningEffort/FromChatRequest dönüşümleri 23 Ekim 2026'da incelendi:
`reasoning_effort: "none"` → `think: false`;
`max_tokens` → `options.num_predict`. Adapter /v1 isteğine yalnızca desteklenen
OpenAI uyumlu alanları gönderir; native /api/chat alanlarını karıştırmaz.
Qwen3.5 için /no_think metnine güvenilmez. Klasik Qwen3 davranışı korunur.

Yerel kurulu Ollama sürümü/modeli bu kontrolü desteklemelidir. Gerekirse
`ollama --version` ve yerel `/api/show` cevabındaki `thinking.values` kontrol edilir:
false bulunması kapatma desteğini gösterir; metadata eksikse destek varsayılmaz.
[Ollama düşünme belgesi](https://docs.ollama.com/capabilities/thinking).
Desteklemeyen sürümde otomatik parametre kaldırma, tekrar veya ücretli API'ye geçiş yoktur.
Bu ortamda kurulu sunucuyla destek/gerçek üretim doğrulanmış değildir.

İlk test (nitel_demo.py bulunan klasörde; mevcut örnek verinin ilk görüşü):
```powershell
.\.venv\Scripts\python.exe -m scripts.nitel_gercek_model_pilot --model qwen3.5:4b --limit 1 --max-tokens 1024 --timeout 600 --output qwen35_tek.csv
```
Kendi veriniz için --input ile katilimci_id,metin sütunlu UTF-8 CSV verin.
Hiyerarşi çağrısının ayrı sınırı `--hierarchy-max-tokens 3072`'dir.
Her iki sınır 128–4096 arasındadır; sınırsız num_predict kullanılmaz.
Otomatik büyütme/tekrar yoktur. Kesilirse düşünme kontrolü ve sürümü incelendikten
sonra yalnızca tek görüş için bilinçli olarak 1536/2048 seçilebilir.

CSV'de kaynakla birebir alıntıları, JSON'da succeeded=1, failed=0, boş errors,
status=validated_proposals ve kaydedilen token sınırlarını inceleyin; kod–kanıt
anlam uyumunu araştırmacı değerlendirsin. no_codes bağlantı/biçim başarısı olabilir,
nitel bulgu değildir. finish_reason=length, boş yanıt, eksik/geçersiz JSON
(Markdown dahil) veya kanıt uyuşmazlığı başarı sayılmaz; yanıt onarılmaz/uydurulmaz.
Tek görüş gerçekten doğrulanmadan 20 görüşe geçmeyin. Bu değişikliğin 59 hedefli
testi sahte transport/yanıt kullanır; gerçek Qwen3.5 sonuçları değildir.
