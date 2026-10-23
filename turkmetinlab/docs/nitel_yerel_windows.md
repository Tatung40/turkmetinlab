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

