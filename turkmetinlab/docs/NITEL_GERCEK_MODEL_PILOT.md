# TürkMetinLab 3.0 — gerçek ücretsiz yerel model pilotu

**Durum:** Deneysel, henüz gerçek model performansı doğrulanmadı. Üretim uygulamasını etkilemez.

## Kullanıcının Windows bilgisayarında (ek API ücreti olmadan)

1. Ollama'yı resmî adresinden kurun: https://ollama.com/download
2. PowerShell'de `ollama pull qwen3:4b` komutunu çalıştırın. Modelin boyutu, bellek ve cihaz uyumu makinenize bağlıdır. Bu bir ön seçimdir, Türkçe nitel analiz başarısı doğrulanmış değildir.
3. Bu geliştirme dalını bilgisayarınıza alın ve `turkmetinlab` alt dizinine geçin.
4. Komutu çalıştırın:

```powershell
python scripts/nitel_gercek_model_pilot.py --model qwen3:4b --input tests/fixtures/nitel_20_gorusme.csv --output nitel_pilot_kodlar.csv
```

5. `nitel_pilot_kodlar.csv` önerilen kod–kategori–tema–kanıtları, `nitel_pilot_kodlar.report.json` frekanslar, kalite uyarıları ve hata kayıtlarını içerir.

**Önemli:** Streamlit Cloud'da çalışan uygulama sizin bilgisayarınızdaki Ollama'ya bağlanamaz; gerçek model testi Ollama ile aynı cihazda çalıştırılmalıdır. Ollama bulut modeli veya bir ücretli servis kullanılmamalıdır. Tamamen yerel çalıştırma için Ollama'nın yalnızca cihazda indirilmiş modelini kullanın.

**Model sınırlamaları:** Qwen3'ün düşünme biçimi JSON dışı yanıt üretebilir. Geçersiz yanıtlar hata olarak raporlanır, kabul edilmez. Bu başarısızlık da test sonucudur. Yerel modelden öneri almak, öneriyi bilimsel olarak doğrulamaz.

**Gizlilik:** Bu sentetik veri test içindir. Gerçek katılımcı metinlerinde kimliksizleştirme ve veri işleme izinleri gerekir. Çıktılar yalnızca araştırmacının yerel dizinine yazılır.
