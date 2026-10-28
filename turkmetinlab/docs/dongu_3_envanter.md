# Döngü 3 — konu modelleme ve duygu analizi kararı (28 Ekim 2026)

Taban: gelistirme-3-0-nitel-mvp `a5db407` (PR #11, #12 birleşik). PR #6/#7 açık, hedefleri
gelistirme-2.5-staging; dokunulmadı. main değişmedi.

| Alan | Karar | Gerekçe |
| --- | --- | --- |
| `konu._model_kur` NMF yakınsama | KÜTÜPHANE ÖZNİTELİĞİ — PR (konu) | Uyarı yakalama yerine `NMF.n_iter_`; sklearn 1.9.1'in uyarı koşulu `n_iter == max_iter and tol > 0` ile aynı. W/H değişmedi (atol 1e−12). "NMF yineleme sayısı" eklendi. |
| LDA perplexity adı | YENİDEN ADLANDIRILDI | Eğitim belgeleri üzerinde hesaplanıyor → "Eğitim şaşkınlığı (perplexity, örneklem içi)". Değer aynı. Batch LDA'da `evaluate_every=-1`, yani yakınsama sınanmaz; sabit yineleme yapılır. |
| NPMI / UMass / çeşitlilik / AJ kararlılığı | KORU (belgeli tanım) | gensim kurulu değil ve ağır. Tanımları farklı: `u_mass` +1 yerine ε=1e−12, `c_npmi` belge yerine kayan pencere kullanır. Sessizce değiştirilmedi; mevcut tanımlar elle hesaplanmış testlerle sınanıyor. |
| BERTopic (E5 + KMeans) / PR #6 | DENEYSEL, değişmedi | KMeans aykırı konu üretmez (HDBSCAN'den farklı); konu sayısı kullanıcıya sabittir. PR #6 staging'e açık; kör birleştirme yok. |
| `electra_duygu.infer` kırpma | KÜTÜPHANEYE TAŞINDI — PR (ELECTRA) | `ids[:64]` dilimi kapanış [SEP] belirtecini düşürüyordu (eğitimdeki girdi biçimine aykırı). `tokenizers.Tokenizer.enable_truncation` özel belirteçleri korur; "Kesildi mi" = `encoding.overflowing`. Kırpılan metinlerin sonucu kasıtlı olarak değişir; kısa metinler birebir aynı. |
| ELECTRA softmax / etiket | `scipy.special.softmax` | Elle yazılan exp/normalize kaldırıldı (1e−4 içinde aynı). Çıktı sayısı ≠ etiket sayısı veya id2label 0..n−1 değilse hata verilir. Her satıra `Model = repo@rev` yazılır. |
| SWNetTR++ / TES-Lex | DENEYSEL gösterge, değişmedi | Zaten "sözlük sinyali, doğrulanmış etiket değil" diye sunuluyor; karar algoritması yok. |

Lisans/kaynak riski: ELECTRA modeli `main` dalından indiriliyor; Hugging Face bu ortamdan erişilemediği
için commit SHA'sı ve lisans doğrulanamadı. Revizyon her satırda kayıtlı, ama sabitleme bir yayın
kapısıdır. SWNetTR++ commit'e sabitli; lisansı yine de ayrıca denetlenmeli.

## Döngü 4 öncelikleri
1. ELECTRA revizyonunu commit SHA'sına sabitlemek ve model kartı lisansını/etiket sırasını erişimi olan bir makinede doğrulamak.
2. `anlamsal.tfidf_sorgu_vektoru` ile eğitilmiş vectorizer'ın `transform` çıktısının eşdeğerlik testi.
3. PR #6'nın kalite karşılaştırmasını, BERTopic'in isteğe bağlı kalmasıyla nitel dala taşıma kararı.
