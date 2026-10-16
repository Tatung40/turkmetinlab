# Çekirdek terim sözlüğü denetimi (2.1)

Sözlük: `tmlab/sozluk/akademik_terimler_cekirdek.csv`
- 637 aday terim, 6 alan.
- SHA-256: `bc673e819c6f104be06e812a2bb0fd2aedd1d6a6e03aabcb9a66a9ccf2558941`.
- Kullanıcının verdiği dosya **değiştirilmeden** kullanılmıştır.

Aşağıdaki kayıtlar uzman incelemesi için **raporlanmıştır**; hiçbiri sessizce düzeltilmemiştir.

Kanıtlar iki ölçümden geliyor:
- 1.397 gerçek DergiPark özetinde (mühendislik ağırlıklı) yapılan sayımlar;
- test yarısının etiketlenmesi.

## 1. Yapı

| Denetim | Sonuç |
|---|---|
| Benzersiz terim / kimlik | 637 / 637 |
| CSV ile Excel'deki *Terimler* sayfası | Birebir aynı |
| Tür | 630 çok sözcüklü, 7 tek sözcüklü |
| Doğrulama durumu | Tamamı "aday - uzman doğrulaması gerekli" |
| Yazım varyantı olan terim | 20; 7'si kısaltma (KTK, MTK, YEM, LLM, NLP, RAG, IoT) |

Alanlara göre terim sayısı:

| Alan | Terim |
|---|---|
| Genel araştırma ve istatistik | 122 |
| Eğitim bilimleri ve psikometri | 114 |
| Mühendislik, bilgisayar ve teknoloji | 104 |
| Psikoloji ve sosyal bilimler | 103 |
| Sağlık bilimleri | 98 |
| Ekonomi, işletme ve yönetim | 96 |

## 2. Gerçek veride anlam ya da alan sorunu görülen kayıtlar

| Kayıt | Alan | Gözlem (DergiPark test yarısı) | Öneri |
|---|---|---|---|
| derin öğrenme (T0020) | Eğitim | 62 geçişin hepsi makine öğrenmesi anlamında. Eğitimdeki "derin öğrenme yaklaşımı" anlamında hiç geçmedi. | Ayrı bir mühendislik kaydı ("derin öğrenme", ML) ve/veya bağlam notu |
| performans değerlendirmesi (T0030) | Eğitim | 6 geçişin hepsi model, denetleyici ya da sistem performansı anlamında | Genel kayıt ve eğitim kaydı ayrılabilir |
| faktör analizi (T0609) | Genel | 1 geçiş: mühendislikte "dinamik (yük) faktör analizi" | — |
| açık veri (T0538) | Genel | "erişime açık veri seti", "halka açık veri tabanları": rastlantısal yan yana gelme | Bağlam gerektirir; kuralla çözülemez |
| YEM (T0104 varyantı) | Eğitim | Küçük harfli "yem" 20 kez geçti, hepsi hayvan yemi anlamında; büyük harfli "YEM" 6 kez geçti, hepsi yapısal eşitlik modeli anlamında | Kısaltmalar büyük/küçük harfe duyarlı ve isteğe bağlı yapıldı |

**Başka bir alandaki daha uzun terimin parçası olan kayıtlar** (uygulama bunlar için kendiliğinden uyarı
verir; bu bir sezgiseldir, her biri gerçek bir belirsizlik değildir):

| Kayıt (alan) | Daha uzun terim (alan) |
|---|---|
| aktif öğrenme (Eğitim) | aktif öğrenme algoritması (Mühendislik) |
| derin öğrenme (Eğitim) | derin öğrenme modeli (Mühendislik) |
| faktör analizi (Genel) | doğrulayıcı faktör analizi (Eğitim) |
| yapay zekâ (Mühendislik) | yapay zekâ okuryazarlığı (Eğitim) |
| lojistik regresyon (Mühendislik) | lojistik regresyon modeli (Genel) |
| öz yeterlik (Psikoloji) | akademik öz yeterlik (Eğitim) |

## 3. Eş anlamlı ya da çok yakın kayıtlar (birleştirilmedi)

| Kayıt 1 | Kayıt 2 |
|---|---|
| temel bileşenler analizi (Mühendislik) | başlıca bileşenler analizi (Genel) |
| yapısal eşitlik modellemesi (Eğitim) | yapısal eşitlik modeli (Genel) |
| lojistik regresyon (Mühendislik) | lojistik regresyon modeli (Genel) |
| maliyet fayda analizi | fayda maliyet analizi |
| literatür taraması | alanyazın taraması |
| benlik belirleme kuramı | öz belirleme kuramı |
| psikolojik dayanıklılık | psikolojik sağlamlık |
| diferansiyel madde işleyişi | değişen madde fonksiyonu |
| psikolojik danışma | psikolojik danışmanlık |

Ek gözlemler:
- **"doğrulayıcı analiz" (Genel)** eksik bir terim gibi görünüyor ("doğrulayıcı faktör analizi" ayrıca var).
- **"öğretmen adayları"** standart biçimi çoğul; tekil "öğretmen adayı" tanınmaz.

## 4. Olası eksik yazım varyantları (eklenmedi)

Kullanıcı bunları arayüzden ya da kendi sözlük dosyasıyla ekleyebilir.

| Sözlükteki biçim | Listede olmayan biçim | Kanıt (DergiPark) |
|---|---|---|
| makine öğrenmesi | **makine öğrenimi** | 15 geçiş (121 "makine öğrenmesi"ye karşı) |
| bilgisayarlı görü | **bilgisayarlı görme** | Test yarısında 2 geçiş |
| transfer öğrenme | **transfer öğrenim** | 1 geçiş |
| veri artırma | **veri artırımı** | 1 geçiş |
| tıbbi görüntüleme | "tıbbi görüntüler" farklı bir addır (görüntüleme ≠ görüntü) | 3 geçiş; doğru olarak eşleşmedi |
| iş gücü … | **işgücü** … | 9 "işgücü", 11 "iş gücü" |
| veri tabanı (vektör veri tabanı) | **veritabanı** | 23 "veritabanı", 32 "veri tabanı" |
| iş birlikli öğrenme | **işbirlikli öğrenme** | Bu derlemde yok; eğitim yazınında yaygın |
| öz düzenleme, öz değerlendirme | **özdüzenleme, özdeğerlendirme** | Yalnızca "öz yeterlik" türü bazı kayıtlarda bitişik varyant var |
| duygusal zekâ, açıklanabilir yapay zekâ… | "zeka" yazımları | Varyant gerekmez: eşleştirme şapkayı (â/a) kural olarak yok sayar |
| Kaplan Meier eğrisi, meta analiz… | Tireli yazımlar | Varyant gerekmez: tire ve boşluk eşleştirmede eşdeğerdir |

## 5. Uzman incelemesi için kontrol listesi (entegrasyon notlarından)

- [ ] Alan uzmanı terimleri ve varyantları gözden geçirdi.
- [x] Çekimli terimlerin eşleştirilmesi ayrı ölçüldü (bilimsel değerlendirme, bölüm 4).
- [x] Yanlış pozitif örnekler testi geçti (yapay küme, UD BOUN, birim testler).
- [x] Kullanıcı eklediği terimi silebiliyor (arayüz testi).
- [x] Oturum gizliliği ve dışa aktarım korunuyor (testler).
- [x] Mevcut 2.0 analizleri regresyon testlerini geçti (312 test).
