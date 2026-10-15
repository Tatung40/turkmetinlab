# TürkMetinLab 2.0 — Bilimsel Validasyon ve Performans Raporu

Bu belge, 2.0 sürümüne eklenen dört analiz modülünün nasıl doğrulandığını anlatır. Doğrulamanın üç
katmanı vardır:

1. Elle hesaplanmış küçük örnekler.
2. Bağımsız uygulamalarla karşılaştırma.
3. Gerçek akademik veriyle yöntemsel sınama.

1.1 sürümünün değerlendirmesi için `docs/bilimsel_degerlendirme_1.1.md` dosyasına bakın; o sürümün bütün
özellikleri ve testleri korunmuştur.

Tüm sayılar bu sürüm geliştirilirken gerçekten çalıştırılan ölçümlerden alınmıştır. Yeniden üretmek için:

```bash
python degerlendirme/v2_dogrulama.py                       # gerçek veriyle validasyon (internet gerekir)
python degerlendirme/performans_olcumu.py --modul orta_genis  # modül performansı
python -m pytest -q                                         # otomatik testler
```

## Kullanılan gerçek veriler

| Veri | İçerik | Kullanım | Lisans / dağıtım |
|---|---|---|---|
| **DergiPark Türkçe özetleri** (Kılıç Ünlü & Çetin, 2023; github.com/humakilicunlu/DergiParkEn) | DergiPark'ta yayımlanmış 1.417 Türkçe makalenin başlığı, özeti ve **yazarların verdiği anahtar sözcükler** (ağırlıklı olarak mühendislik) | Dört modülün gerçek akademik veride sınanması; anahtar sözcükler analizden bağımsız bir **dış ölçüt** olarak kullanıldı | Lisans belirtilmediği için **projeye eklenmedi**; betik, veriyi çalışırken belleğe indirir |
| **UD Turkish-BOUN** (testlerde 400 cümle) | Türü bilinen gerçek Türkçe metinler (biyografi, deneme, bilgilendirici, haber, popüler) | Bağımsız uygulamalarla karşılaştırmalar, otomatik testler | CC BY-SA 4.0, `tests/veri` |

DergiPark özetlerinin 70'i (%4,9) İngilizce özet de içeriyordu. Bu, Türkçe akademik verilerde sık
görülen ve konu modellerini bozan bir durumdur; bu yüzden 2.0'a İngilizce içerik tespiti eklenmiştir
(Bölüm 5). Konu, grup ve ağ doğrulamalarında "İngilizce durak sözcükleri de çıkar" seçeneği açıktır.

---

## Modül 1 — Belge benzerliği

**Yöntem.** TF-IDF sekmesindeki ayarlarla üretilen ve uzunluğu 1'e normalleştirilen vektörler
kullanılır; bu durumda kosinüs benzerliği iki vektörün iç çarpımına eşittir.
- **Tam matris:** 2.000 belgeye kadar hesaplanır; ısı haritası 150 belgeye, dendrogram 200 belgeye kadar çizilir.
- **Büyük derlemler:** Satırlar parça parça işlenerek her belgenin en yakın k komşusu bulunur. Bellek
  kullanımı en fazla parça × belge sayısı kadardır ve sonuç tam matrisle birebir aynıdır.
- **Kümeleme:**
  - Ortalama bağlantı: uzaklık = 1 − kosinüs.
  - Ward: vektörler normalleştirilmiş olduğu için ‖a−b‖ = √(2 − 2·cos) eşitliğiyle hesaplanır.
  - Küme sayısı seçimine yardım için siluet değerleri gösterilir.

**Doğrulama.**

| Sınama | Sonuç |
|---|---|
| Elle hesaplanan örnek: cos(A,B) = idf(elma)² / (idf(elma)² + idf(armut)²) | 0,36644, birebir aynı |
| scikit-learn `cosine_similarity` ve saf Python döngüsü (UD verisi, 80 belge) | fark < 10⁻¹² |
| Parçalı en yakın komşu ile tam sıralama (25 rastgele, eşitlik ağırlıklı durum) | 25/25 aynı |
| Ortalama bağlantı ↔ scipy `linkage(pdist(·,'cosine'),'average')` | fark < 10⁻⁹ |
| Ward (kosinüsten türetilmiş uzaklık) ↔ scipy `linkage(X,'ward')` | fark < 10⁻⁹ |

**Gerçek akademik veride anlamlılık (DergiPark, 1.417 özet, 1.003.236 belge çifti).**

| Ölçüt | Değer |
|---|---|
| Ortak yazar anahtar sözcüğü olan çiftlerin ortalama benzerliği | 0,058 (olmayanlar: 0,012) |
| Bu iki grubu ayırma başarısı (ROC AUC) | **0,865** |
| Makale başlığıyla sorgulanınca kendi özetinin 1. sırada gelmesi | **%84,8** (rastgele beklenti: %0,07) |
| Kendi özetinin ilk 5 sırada gelmesi | %96,0 |

**Olumsuz bulgu (raporlanması gereken).** UD verisinde metin türüne (haber, biyografi, deneme…)
göre kümeleme rastgele düzeyin çok az üstündedir. Düzeltilmiş Rand indeksi ortalama bağlantıda 0,028,
Ward'da 0,002; rastgele atamada −0,001'dir. Bu beklenen bir sonuçtur: TF-IDF benzerliği **konuyu**
yakalar, **üslubu ya da türü** yakalamaz. Arayüzdeki uyarı metni bu nedenle benzerliğin anlamsal
eşdeğerlik, intihal ya da metin kökeni tespiti olmadığını açıkça belirtir.

---

## Modül 2 — Sözcük birliktelik ağları

**Yöntem.** Üç bağlam türü vardır:
- **Pencere (±k sözcük):** Uzaklık özgün metindeki konumlara göre ölçülür; çıkarılan sözcükler de yer
  kaplar. İstenirse pencere cümle sınırını aşmaz.
- **Aynı cümle**
- **Aynı belge**

Ölçütler:
- **Ham birlikte görülme sayısı**
- **PPMI** (Church & Hanks, 1990; Levy vd., 2015)
- **Jaccard**

Merkezîlik ölçütleri: derece, normalleştirilmiş derece ve ağırlıklı derece. Hesap NumPy ile vektörel
yapılır ve en sık 1.000 sözcükle (en fazla 2.000) sınırlıdır. Ağda en fazla 150 düğüm ve 500 kenar
gösterilir; dışa aktarım en güçlü 100.000 kenarla sınırlıdır. Ağ yerleşimi belirlenimcidir: her bağlı
bileşen ayrı yerleştirilir (Kamada–Kawai) ve bileşenler bir ızgaraya dizilir.

**Doğrulama.**

| Sınama | Sonuç |
|---|---|
| Elle: pencere genişliği, çıkarılan sözcüğün yer kaplaması, cümle/belge sınırları, PMI = log₂(2·6/(3·2)) = 1, Jaccard = 2/3 | hepsi aynı |
| Kaba kuvvet (her konum çiftini tek tek sayan) uygulama ile vektörel sayım, 6 bağlam ayarı, UD verisi | 6/6 birebir aynı |
| PPMI ve Jaccard ↔ bağımsız formül (300 sözcük, tüm kenarlar) | fark < 10⁻⁵ (6 basamak yuvarlama) |
| Derece, ağırlıklı derece ve derece merkezîliği ↔ networkx | birebir aynı |

**Gerçek akademik veride anlamlılık.** Dış ölçüt olarak yazarların verdiği **çok sözcüklü anahtar
terimler** kullanıldı (ör. "yapay sinir ağları"). Aynı terimde geçen sözcük çiftleri olumlu kabul edildi
ve her ölçütün bu çiftleri diğerlerinden ne kadar iyi ayırdığı ölçüldü.

| Bağlam | AUC: Jaccard | AUC: PPMI | AUC: ham sayı | PPMI ilk 100 içinde terim çifti | Tüm kenarlarda terim çifti oranı |
|---|---|---|---|---|---|
| Pencere ±5, cümle içinde | **0,740** | 0,721 | 0,663 | %33 | %8,2 |
| Pencere ±2, cümle içinde | **0,695** | 0,677 | 0,642 | %33 | %14,2 |
| Aynı cümle | **0,775** | 0,751 | 0,656 | %35 | %5,0 |
| Aynı belge | **0,735** | 0,708 | 0,627 | %20 | %0,6 |

Normalleştirilmiş ölçütler ham sayımdan tutarlı biçimde daha iyidir. Jaccard dört bağlamın dördünde
de en yüksek değeri verdi; bu nedenle **varsayılan ölçüt Jaccard** olarak seçildi. PPMI'ya göre en
güçlü çiftler gerçek teknik terimlerdir: kripto para, taramalı elektron mikroskobu, nesnelerin interneti,
tedarik zinciri, naive Bayes, atıksu arıtma, Fourier dönüşümü.

Akademik özetlerde ağın merkezini "elde edilmiştir", "kullanılarak" gibi kalıp ifadeler kapladığı
gözlendi. Bu nedenle akademik kalıp sözcük listesi gerçek veride sık görülen 9 kalıp fiille
genişletildi; arayüz de bu seçeneği önerir.

---

## Modül 3 — Konu modellemesi

**Yöntem.**
- **LDA:** CountVectorizer ile sözcük sayıları; toplu varyasyonel çıkarım; α = η = 1/k; 30 yineleme.
  Gerçek veride tutarlılık 30 yinelemede en yüksek değerine ulaştı; daha fazla yineleme yalnızca süreyi uzattı.
- **NMF:** TF-IDF girdisi; NNDSVDa başlangıcı; Frobenius kaybı.
- Her iki yöntemde `random_state` sabittir.

Değerlendirme ölçütleri:
- **NPMI tutarlılığı** (Aletras & Stevenson, 2013; Lau vd., 2014)
- **UMass tutarlılığı** (Mimno vd., 2011)
- **Konu çeşitliliği** (Dieng vd., 2020)
- **Kararlılık** (Greene vd., 2014): Model belgelerin %80'iyle yeniden kurulur, konular Macar
  algoritmasıyla eşleştirilir ve en üstteki sözcük listelerinin Ortalama Jaccard benzerliği hesaplanır.

Farklı konu sayıları karşılaştırılabilir. Küçük derlem, kısa belge ve yüksek konu sayısı için uyarılar
verilir. Konuları araştırmacı adlandırır; adlar tablolara ve indirilen dosyalara yansır.

**Doğrulama.**

| Sınama | Sonuç |
|---|---|
| Elle: NPMI ve UMass (4 belgelik örnek) | fark < 10⁻¹² |
| Bağımsız saf Python tutarlılık uygulaması (60 belge, 3 konu) | fark < 10⁻¹² |
| Ortalama Jaccard: AJ([a,b,c], [a,c,b]) = 7/9; Macar eşleştirmesi | aynı |
| **Ekilmiş konular:** 4 bilinen konu, 240 belge, Zipf dağılımlı sözcükler (LDA ve NMF) | her gerçek konu doğru bulundu (ilk 10 sözcüğün 10'u konudan, en az 8'i gerçek ilk 10'dan); ARI > 0,9; kararlılık > 0,8 |
| Aynı ayarlarla iki çalıştırma | birebir aynı |

**Gerçek akademik veride karşılaştırma (DergiPark, 1.417 özet).** Dış ölçüt: bir belgenin baskın
konusunun ilk 10 sözcüğünden en az biri, o belgenin yazar anahtar sözcükleri arasında mı? Taban
çizgisi, aynı konular belgelere rastgele (permütasyonla) atandığında beklenen örtüşmedir.

| Yöntem | k | NPMI | UMass | Çeşitlilik | Kararlılık | Anahtar sözcük örtüşmesi | Rastgele | Süre |
|---|---|---|---|---|---|---|---|---|
| NMF | 5 | 0,219 | −1,72 | 0,90 | **0,81** | %38,2 | %11,5 | 1,0 sn |
| NMF | 10 | 0,241 | −1,67 | 0,86 | 0,62 | %42,6 | %9,2 | 1,7 sn |
| NMF | 15 | 0,258 | −1,76 | 0,85 | 0,58 | %50,2 | %8,4 | 2,1 sn |
| NMF | 20 | **0,263** | −1,77 | 0,83 | 0,67 | **%51,4** | %7,4 | 3,3 sn |
| LDA | 5 | 0,079 | −1,54 | 0,66 | 0,48 | %20,4 | %8,3 | 33 sn |
| LDA | 10 | 0,061 | −1,92 | 0,65 | 0,41 | %31,6 | %10,8 | 31 sn |
| LDA | 15 | 0,085 | −1,88 | 0,59 | 0,39 | %31,9 | %8,7 | 32 sn |
| LDA | 20 | 0,067 | −1,92 | 0,57 | 0,37 | %32,2 | %7,7 | 31 sn |

(Süreler 3 kararlılık tekrarı dahil, 2 çekirdek.)

Bu kısa akademik metinlerde (ortanca ~150 sözcük) NMF; tutarlılık, kararlılık, dış ölçütle örtüşme ve
hız bakımından LDA'dan açıkça üstündür. Bu bulgu, LDA'nın kısa metinlerde zorlandığı bilgisiyle de
tutarlıdır. Bu nedenle **varsayılan yöntem NMF**'dir. LDA seçenek olarak durur; kararlılığı 0,5'in
altına düştüğünde arayüz uyarır.

**Ölçütle ilgili not.** Ekilmiş konular sözcükleri eşit olasılıklı üretildiğinde, konular doğru
bulunsa bile kararlılık 0,68'e düştü. Ortalama Jaccard **sıraya duyarlıdır**; eşit ağırlıklı
sözcüklerin sırası rastgele değişir. Bu yüzden kararlılık, ağırlıkları düz olan konularda gerçek
kararlılığı olduğundan düşük gösterebilir.

---

## Modül 4 — Grup karşılaştırması

**Yöntem.** Monroe, Colaresi & Quinn (2008) "Fightin' Words" yöntemi kullanılır: Dirichlet önselli
log-odds oranı ve z değeri. Her grup, geri kalan grupların birleşimiyle karşılaştırılır. İki önsel
seçeneği vardır:
- **Bilgilendirici** (varsayılan): α_w = derlem sıklığı.
- **Zayıf:** α_w = 0,01.

Gruplara göre ham ve bin sözcüğe göre normalleştirilmiş sıklıklar da verilir. Ayrıca her sözcük için
gruptaki belge oranı ve tek bir belgeden gelen en büyük pay gösterilir; böylece tek bir belgenin
sürüklediği sonuçlar fark edilebilir.

**Neden permütasyon?** z formülü sözcükleri birbirinden bağımsız sayar. Oysa aynı belgedeki sözcükler
bağımlıdır ve binlerce sözcük aynı anda sınanır. Bu nedenle grup etiketleri belgeler arasında (grup
büyüklükleri korunarak) rastgele karıştırılır. Her grubun en büyük z değerinin %95'lik dilimi, o veri
için "tesadüfen aşılmayan" eşik olarak raporlanır; bu, aile bazında hata kontrolü sağlar.

**Doğrulama.**

| Sınama | Sonuç |
|---|---|
| Elle (zayıf önsel): δ = 2·ln(3,01/1,01), σ² = 1/3,01 + 1/1,01 | aynı |
| Elle (bilgilendirici önsel): δ = 2·ln(7/5), σ² = 1/7 + 1/5 | aynı |
| **jmhessel/FightingWords** özgün kodu, UD verisi, 1.185 sözcük, iki önsel | fark < 5·10⁻⁵ (referans 32 bit sayı kullanır) |
| 3 grupta "haber" sonucu ↔ diğer iki grup birleştirilerek yapılan iki grup karşılaştırması | birebir aynı |
| İki grupta simetri (z_A = −z_B) | birebir |
| Permütasyonun belirlenimciliği (sabit tohum) | aynı |

**Gerçek akademik veride kalibrasyon (DergiPark).** 30 anlamsız, rastgele 2 grup bölmesi yapıldı;
her birinde 100 permütasyon kullanıldı.

| Ölçüt | Bilgilendirici önsel | Zayıf önsel | Beklenti |
|---|---|---|---|
| \|z\| > 1,96 olan sözcük oranı | %0,85 (tutucu) | %6,2 (hafif şişkin) | %5 |
| Permütasyon eşiğini en az bir sözcüğün aştığı bölme oranı | %10 | %10 | ≈ %10 (2 grup × %5) |
| Bölme başına eşiği aşan sözcük (ortalama) | 0,1 | 0,1 | ≈ 0 |

Geleneksel |z| > 1,96 eşiği önsele göre ya çok tutucu ya da hafif gevşek kalıyor. Permütasyon eşiği
ise iki önselde de tam hedeflenen düzeyde çalıştı.

**Pozitif kontrol.** Yazar anahtar sözcüklerine göre iki gerçek konu grubu ayrıldı: deprem (19 makale)
ve enerji (113 makale).
- **Deprem grubu:** deprem (z = 7,8*), sismik (4,6*), meydana (4,2*), zemin (3,9*), ivme, yapısal,
  fayı, kayma, spektral.
- **Enerji grubu:** enerji (3,1*), enerjisi*, güneş*, elektrik*, rüzgâr, güç, yenilenebilir, fotovoltaik.

(* = permütasyon eşiğini aşıyor; her grupta 4 sözcük.) Küçük ve dengesiz gruplarda z değerlerinin
düşük çıkması (ör. enerji grubunun en yüksek z değeri 3,1) beklenen bir durumdur ve arayüzde
uyarılır.

---

## 5. Veri kalitesi: İngilizce içerik tespiti (yeni)

En sık 15 İngilizce işlev sözcüğünün (the, of, and…) metindeki oranı hesaplanır; %8'in üzerindeki
belgeler, satır numaralarıyla birlikte bildirilir. Türkçede de sözcük olan kısa biçimler (on, an, as,
in) yanlış alarmı önlemek için dışarıda bırakılmıştır.
- **DergiPark:** 70 belge işaretlendi. Hepsinde gerçekten İngilizce özet olduğu gözle kontrol edildi.
- **Yanlış alarm:** UD BOUN (5 cümlelik 80 belge) ve örnek veri setinde 0.

İsteğe bağlı "İngilizce durak sözcükleri de çıkar" seçeneği eklendi. Türkçede anlamlı olan biçimler
(can, his, but, had) bu listeye alınmamıştır.

---

## 6. Performans ve eşzamanlı kullanım

**Ölçüm koşulları.** Ölçümler 2 çekirdekli bir makinede, her biri ayrı bir süreçte yapıldı; tepe bellek
için `VmHWM` kullanıldı. Derlemler gerçekçi sözcük dağarcıklıdır (Zipf dağılımı).

**Modüllerin süresi:**

| Modül | Orta derlem (288 bin sözcük, 2.000 belge) | Büyük derlem (2,9 milyon sözcük, 20.000 belge) |
|---|---|---|
| Benzerlik (+ kümeleme ve siluet ortada; en yakın komşu büyükte) | 1,3 sn | 28 sn |
| Birliktelik ağı (1.000 aday sözcük) | 0,5 sn | 3,4 sn |
| Konu modeli (NMF, 10 konu, 3 kararlılık tekrarı) | 5,9 sn | 36 sn |
| Grup karşılaştırması (4 grup, 200 permütasyon) | 1,3 sn | 12 sn |
| **Tepe bellek (derlem dahil)** | **363 MB** | **823 MB** |

Büyük derlem, uygulamanın ücretsiz buluttaki 1,5 milyon sözcük sınırının üzerindedir; yalnızca sınırları
sınamak için ölçülmüştür. LDA yavaştır (orta derlemde 21 sn, 1,4 milyon sözcükte 162 sn); bu yüzden
bulutta 500 bin sözcükle sınırlandı ve çalıştırmadan önce tahmini süre gösterilir.

**Önlemler:**
- **Uzun analizler düğmeyle başlar:** konu modeli, konu sayısı karşılaştırması ve 2.000 belgeden büyük
  derlemlerde en yakın komşu hesabı.
- **Büyük derlemde ağ ve grup analizleri:** 300 bin sözcüğün üstündeki derlemlerde bu iki analiz de
  düğmeyle başlar; böylece sayfa açılışı hızlı kalır.
- **Sonuçların saklanması:** Tüm sonuçlar yalnızca o kullanıcının oturumunda tutulur.

**Eşzamanlı kullanıcı testi.** Gerçek bir sunucuya 4 ayrı tarayıcı oturumu aynı anda bağlandı. Her biri
600 belgelik, kendine özgü bir "gizli sözcük" içeren farklı bir dosya yükleyip tüm modülleri çalıştırdı.

| Ölçüt | Sonuç |
|---|---|
| Her oturum yalnızca kendi gizli sözcüğünü gördü | **4/4** (başkasınınki hiçbir oturumda görünmedi) |
| Hata | 0 |
| Sunucu günlüğünde gizli sözcük | 0 |
| Sunucu belleği | 75 MB → tepe 566 MB |
| Yükleme + ilk analiz | 28–32 sn (tek kullanıcıda aynı dosya: 4,8 sn) |
| Konu modeli | 9–14 sn |

Sonuç: Ücretsiz sunucunun 2 çekirdeği paylaşıldığı için eşzamanlı kullanıcılar birbirini belirgin
biçimde yavaşlatır, ancak bellek sınırlar içinde kalır ve oturumlar birbirinden yalıtılmıştır.

---

## 7. Kalan sınırlılıklar

- **Gerçek veri sınırlı:** Gerçek akademik veri ağırlıklı olarak mühendislik özetleridir. Sosyal
  bilimler, eğitim ya da uzun metinlerde (tez, görüşme) sonuçlar farklı olabilir; özellikle LDA uzun
  metinlerde daha iyi çalışabilir.
- **Dış ölçütler dolaylıdır:** Anahtar sözcük örtüşmesi ve terim çiftleri gibi dış ölçütler, yalnızca
  insan değerlendirmesinin yerini kısmen tutar. Konuların ve ağların anlamlılığına araştırmacının belgeleri
  okuyarak karar vermesi gerekir.
- **Kararlılık ölçütü sıraya duyarlıdır:** Düz ağırlıklı konularda kararlılığı olduğundan düşük gösterebilir.
- **Tutarlılık aynı derlem üzerinde ölçülür:** Dış bir başvuru derlemi kullanılmaz.
- **Permütasyon eşiği:** Grup başına aile bazında kontrol sağlar; gruplar arası çoklu karşılaştırma için
  ek düzeltme yapılmaz. Belge sayısı çok azsa (ör. grup başına 2–3 belge) permütasyon dağılımı kaba kalır.
- **Benzerlik konu odaklıdır:** Benzerlik ve kümeleme konuyu yakalar; üslup, tür ve anlamsal eşdeğerliği yakalamaz.
- **Eşzamanlı kullanımda yavaşlama:** Yoğun eşzamanlı kullanımda analizler belirgin biçimde yavaşlar.
  Ücretsiz plan kaynakları Streamlit tarafından değiştirilebilir.

## Kaynaklar

- Aletras, N., & Stevenson, M. (2013). Evaluating topic coherence using distributional semantics. *IWCS*.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *JMLR*, 3, 993–1022.
- Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. *GSCL*.
- Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., & Blei, D. (2009). Reading tea leaves: How humans interpret topic models. *NeurIPS*.
- Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics*, 16(1).
- Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2020). Topic modeling in embedding spaces. *TACL*, 8.
- Greene, D., O'Callaghan, D., & Cunningham, P. (2014). How many topics? Stability analysis for topic models. *ECML PKDD*.
- Kılıç Ünlü, H., & Çetin, A. (2023). *Neural Computing and Applications*, 35, 3413–3422 (DergiParkEn veri seti).
- Lau, J. H., Newman, D., & Baldwin, T. (2014). Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality. *EACL*.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401.
- Levy, O., Goldberg, Y., & Dagan, I. (2015). Improving distributional similarity with lessons learned from word embeddings. *TACL*, 3.
- Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. *EMNLP*.
- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature selection and evaluation for identifying the content of political conflict. *Political Analysis*, 16(4), 372–403.
