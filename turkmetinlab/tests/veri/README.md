# Test verileri

## ud_boun_test_ornek.conllu

UD Turkish-BOUN ağaç bankası **test** kümesinin ilk 400 cümlesi (değiştirilmeden alınmıştır).

- Kaynak: https://github.com/UniversalDependencies/UD_Turkish-BOUN
- Atıf (güncel sürüm): Marşan, B., Akkurt, S. F., Şen, M., Gürbüz, M., Güngör, O., Özateş, Ş. B.,
  Üsküdarlı, S., Özgür, A., Güngör, T., & Öztürk, B. (2022). Enhancements to the BOUN Treebank Reflecting
  the Agglutinative Nature of Turkish. arXiv:2207.11782.
- Atıf (ilk sürüm): Türk, U., Atmaca, F., Özateş, Ş. B., Berk, G., Bedir, S. T., Köksal, A., Öztürk Başaran, B.,
  Güngör, T., & Özgür, A. (2022). Resources for Turkish dependency parsing: introducing the BOUN Treebank
  and the BoAT annotation tool. *Language Resources and Evaluation*, 56(1), 259–307.
  https://doi.org/10.1007/s10579-021-09558-0
- Lisans: Creative Commons Atıf-AynıLisanslaPaylaş 4.0 Uluslararası (CC BY-SA 4.0),
  https://creativecommons.org/licenses/by-sa/4.0/ — bu dosya aynı lisansla dağıtılır.

Dilbilimcilerin elle doğruladığı sözcük ayrımı, lemma ve sözdizim (bağımlılık) bilgisi içerir. Testlerde
şunlar için altın standart olarak kullanılır:

1. Lemmatizasyon doğruluğu (`tests/test_lemma.py`)
2. N-gram bölütleme kurallarının dilbilimsel gerekçesi (`tests/test_bolutleme.py`)
3. Türkçe sözcük ayrımı ve TF-IDF karşılaştırmaları (`tests/test_tfidf_referans.py`)
