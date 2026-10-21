"""BERTopic ↔ NMF/LDA karşılaştırması. Girdi: CSV (metin) ve aynı sırada E5 embedding .npy.
Yalnızca araştırma/veri sahibi tarafından sağlanan gerçek E5 vektörleri ile bilimsel kullanım.
Çalıştırma: python -m tmlab.bertopic_kalite --csv veri.csv --metin-sutunu ozet --vektorler e5.npy --cikti sonuc.json
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def _npmi(topics, texts):
    """Belge-düzeyi NPMI; aynı Unicode sözcükleştirme tüm modeller için kullanılır."""
    pattern = re.compile(r"(?u)\\b\\w\\w+\\b")
    docs = [set(pattern.findall(t.casefold())) for t in texts]
    n = len(docs)
    if not n:
        return None
    vocab = set(w.casefold() for topic in topics for w in topic)
    frequencies = Counter(w for doc in docs for w in doc if w in vocab)
    cooccurrence = Counter()
    for doc in docs:
        found = sorted(doc & vocab)
        for pos, a in enumerate(found):
            for b in found[pos + 1:]:
                cooccurrence[(a, b)] += 1
    topic_scores = []
    for topic in topics:
        scores = []
        words = [w.casefold() for w in topic]
        for pos, a in enumerate(words):
            for b in words[pos + 1:]:
                ab = cooccurrence[tuple(sorted((a, b)))]
                if not ab or not frequencies[a] or not frequencies[b]:
                    scores.append(-1.0)
                    continue
                p_ab = ab / n
                if p_ab == 1.0:
                    scores.append(1.0)
                else:
                    score = np.log(p_ab / ((frequencies[a]/n) * (frequencies[b]/n))) / -np.log(p_ab)
                    scores.append(float(np.clip(score, -1.0, 1.0)))
        if scores:
            topic_scores.append(float(np.mean(scores)))
    return float(np.mean(topic_scores)) if topic_scores else None

def _diversity(topics):
    words = [w for t in topics for w in t]
    return len(set(words))/len(words) if words else None

def compare(texts, embeddings, k=5, seed=42, top_n=10):
    """Return comparable diagnostics, not a claim of expert-validated themes."""
    from .bertopic_adapter import run
    texts = list(texts)
    vec = np.asarray(embeddings, dtype=np.float32)
    if not (10 <= len(texts) <= 500) or not (2 <= k <= min(30,len(texts)//3)):
        raise ValueError("10+ belge ve belge sayısına uygun konu sayısı gerekli")
    if vec.ndim != 2 or vec.shape[0] != len(texts) or not np.isfinite(vec).all():
        raise ValueError("Embedding matrisi geçersiz")
    if not all(isinstance(t,str) and t.strip() for t in texts):
        raise ValueError("Boş veya metin dışı belge")
    if seed != 42:
        raise ValueError("BERTopic adaptöründe tohum 42; tüm yöntemler için seed=42 kullanın")
    model, docs, info = run(texts,vec,n_topics=k,min_df=1)
    labels = docs["Konu"].tolist()
    names = [int(i) for i in info["Topic"] if int(i) >= 0]
    topics = [[str(w) for w,_ in (model.get_topic(i) or [])[:top_n]] for i in names]
    results = {"BERTopic-E5-KMeans":{"topic_words":topics,
               "topic_counts":{str(i):labels.count(i) for i in sorted(set(labels))},
               "npmi":_npmi(topics,texts),"diversity":_diversity(topics),
               "note":"KMeans: zorunlu tek küme; HDBSCAN outlier (-1) yok."}}
    for method in ("NMF","LDA"):
        vectorizer = (TfidfVectorizer if method=="NMF" else CountVectorizer)(
            lowercase=True, token_pattern=r"(?u)\b\w\w+\b",max_features=5000)
        X = vectorizer.fit_transform(texts)
        if X.shape[1] < k:
            raise ValueError("Yeterli terim yok")
        est = (NMF(n_components=k,init="nndsvda",random_state=seed,max_iter=500)
               if method=="NMF" else
               LatentDirichletAllocation(n_components=k,random_state=seed,max_iter=30))
        scores = est.fit_transform(X)
        terms = vectorizer.get_feature_names_out()
        words = [[str(terms[j]) for j in np.argsort(-row,kind="stable")[:top_n]]
                 for row in est.components_]
        labels2 = scores.argmax(axis=1)
        results[method]={"topic_words":words,
                         "topic_counts":{str(i):int((labels2==i).sum()) for i in range(k)},
                         "npmi":_npmi(words,texts),"diversity":_diversity(words)}
    return {"n":len(texts),"k":k,"seed":seed,
            "evaluation":"Keşifsel belge-düzeyi NPMI; karşılaştırma eşit k/top_n ile yapılır. Uzman geçerliği/kararlılık ayrıca gerekli.",
            "models":results}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--metin-sutunu",default="metin")
    p.add_argument("--vektorler",required=True)
    p.add_argument("--cikti",required=True)
    p.add_argument("--k",type=int,default=5)
    a=p.parse_args()
    data=pd.read_csv(a.csv)
    texts=data[a.metin_sutunu].tolist()
    vectors=np.load(a.vektorler,allow_pickle=False)
    result=compare(texts,vectors,k=a.k)
    path=Path(a.cikti)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Kaydedildi: {path}; n={result['n']}")

if __name__=="__main__":
    main()
