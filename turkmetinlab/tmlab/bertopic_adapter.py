"""Optional BERTopic adapter. Never imported by Streamlit unless feature selected.

Use the project's precomputed ONNX E5 vectors; no transformer downloads here.
KMeans avoids optional UMAP/HDBSCAN runtime, but is NOT BERTopic default.
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

MAX_DOCUMENTS = 500

def validate(documents, embeddings, n_topics=8):
    documents = list(documents)
    vectors = np.asarray(embeddings, dtype=np.float32)
    if not (10 <= len(documents) <= MAX_DOCUMENTS):
        raise ValueError("BERTopic prototipi 10–500 belge gerektirir")
    if vectors.ndim != 2 or vectors.shape[0] != len(documents) or not np.isfinite(vectors).all():
        raise ValueError("E5 embedding matrisi geçersiz")
    if not (2 <= n_topics <= min(50, len(documents)//3)):
        raise ValueError("Konu sayısı veri boyutuyla uyumlu değil")
    if any(not isinstance(t, str) or not t.strip() for t in documents):
        raise ValueError("Boş belge var")
    return documents, vectors

def run(documents, embeddings, n_topics=8, min_df=2):
    """Call real BERTopic with precomputed E5 and transparent KMeans baseline.

    Returns model + per-document/topic tables. Not mock topic results.
    Dependencies bertopic are intentionally absent from production requirements.
    """
    docs, vec = validate(documents, embeddings, n_topics)
    try:
        from bertopic import BERTopic
        from bertopic.dimensionality import BaseDimensionalityReduction
    except ImportError as exc:
        raise RuntimeError("BERTopic isteğe bağlı ortamda kurulu değil") from exc
    from pandas import DataFrame
    model = BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),  # E5 dimensions retained; no UMAP.
        hdbscan_model=KMeans(n_clusters=n_topics, random_state=42, n_init=10),
        vectorizer_model=CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b",
                                         min_df=min_df),
        calculate_probabilities=False,
        verbose=False,
    )
    labels, _ = model.fit_transform(docs, vec)
    rows = DataFrame({"Belge sırası": range(1,len(docs)+1), "Konu":labels})
    return model, rows, model.get_topic_info()
