"""E5 ONNX vektörlerinden ücretsiz, onay gerektiren anlamsal kod eşleştirme adayları."""
from __future__ import annotations
import numpy as np

def semantic_merge_candidates(codes, vectors, *, threshold=0.84, max_codes=300):
    """Kosinüs benzerliği; kodların anlamsal eşit olduğunu KANITLAMAZ."""
    names = list(dict.fromkeys(s.strip() for s in codes if isinstance(s, str) and s.strip()))
    if len(names) < 2:
        return []
    if len(names) > max_codes or not (0.5 <= threshold <= 1.0):
        raise ValueError("Kod sayısı veya benzerlik eşiği geçersiz.")
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2 or v.shape[0] != len(names) or v.shape[1] < 2 or not np.isfinite(v).all():
        raise ValueError("Kod sırasıyla eşleşen, sonlu embedding matrisi gerekli.")
    length = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(length <= 1e-12):
        raise ValueError("Sıfır uzunluklu kod embedding'i.")
    v = v / length
    cosine = v @ v.T
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            similarity = float(cosine[i, j])
            if similarity >= threshold:
                out.append({"code_a": names[i], "code_b": names[j],
                            "cosine": round(similarity, 4),
                            "researcher_approval_required": True})
    return sorted(out, key=lambda row: (-row["cosine"], row["code_a"], row["code_b"]))

def from_e5(codes, encoder, *, threshold=0.84):
    """Var olan E5Kodlayici'yı yeniden kullanır; model indirmez, API çağırmaz."""
    from .anlamsal import BELGE_ONEKI
    names = list(dict.fromkeys(s.strip() for s in codes if isinstance(s, str) and s.strip()))
    if len(names) < 2:
        return []
    if len(names) > 300:
        raise ValueError("Tek seferde en çok 300 kod.")
    vectors = encoder.kodla(names, BELGE_ONEKI)
    return semantic_merge_candidates(names, vectors, threshold=threshold)
