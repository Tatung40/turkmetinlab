"""Side-by-side descriptive comparison of independent Turkish sentiment methods.

No calibration, gold labels, or cross-method numeric comparability is assumed.
"""
from __future__ import annotations
import pandas as pd

def birlestir(names, sw=None, tes=None, electra=None):
    """Join existing outputs by stable row index, not duplicate-prone document name."""
    n=len(names)
    all_sources=((sw,"SWNetTR++"),(tes,"TES-Lex"),(electra,"ELECTRA"))
    for rows,src in all_sources:
        if rows is not None and len(rows)!=n:
            raise ValueError(src+" belge sayısı analiz edilen derlemle uyuşmuyor")
    result=[]
    for i,name in enumerate(names):
        row={"Belge sırası":i+1,"Belge":str(name)}
        if sw is not None:
            x=sw[i]
            row.update({"SWNetTR++ ton ort.":x.get("Ton ortalaması"),
                        "SWNetTR++ olumlu eşleşme":x.get("Olumlu eşleşme"),
                        "SWNetTR++ olumsuz eşleşme":x.get("Olumsuz eşleşme"),
                        "SWNetTR++ kapsama (%)":x.get("Kapsama (%)")})
        if tes is not None:
            x=tes[i]
            row.update({"TES-Lex kapsama (%)":x.get("Kapsama (%)"),
                        **{"TES-Lex "+k:x.get(k) for k in (
                            "Öfke","Neşe","Üzüntü","Korku","İğrenme","Şaşkınlık","Beklenti","Güven")}})
        if electra is not None:
            x=electra[i]
            row.update({"ELECTRA sınıfı":x.get("Etiket"),
                        "ELECTRA model skoru":x.get("En yüksek model skoru"),
                        "ELECTRA kırpıldı":x.get("Kesildi mi")})
        result.append(row)
    return pd.DataFrame(result)

def ozetsayim(frame):
    """Only observed categorical predictions, never ground truth accuracy."""
    if "ELECTRA sınıfı" not in frame:
        return pd.DataFrame(columns=["Sınıf","Frekans","Yüzde"])
    counts=frame["ELECTRA sınıfı"].value_counts(dropna=False)
    denom=len(frame)
    return pd.DataFrame({"Sınıf":counts.index.astype(str),
                         "Frekans":counts.to_numpy(),
                         "Yüzde":[round(100*v/denom,2) if denom else 0 for v in counts]})
