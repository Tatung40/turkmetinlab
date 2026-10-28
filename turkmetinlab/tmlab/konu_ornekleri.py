"""Extractive topic examples; faithful text spans, not generated summaries."""
from collections import defaultdict
import pandas as pd

def topic_examples(texts, assignments, max_per_topic=3, max_chars=400):
    if len(texts) != len(assignments):
        raise ValueError("Belge sayısı ve konu atamaları uyuşmuyor")
    if not 1 <= max_per_topic <= 10 or max_chars < 50:
        raise ValueError("Temsil sınırı geçersiz")
    grouped=defaultdict(list)
    for i,(text,label) in enumerate(zip(texts,assignments)):
        if text and str(text).strip():
            grouped[int(label)].append((i,str(text).strip()))
    rows=[]
    for topic,entries in sorted(grouped.items()):
        seen=set()
        for i,text in entries:
            norm=text.casefold()
            if norm in seen: continue
            seen.add(norm)
            rows.append({"Konu":topic,"Belge sırası":i+1,
                         "Kaynak metin alıntısı":text[:max_chars],
                         "Kırpılmış alıntı":len(text)>max_chars})
            if len(seen)>=max_per_topic:break
    return pd.DataFrame(rows,columns=["Konu","Belge sırası","Kaynak metin alıntısı","Kırpılmış alıntı"])
