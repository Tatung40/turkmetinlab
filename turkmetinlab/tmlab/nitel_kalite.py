"""Nitel analiz için ücretsiz, açıklanabilir hiyerarşi kalite uyarıları.

Bu kontroller tanısaldır; otomatik bilimsel doğruluk kararı vermez.
"""
from collections import Counter, defaultdict

def audit_hierarchy(records, participant_ids=None):
    rows = list(records)
    ids = set(participant_ids) if participant_ids is not None else {r["participant_id"] for r in rows}
    codes = {r["code"] for r in rows}
    categories = {r["category"] for r in rows}
    themes = {r["theme"] for r in rows}
    by_category = defaultdict(set)
    by_code = defaultdict(set)
    warnings = []
    for r in rows:
        by_category[(r["category"],r["theme"])].add(r["code"])
        by_code[r["code"]].add((r["category"],r["theme"]))
        quote = r.get("quote", "")
        if not isinstance(quote,str) or not quote.strip():
            warnings.append({"type":"missing_evidence","item":str(r.get("participant_id",""))})
    singleton_categories = sum(len(v)==1 for v in by_category.values())
    if len(by_category) >= 5 and singleton_categories/len(by_category) >= .8:
        warnings.append({"type":"fragmentation","item":f"{singleton_categories}/{len(by_category)} tek kodlu kategori"})
    if len(themes)==1 and len(codes)>=6:
        warnings.append({"type":"broad_theme","item":"Bütün kodlar tek temada"})
    for code, assignments in sorted(by_code.items()):
        if len(assignments)>1:
            warnings.append({"type":"inconsistent_hierarchy","item":code})
    return {"participants":len(ids),"assignments":len(rows),
            "unique_codes":len(codes),"categories":len(by_category),"themes":len(themes),
            "singleton_categories":singleton_categories,"warnings":warnings,
            "needs_researcher_review":True}
