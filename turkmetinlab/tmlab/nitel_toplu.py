"""Ücretsiz yerel model ile toplu nitel kod önerisi. Ağ ücretli API'ye açılmaz."""
from __future__ import annotations
from .nitel_cekirdek import prepare_prompt, analyze_model_response
from .nitel_hiyerarsi import build_hierarchy, prepare_hierarchy_prompt, validate_hierarchy_response, apply_hierarchy

MAX_BATCH = 200

def batch_analyze(records, completion, *, codebook=(), progress=None):
    """Her kayıt için tek model isteği. Hatalı kayıtları ayrı raporlar, sessizce silmez.

    completion(prompt) -> JSON metni. Doğrudan yerel adapter veya test sahte fonksiyonu.
    """
    items = list(records)
    if not items or len(items) > MAX_BATCH:
        raise ValueError("Tek toplu işlem 1–200 kayıt içermelidir.")
    ids = [str(pid).strip() for pid, _ in items]
    if any(not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Toplu işlem için dolu, benzersiz kayıt numaraları gerekir.")
    labels = tuple(codebook)
    accepted, failures = [], []
    for index, (pid, source) in enumerate(items):
        try:
            prompt = prepare_prompt(source, labels)
            result = analyze_model_response(pid, source, completion(prompt),
                                            labels if labels else None)
            accepted.extend(result["codes"])
            for reason in result["rejected"]:
                failures.append({"participant_id":pid,"reason":reason})
        except Exception as exc:
            failures.append({"participant_id":pid,"reason":type(exc).__name__})
        if progress is not None:
            progress(index + 1, len(items))
    hierarchy = build_hierarchy(accepted, ids)
    return {"proposals":accepted, "errors":failures, "hierarchy":hierarchy,
            "processed":len(items), "participant_ids":ids, "requires_researcher_approval":True}

def regroup_once(batch_result, completion):
    """Ham metinleri tekrar göndermeden tüm kodlar için yalnızca BİR model çağrısı."""
    codes = batch_result["proposals"]
    if not codes:
        return batch_result
    prompt = prepare_hierarchy_prompt(codes)
    response = completion(prompt)
    mapping = validate_hierarchy_response(response, {r["code"] for r in codes})
    result = dict(batch_result)
    result["proposals"] = apply_hierarchy(codes, mapping)
    result["hierarchy"] = build_hierarchy(result["proposals"],
                                          batch_result["participant_ids"])
    return result
