"""Ücretsiz yerel model ile toplu nitel kod önerisi. Ağ ücretli API'ye açılmaz."""
from __future__ import annotations
from .nitel_local_model import LocalModelError
from .nitel_cekirdek import prepare_prompt, analyze_model_response
from .nitel_hiyerarsi import build_hierarchy, prepare_hierarchy_prompt, validate_hierarchy_response, apply_hierarchy

MAX_BATCH = 200

def batch_analyze(records, completion, *, codebook=(), progress=None, detail='Dengeli'):
    """Her kayıt için tek model isteği. processed=denenen kayıt sayısıdır.

    succeeded: yanıtı ayrıştırılan ve hiçbir önerisi reddedilmeyen kayıtlar
    (geçerli boş kod listesi dahil). failed: çağrı/ayrıştırma/doğrulama hatalı
    kayıtlar; aynı kaydın birden fazla hatası sayacı artırmaz. Kısmen geçerli
    öneriler korunur. Hiyerarşi çağrısı bu görüş sayaçlarına dahil değildir.

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
    succeeded = failed = 0
    for index, (pid, source) in enumerate(items):
        try:
            prompt = prepare_prompt(source, labels, detail=detail)
            result = analyze_model_response(pid, source, completion(prompt),
                                            labels if labels else None)
            accepted.extend(result["codes"])
            if result["rejected"]:
                failed += 1
            else:
                succeeded += 1
            for reason in result["rejected"]:
                failures.append({"participant_id":pid,"reason":reason})
        except Exception as exc:
            failed += 1
            error = {"participant_id":pid,"reason":type(exc).__name__}
            if isinstance(exc, LocalModelError):
                error["message"] = str(exc)
            failures.append(error)
        if progress is not None:
            progress(index + 1, len(items))
    hierarchy = build_hierarchy(accepted, ids)
    return {"proposals":accepted, "errors":failures, "hierarchy":hierarchy,
            "processed":len(items), "succeeded":succeeded, "failed":failed,
            "participant_ids":ids, "requires_researcher_approval":True}

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
