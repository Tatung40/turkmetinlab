"""Gerçek yerel modelle 20 görüş pilotu; yalnızca kullanıcının bilgisayarında çalışır.

Kullanım (repo içindeki turkmetinlab dizininde):
    python scripts/nitel_gercek_model_pilot.py --model qwen3:4b --output pilot_gercek.csv
Önce Ollama kurulmalı ve model kullanıcı tarafından yerelde indirilmelidir.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from tmlab.nitel_local_model import local_completion, LocalModelError
from tmlab.nitel_toplu import batch_analyze, regroup_once, validate_records
from tmlab.nitel_rapor import frequency_tables
from tmlab.nitel_kalite import audit_hierarchy

def run(records, complete, *, verify_first=False):
    records, _ = validate_records(records)
    if verify_first and len(records) > 1:
        result = batch_analyze(records[:1], complete, detail="Dengeli")
        if result["failed"] or not result["proposals"]:
            result["errors"].append({"participant_id":"_gate",
                "reason":"FirstRecordNotVerified",
                "message":"İlk görüşte geçerli kanıtlı kod üretilemedi; kalan kayıtlar işlenmedi."})
        else:
            remaining = batch_analyze(records[1:], complete, detail="Dengeli")
            for key in ("proposals", "errors", "participant_ids"):
                result[key] += remaining[key]
            for key in ("processed", "succeeded", "failed"):
                result[key] += remaining[key]
    else:
        result=batch_analyze(records,complete,detail="Dengeli")
    if result["proposals"] and not any(e["participant_id"] == "_gate" for e in result["errors"]):
        try:
            result=regroup_once(result,lambda prompt:complete(prompt, hierarchy=True))
        except (ValueError,KeyError,TypeError,LocalModelError,json.JSONDecodeError) as exc:
            error = {"participant_id":"_hierarchy","reason":type(exc).__name__}
            if isinstance(exc, LocalModelError):
                error["message"] = str(exc)
            result["errors"].append(error)
    result["frequencies"]=frequency_tables(result["proposals"],result["participant_ids"])
    result["quality"]=audit_hierarchy(result["proposals"],result["participant_ids"])
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="tests/fixtures/nitel_20_gorusme.csv")
    ap.add_argument("--model",required=True)
    ap.description = "processed=denenen, succeeded=doğrulanan, failed=hatalı görüş; boş kod listesi geçerlidir."
    ap.add_argument("--limit",type=int,default=1,help="Önce 1 kayıtla bağlantıyı deneyin; en çok 20.")
    ap.add_argument("--output",default="nitel_pilot_kodlar.csv")
    ap.add_argument("--endpoint",default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--timeout",type=int,default=180,help="CPU için 1–900 saniye; otomatik tekrar yok.")
    args=ap.parse_args()
    if not 1 <= args.timeout <= 900:
        ap.error("--timeout 1 ile 900 arasında olmalı.")
    with open(args.input,encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows or any(not r.get("katilimci_id") or not r.get("metin") for r in rows):
        raise SystemExit("CSV katilimci_id ve metin sütunlarını içermeli.")
    def complete(prompt,*,hierarchy=False):
        return local_completion(prompt,args.endpoint,args.model,
            timeout=args.timeout,max_tokens=3072 if hierarchy else 768)
    if not (1 <= args.limit <= 20):
        raise SystemExit("--limit 1 ile 20 arasında olmalı.")
    try:
        report=run([(r["katilimci_id"],r["metin"]) for r in rows[:args.limit]],complete,verify_first=True)
    except ValueError as exc:
        ap.error(str(exc))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    columns=("participant_id","quote","code","category","theme","start","end")
    with output.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows({k:r.get(k,"") for k in columns} for r in report["proposals"])
    json_path=output.with_suffix(".report.json")
    json_path.write_text(json.dumps({
        "processed":report["processed"],"succeeded":report["succeeded"],"failed":report["failed"],
        "model":args.model,"endpoint":args.endpoint,
        "requested":min(args.limit,len(rows)),
        "status":"incomplete" if report["failed"] or report["errors"] else
                 ("validated_proposals" if report["proposals"] else "no_codes"),
        "quality":report["quality"],
        "summary":report["frequencies"]["summary"],"codes":report["frequencies"]["codes"],
        "categories":report["frequencies"]["categories"],"themes":report["frequencies"]["themes"],
        "warnings":report["quality"]["warnings"],"errors":report["errors"],
        "requires_researcher_approval":True,
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"processed":report["processed"],
                      "succeeded":report["succeeded"],"failed":report["failed"],"proposals":len(report["proposals"]),
                      "errors":len(report["errors"]),"csv":str(output),"report":str(json_path)},
                     ensure_ascii=False))
    return 1 if report["failed"] or report["errors"] else 0
if __name__=="__main__":
    raise SystemExit(main())
