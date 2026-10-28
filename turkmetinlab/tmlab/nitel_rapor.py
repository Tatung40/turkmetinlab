"""Onaylı kodlardan güvenilir çoklu-kodlama frekans tabloları.

Her düzeyde payda benzersiz katılımcı sayısıdır; yüzdeler 100'ü aşabilir.
"""
from collections import defaultdict

# Raporla birlikte dışa aktarılan sabit yöntem ve sınırlılık notları (her rapor için aynı, veri içermez).
METHOD_NOTES = (
    ("Kapsam", "Yalnızca araştırmacının tek tek onayladığı kod atamaları sayılır; onaylanmamış YZ önerileri rapora girmez."),
    ("Kanıt", "Her atamanın alıntısı özgün katılımcı metninde birebir bulunmuştur; alıntının kodu anlamca desteklediğine araştırmacı karar verir."),
    ("Payda", "Yüzdeler benzersiz katılımcı sayısına göredir; çoklu kodlamada toplam %100'ü aşabilir."),
    ("Kodlayıcı güvenirliği", "Tek araştırmacı onayıdır; kodlayıcılar arası uyum (ör. Cohen kappa) hesaplanmamıştır."),
    ("Genelleme", "Frekanslar örnekleme özgüdür; nitel verideki sıklık önem ya da evren oranı kanıtı değildir."),
)

def frequency_tables(records, participant_ids=None):
    rows=list(records)
    universe=set(participant_ids) if participant_ids is not None else {r["participant_id"] for r in rows}
    if any(r["participant_id"] not in universe for r in rows):
        raise ValueError("Katılımcı evreni ile kod kayıtları tutarsız.")
    n=len(universe)
    results={}
    for level, fields in (("codes",("code",)),("categories",("theme","category")),("themes",("theme",))):
        groups=defaultdict(set)
        counts=defaultdict(int)
        for row in rows:
            key=tuple(row[field] for field in fields)
            groups[key].add(row["participant_id"])
            counts[key]+=1
        results[level]=[
            dict(zip(fields,key),n_participants=len(groups[key]),
                 percentage=round(100*len(groups[key])/n,2) if n else 0,
                 n_evidence=counts[key])
            for key in sorted(groups)
        ]
    results["summary"]={"n_participants":n,"n_assignments":len(rows),
        "n_unique_codes":len({r["code"] for r in rows}),
        "n_categories":len({(r["theme"],r["category"]) for r in rows}),
        "n_themes":len({r["theme"] for r in rows}),
        "percentage_denominator":"unique_participants",
        "percentages_may_exceed_100":True}
    return results
