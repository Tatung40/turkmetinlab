from tmlab.nitel_kalite import audit_hierarchy

def test_fragmentation_warning():
    rows=[{"participant_id":f"T{i}", "code":f"K{i}", "category":f"C{i}",
           "theme":"T", "quote":"kanıt"} for i in range(8)]
    result=audit_hierarchy(rows)
    assert result["assignments"]==8
    assert any(x["type"]=="fragmentation" for x in result["warnings"])

def test_conflicting_parent_flagged():
    rows=[{"participant_id":"T1","code":"Zaman","category":"A","theme":"Üst","quote":"örnek"},
          {"participant_id":"T2","code":"Zaman","category":"B","theme":"Üst","quote":"örnek"}]
    result=audit_hierarchy(rows)
    assert any(x["type"]=="inconsistent_hierarchy" for x in result["warnings"])

def test_no_false_certification():
    result=audit_hierarchy([],["T1","T2"])
    assert result["needs_researcher_review"]
    assert result["participants"]==2
