from tmlab.nitel_rapor import frequency_tables

def test_multiple_codes_same_person_count_once():
    rows=[
        {"participant_id":"T1","code":"Zaman","category":"Verim","theme":"Fayda"},
        {"participant_id":"T1","code":"Zaman","category":"Verim","theme":"Fayda"},
        {"participant_id":"T1","code":"Kolaylık","category":"Verim","theme":"Fayda"},
        {"participant_id":"T2","code":"Zaman","category":"Verim","theme":"Fayda"}
    ]
    data=frequency_tables(rows,["T1","T2","T3"])
    assert data["summary"]["n_participants"]==3
    assert data["summary"]["n_assignments"]==4
    assert next(x for x in data["codes"] if x["code"]=="Zaman")["n_participants"]==2
    assert data["themes"][0]["n_participants"]==2
    assert data["categories"][0]["n_evidence"]==4

def test_uncoded_participants_preserved():
    result=frequency_tables([],["T1","T2"])
    assert result["summary"]["n_participants"]==2
    assert result["codes"]==[]

def test_unknown_participant_rejected():
    import pytest
    with pytest.raises(ValueError):
        frequency_tables([{"participant_id":"T9","code":"x","category":"y","theme":"z"}],["T1"])
