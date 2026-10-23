import pytest
from tmlab.nitel_cekirdek import prepare_prompt, DETAIL_LEVELS
from tmlab.nitel_toplu import batch_analyze

def test_all_three_detail_levels_produce_distinct_prompts():
    prompts=[prepare_prompt("Dersi hızlandırıyor.",detail=k) for k in DETAIL_LEVELS]
    assert len(set(prompts))==3
    assert all("KAYNAK:\nDersi hızlandırıyor." in p for p in prompts)

def test_invalid_detail_rejected():
    with pytest.raises(ValueError,match="ayrıntı"):
        prepare_prompt("Metin",detail="Geçersiz")

def test_batch_detail_passed_through():
    seen=[]
    def respond(prompt):
        seen.append(prompt)
        return '{"codes":[]}'
    result=batch_analyze([("T01","Tek metin")],respond,detail="Özetleyici")
    assert result["processed"]==1
    assert "baskın anlamı" in seen[0]
