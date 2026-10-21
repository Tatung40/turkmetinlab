from tmlab.duygu_sosyal import clean_tweet,filtered_lexicon,contribution_table
def test_preserve_emojis_negation_and_hashtag_word():
    x=clean_tweet("RT @hesap: #İklimKrizi 😡 hiç iyi değil https://t.co/a")
    assert "@hesap" not in x and "https://" not in x
    assert "İklimKrizi" in x and "😡" in x and "hiç iyi değil" in x
def test_opt_in_filtered_lexicon():
    x={("ve",):1,("korku",):2,("yok",):3}
    y=filtered_lexicon(x)
    assert ("ve",) not in y and ("yok",) in y and ("korku",) in y
def test_contribution_table():
    t=contribution_table([{"Sözcük/ifade":"ve","Duygular":"Beklenti"},{"Sözcük/ifade":"umut","Duygular":"Beklenti"}])
    assert set(t["Frekans"])=={1}
