"""TürkMetinLab 3.0 — bağımsız nitel analiz pilot ekranı.

Çalıştırma: streamlit run nitel_demo.py
LLM/API yok; veriler yalnızca Streamlit oturumunda, dışa aktarım kullanıcı isteğiyle.
"""
import json
from dataclasses import asdict
import pandas as pd
import streamlit as st
from tmlab.nitel_cekirdek import EvidenceCode, prepare_prompt, analyze_model_response, frequencies

st.set_page_config(page_title="TürkMetinLab | Nitel Analiz Pilot", layout="wide")
st.title("TürkMetinLab 3.0 — Nitel Veri Analizi (Pilot)")
st.info("Deneysel ekran. Otomatik kodlama henüz modele bağlı değil; "
        "taslak kodlar araştırmacı onayı olmadan bilimsel sonuç sayılmaz.")
if "nitel_accepted" not in st.session_state:
    st.session_state.nitel_accepted = []
if "nitel_pending" not in st.session_state:
    st.session_state.nitel_pending = []
if st.button("Oturumdaki tüm nitel analiz kayıtlarını temizle"):
    st.session_state.nitel_accepted = []
    st.session_state.nitel_pending = []
    st.success("Nitel analiz oturum kayıtları temizlendi.")

uploaded = st.file_uploader("CSV yükleyin (katılımcı ID ve metin sütunları)", type=["csv"])
records = []
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
        if len(df) > 5000 or len(df.columns) > 100:
            st.error("Pilot sınırı: 5000 satır, 100 sütun.")
        else:
            id_col = st.selectbox("Katılımcı ID sütunu", list(df.columns), index=0)
            text_col = st.selectbox("Metin sütunu", list(df.columns), index=min(1,len(df.columns)-1))
            if id_col == text_col:
                st.warning("ID ile metin sütunları farklı olmalı.")
            else:
                records = [(str(r[id_col]), str(r[text_col]))
                           for _, r in df.iterrows()
                           if str(r[id_col]).strip() and str(r[text_col]).strip()]
                st.caption(f"Geçerli kayıt: {len(records)}. Metinleri üçüncü taraf hizmetlere göndermiyoruz.")
    except Exception:
        st.error("CSV okunamadı; UTF-8 kodlama ve sütunları denetleyin.")

if records:
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)):
        st.warning("Aynı katılımcı birden fazla satırda bulunuyor; frekanslar benzersiz kimliğe göre hesaplanacak.")
    chosen = st.selectbox("Analiz edilecek kayıt", range(len(records)),
                          format_func=lambda i: f"{records[i][0]} — {records[i][1][:75]}")
    pid, source = records[chosen]
else:
    pid = st.text_input("Katılımcı ID", "T01")
    source = st.text_area("Katılımcının özgün ifadesi", height=135)

codebook = st.text_area("İsteğe bağlı kod kitabı (her satırda bir kod)", "")
try:
    prompt = prepare_prompt(source, codebook.splitlines()) if source.strip() else ""
except ValueError as ex:
    st.warning(str(ex))
    prompt = ""
if prompt:
    with st.expander("Model için kısa komutu görüntüle / kopyala"):
        st.code(prompt, language="text")
        st.caption("Üçüncü taraf modele kişisel veri göndermeden önce anonimleştirin ve gerekli izinleri alın.")

raw = st.text_area("Modelden gelen JSON (veya aşağıdaki örnek)", height=150,
    placeholder='{"codes":[{"quote":"özgün metindeki birebir parça","code":"kod","category":"kategori","theme":"tema"}]}')
if st.button("Önerileri denetle", disabled=not bool(source.strip() and raw.strip())):
    try:
        result = analyze_model_response(pid,source,raw)
        st.session_state.nitel_pending = result["codes"]
        if result["rejected"]:
            st.warning("Reddedilen öneriler: " + "; ".join(result["rejected"]))
        if not result["codes"]:
            st.warning("Doğrulanmış kanıt içeren öneri yok.")
    except (ValueError, TypeError, json.JSONDecodeError) as ex:
        st.error(f"Geçersiz model yanıtı: {ex}")

pending = list(st.session_state.nitel_pending)
if pending:
    st.subheader("Önerileri araştırmacı olarak inceleyin")
    for i, item in enumerate(pending):
        st.markdown(f"**{item['participant_id']} — Özgün kanıt:** {item['quote']}")
        with st.form(f"review_{i}"):
            code = st.text_input("Kod", item["code"])
            category = st.text_input("Kategori", item["category"])
            theme = st.text_input("Tema", item["theme"])
            outcome = st.radio("Karar", ["Kabul et", "Reddet"], horizontal=True)
            if st.form_submit_button("Kararı kaydet"):
                if outcome == "Kabul et" and all(x.strip() for x in (code,category,theme)):
                    accepted = dict(item,code=code.strip(),category=category.strip(),theme=theme.strip())
                    if accepted not in st.session_state.nitel_accepted:
                        st.session_state.nitel_accepted.append(accepted)
                st.session_state.nitel_pending.pop(i)
                st.rerun()

accepted = st.session_state.nitel_accepted
if accepted:
    st.subheader("Araştırmacı onaylı kodlar")
    st.dataframe(pd.DataFrame(accepted), use_container_width=True)
    rows = [EvidenceCode(**r) for r in accepted]
    st.subheader("Kodların katılımcı frekansı")
    st.dataframe(pd.DataFrame(frequencies(rows)), use_container_width=True)
    st.download_button("Onaylı kayıtları CSV indir",
        data=pd.DataFrame(accepted).to_csv(index=False).encode("utf-8-sig"),
        file_name="nitel_onayli_kodlar.csv",mime="text/csv")
