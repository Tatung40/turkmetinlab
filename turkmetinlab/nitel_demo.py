"""TürkMetinLab 3.0 — bağımsız nitel analiz pilot ekranı.

Çalıştırma: streamlit run nitel_demo.py
LLM/API yok; veriler yalnızca Streamlit oturumunda, dışa aktarım kullanıcı isteğiyle.
"""
import json
from dataclasses import asdict
import pandas as pd
import streamlit as st
from tmlab.nitel_cekirdek import EvidenceCode, prepare_prompt, analyze_model_response, frequencies
from tmlab.nitel_local_model import local_completion
from tmlab.nitel_ucretsiz import suggest_merges, merge_approved_codes
from tmlab.nitel_semantik import from_e5
from tmlab.nitel_toplu import batch_analyze, regroup_once
from tmlab.nitel_kalite import audit_hierarchy
from tmlab.nitel_rapor import frequency_tables
from tmlab.nitel_hiyerarsi import build_hierarchy, prepare_hierarchy_prompt, validate_hierarchy_response, apply_hierarchy

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

detail = st.selectbox("Kodlama ayrıntı düzeyi", ["Dengeli", "Ayrıntılı", "Özetleyici"], index=0,
    help="Yöntemsel doğruluk garantisi değildir; modelden istenen kodlama ayrıntısını değiştirir.")
codebook = st.text_area("İsteğe bağlı kod kitabı (her satırda bir kod)", "")
try:
    prompt = prepare_prompt(source, codebook.splitlines(), detail=detail) if source.strip() else ""
except ValueError as ex:
    st.warning(str(ex))
    prompt = ""
if prompt:
    with st.expander("Model için kısa komutu görüntüle / kopyala"):
        st.code(prompt, language="text")
        st.caption("Üçüncü taraf modele kişisel veri göndermeden önce anonimleştirin ve gerekli izinleri alın.")

with st.expander("İsteğe bağlı yerel model bağlantısı — ücretsiz / özel"):
    st.caption("Yalnızca uygulama ile model AYNI bilgisayarda çalışıyorsa kullanılabilir. "
               "Streamlit Cloud bu bağlantıyla sizin bilgisayarınızdaki modele erişemez. "
               "Üçüncü taraf API'ye veri gönderilmez.")
    local_url = st.text_input("Yerel model adresi", "http://127.0.0.1:11434/v1/chat/completions")
    local_name = st.text_input("Yerel model adı", "turkish-llama")
    if st.button("Yerel modelden kod öner", disabled=not bool(prompt)):
        try:
            result_text = local_completion(prompt,local_url,local_name)
            result = analyze_model_response(pid,source,result_text,codebook.splitlines() if codebook.strip() else None)
            st.session_state.nitel_pending = result["codes"]
            if result["rejected"]:
                st.warning("Kanıt denetiminde reddedilen: " + "; ".join(result["rejected"]))
            st.success(f"{len(result['codes'])} kanıtı doğrulanmış öneri alındı.")
        except Exception as ex:
            st.error("Yerel model çağrısı başarısız. Adres, model ve JSON biçimini kontrol edin.")
if records:
    with st.expander("Toplu otomatik nitel kod önerileri — ücretsiz yerel model"):
        st.caption("Bir işlemde en çok 200 benzersiz katılımcı. Yerel model her kaydı bir kez işler; öneriler insan onayı gerektirir. "
                   "Streamlit Cloud bilgisayarınızdaki yerel modele bağlanamaz.")
        n_batch = st.number_input("Bu işlemde işlenecek ilk kayıt sayısı", min_value=1,
                                  max_value=min(200,len(records)), value=min(20,len(records)))
        if st.button("Seçilen kayıtları otomatik kodla"):
            bar = st.progress(0.0)
            def _advance(done,total):
                bar.progress(done/total)
            try:
                subset = records[:int(n_batch)]
                outcome = batch_analyze(
                    subset, lambda p: local_completion(p, local_url, local_name),
                    codebook.splitlines(), progress=_advance, detail=detail)
                st.session_state.nitel_batch_result = outcome
                st.success(f"{outcome['processed']} kayıt işlendi; {len(outcome['proposals'])} öneri ve "
                           f"{len(outcome['errors'])} uyarı üretildi.")
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
        report = st.session_state.get("nitel_batch_result")
        if report:
            st.dataframe(pd.DataFrame(report["proposals"]), use_container_width=True)
            if report["errors"]:
                st.dataframe(pd.DataFrame(report["errors"]), use_container_width=True)
            if st.button("Kodlardan otomatik kategori ve tema öner (tek model çağrısı)",
                         disabled=not bool(report["proposals"])):
                try:
                    st.session_state.nitel_batch_result = regroup_once(
                        report,lambda p: local_completion(p,local_url,local_name))
                    st.success("Toplu kategori ve tema önerileri oluşturuldu.")
                    st.rerun()
                except Exception:
                    st.error("Model hiyerarşiyi eksik ya da geçersiz döndürdü; eski sonuçlar korundu.")
            if st.button("Tüm toplu önerileri araştırmacı denetimine ekle"):
                st.session_state.nitel_pending = list(report["proposals"])
                st.success("Öneriler denetim kuyruğuna eklendi. Onaylanmadan raporlanmaz.")
            st.download_button("Toplu önerileri CSV indir", pd.DataFrame(report["proposals"]).to_csv(
                index=False).encode("utf-8-sig"), file_name="nitel_toplu_taslak.csv", mime="text/csv")

raw = st.text_area("Modelden gelen JSON (veya aşağıdaki örnek)", height=150,
    placeholder='{"codes":[{"quote":"özgün metindeki birebir parça","code":"kod","category":"kategori","theme":"tema"}]}')
if st.button("Önerileri denetle", disabled=not bool(source.strip() and raw.strip())):
    try:
        result = analyze_model_response(pid,source,raw,codebook.splitlines() if codebook.strip() else None)
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
    universe = [pid for pid, _ in records] if records else sorted({row.participant_id for row in rows})
    st.dataframe(pd.DataFrame(frequencies(rows, universe)), use_container_width=True)
    stats = frequency_tables(accepted, universe)
    st.subheader("Katılımcı temelli kod, kategori ve tema frekansları")
    st.caption("Payda benzersiz katılımcı sayısıdır; çoklu kodlamada yüzdeler toplamı %100'ü aşabilir. "
               "Frekansları yapay zekâ değil, uygulama hesaplar.")
    st.json(stats["summary"], expanded=False)
    for label, key in (("Kod", "codes"), ("Kategori", "categories"), ("Tema", "themes")):
        st.markdown("**" + label + " frekansları**")
        st.dataframe(pd.DataFrame(stats[key]), use_container_width=True)
    from io import BytesIO
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(accepted).to_excel(writer, sheet_name="Onayli_Kodlar", index=False)
            for sheet, key in (("Kod_Frekans", "codes"),("Kategori_Frekans", "categories"),
                               ("Tema_Frekans", "themes")):
                pd.DataFrame(stats[key]).to_excel(writer, sheet_name=sheet, index=False)
            pd.DataFrame([stats["summary"]]).to_excel(writer, sheet_name="Ozet", index=False)
        st.download_button("Onaylı analiz raporunu Excel indir", output.getvalue(),
                           file_name="turkmetinlab_nitel_rapor.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        st.warning("Excel dışa aktarımı için openpyxl kurulu olmalı; CSV indirme kullanılabilir.")
    st.download_button("Onaylı kayıtları CSV indir",
        data=pd.DataFrame(accepted).to_csv(index=False).encode("utf-8-sig"),
        file_name="nitel_onayli_kodlar.csv",mime="text/csv")

if accepted:
    st.divider()
    st.subheader("Derlem genelinde otomatik kategori–tema düzenleme")
    st.caption("Başlangıç hiyerarşisi doğrulanmış kod önerilerinden derlenir; araştırmacı isterse modelle tüm kodları TEK ÇAĞRIDA yeniden gruplar.")
    hierarchy = build_hierarchy(accepted, universe)
    if hierarchy["conflicting_codes"]:
        st.warning("Birden fazla üst kategori/tema altında bulunan kodlar: " + ", ".join(hierarchy["conflicting_codes"]))
    st.dataframe(pd.DataFrame(hierarchy["mapping"]), use_container_width=True)
    st.download_button("Kod–kategori–tema hiyerarşisi CSV indir",
        pd.DataFrame(hierarchy["mapping"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="nitel_hiyerarsi.csv", mime="text/csv")
    if st.button("Kodları gruplayacak kısa model komutunu hazırla"):
        try:
            st.session_state.nitel_hierarchy_prompt = prepare_hierarchy_prompt(accepted)
        except ValueError as exc:
            st.warning(str(exc))
    if st.session_state.get("nitel_hierarchy_prompt"):
        with st.expander("Tek çağrılık kategori/tema komutu", expanded=True):
            st.code(st.session_state.nitel_hierarchy_prompt)
            st.caption("Bu istem özgün görüşmeleri değil, yalnızca kod adları ve sayıları içerir. Kod adları yine hassas bilgi barındırabilir.")
        if st.button("Yerel modelle kategori ve temaları otomatik öner"):
            try:
                response_auto = local_completion(st.session_state.nitel_hierarchy_prompt,
                                                 local_url, local_name)
                grouping_auto = validate_hierarchy_response(
                    response_auto, {r["code"] for r in accepted})
                st.session_state.nitel_accepted = apply_hierarchy(accepted, grouping_auto)
                st.success("Kategori ve temalar model tarafından önerildi. Araştırmacı onayı gereklidir.")
                st.rerun()
            except Exception:
                st.error("Yerel model kategorileri/temaları uygun JSON biçiminde döndürmedi veya erişilemedi.")
        st.caption("Model cevabını buraya yapıştırın; tüm kodlar korunmadan yeni hiyerarşi kabul edilmez.")
        response = st.text_area("Modelden gelen hiyerarşi JSON", key="nitel_hierarchy_response")
        if st.button("Hiyerarşi önerisini uygula", disabled=not bool(response.strip())):
            try:
                grouping = validate_hierarchy_response(response, {r["code"] for r in accepted})
                st.session_state.nitel_accepted = apply_hierarchy(accepted, grouping)
                st.success("Tüm kodlar korundu; kategori ve tema önerileri uygulandı. Sonuçları araştırmacı olarak denetleyin.")
                st.rerun()
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                st.error(f"Hiyerarşi uygulanmadı: {exc}")

if accepted:
    with st.expander("Ücretsiz kod birleştirme önerileri — API gerekmez"):
        st.caption("Sadece yazım/sözcük benzerliğine göre aday önerilir. Anlamsal eşdeğerlik araştırmacı tarafından belirlenir.")
        threshold = st.slider("Kod adı benzerlik eşiği", 0.50, 1.00, 0.80, 0.05)
        try:
            candidates = suggest_merges([r["code"] for r in accepted], threshold=threshold)
            if candidates:
                st.dataframe(pd.DataFrame(candidates), use_container_width=True)
                choices = [f'{x["code_a"]} → {x["code_b"]}' for x in candidates]
                selection = st.selectbox("İncelenecek kod önerisi", range(len(choices)), format_func=lambda i: choices[i])
                if st.button("Seçilen kod birleştirmesini onayla"):
                    item = candidates[selection]
                    st.session_state.nitel_accepted = merge_approved_codes(
                        accepted, {item["code_a"]: item["code_b"]})
                    st.success("Kod adı birleştirildi; özgün kanıt alıntıları korundu.")
                    st.rerun()
            else:
                st.info("Belirlenen eşikte benzer kod adı bulunmadı.")
        except ValueError as exc:
            st.warning(str(exc))

if accepted:
    with st.expander("E5 ile ücretsiz anlamsal kod eşleştirme — isteğe bağlı"):
        st.caption("Var olan E5 ONNX modelini kullanır; yeni ücretli API gerektirmez. "
                   "İlk model yüklemesi yapılandırılmışsa indirilebilir; bellek kullanır. "
                   "Yalnızca kod adları analiz edilir; kanıt metinleri modele gönderilmez.")
        semantic_threshold = st.slider("Anlamsal benzerlik eşiği", 0.60, 0.99, 0.84, 0.01)
        if st.button("E5 kod benzerliğini hesapla"):
            try:
                from tmlab.anlamsal import ModelKaynagi, model_hazirla, E5Kodlayici
                source_model = ModelKaynagi.ortamdan()
                if source_model.eksik():
                    st.error(source_model.eksik())
                else:
                    with st.spinner("Mevcut E5 ONNX modeli ile kodlar eşleştiriliyor"):
                        folder = model_hazirla(source_model)
                        encoder = E5Kodlayici(folder)
                        suggestions = from_e5([r["code"] for r in accepted], encoder,
                                              threshold=semantic_threshold)
                    st.session_state.nitel_semantic_suggestions = suggestions
            except Exception:
                st.error("E5 kod benzerliği hesaplanamadı; model ayarlarını kontrol edin.")
        proposals = st.session_state.get("nitel_semantic_suggestions", [])
        if proposals:
            st.dataframe(pd.DataFrame(proposals), use_container_width=True)
            choice = st.selectbox("Birleştirilecek anlamsal kod çifti", range(len(proposals)),
                format_func=lambda i: proposals[i]["code_a"] + " → " + proposals[i]["code_b"])
            if st.button("Anlamsal kod birleştirmesini onayla"):
                pair = proposals[choice]
                st.session_state.nitel_accepted = merge_approved_codes(
                    accepted, {pair["code_a"]: pair["code_b"]})
                st.session_state.nitel_semantic_suggestions = []
                st.rerun()
        elif "nitel_semantic_suggestions" in st.session_state:
            st.info("Bu eşikte benzer kod adayı bulunmadı.")

if accepted:
    with st.expander("Nitel analiz kalite denetimi — ücretsiz"):
        audit = audit_hierarchy(accepted, universe)
        st.caption(f"Kod ataması: {audit['assignments']} | Farklı kod: {audit['unique_codes']} | "
                   f"Kategori: {audit['categories']} | Tema: {audit['themes']}")
        st.dataframe(pd.DataFrame(audit["warnings"]), use_container_width=True)
        st.caption("Bunlar yöntemsel inceleme uyarılarıdır; kanıt–kod anlam uyumunu tek başına ölçmez.")
