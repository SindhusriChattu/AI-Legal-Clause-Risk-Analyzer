"""
AI Legal Clause Risk Analyzer
------------------------------
Upload a contract (PDF / DOCX / TXT) or paste clause text, and this app will:
  1. Segment the document into individual clauses
  2. Extract legal entities (parties, dates, monetary amounts) via spaCy NER
  3. Retrieve semantically similar reference clauses using Sentence-Transformers + FAISS (RAG)
  4. Score each clause's risk level (Low / Medium / High)
  5. Generate a plain-English explanation (Groq LLM if a key is provided, else rule-based)

Run locally:    streamlit run app.py
Deploy:         push to GitHub -> Streamlit Community Cloud -> point to app.py
"""

import streamlit as st
import pandas as pd

from utils import (
    extract_text_from_file,
    split_into_clauses,
    extract_entities,
    load_vectorstore,
    retrieve_similar_clauses,
    score_clause_risk,
    generate_explanation,
)

st.set_page_config(page_title="AI Legal Clause Risk Analyzer", page_icon="⚖️", layout="wide")

RISK_COLOR = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60", "Unknown": "#7f8c8d"}


@st.cache_resource(show_spinner="Loading vector store...")
def get_store():
    return load_vectorstore("vectorstore.pkl")


def risk_badge(risk_level: str) -> str:
    color = RISK_COLOR.get(risk_level, "#7f8c8d")
    return f"<span style='background-color:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.85em;font-weight:600;'>{risk_level} RISK</span>"


def main():
    st.title("⚖️ AI Legal Clause Risk Analyzer")
    st.caption("NLP + RAG powered contract clause risk detection — NER · Sentence-Transformers · FAISS · LLM explanations")

    with st.sidebar:
        st.header("⚙️ Settings")
        groq_key = st.text_input("Groq API Key (optional, for LLM explanations)", type="password",
                                  help="Leave blank to use the built-in rule-based explanations. Get a free key at console.groq.com")
        top_k = st.slider("Reference clauses to retrieve (RAG top-k)", min_value=1, max_value=5, value=3)
        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Clause segmentation (regex + spaCy)\n"
            "2. Legal NER (parties, dates, money)\n"
            "3. Semantic search over a labeled clause knowledge base (FAISS)\n"
            "4. Similarity-weighted risk scoring\n"
            "5. LLM / template explanation grounded on retrieved clauses"
        )

    tab1, tab2 = st.tabs(["📄 Upload Document", "✍️ Paste Text"])
    raw_text = ""

    with tab1:
        uploaded = st.file_uploader("Upload a contract", type=["pdf", "docx", "txt"])
        if uploaded is not None:
            with st.spinner("Extracting text..."):
                raw_text = extract_text_from_file(uploaded)
            st.success(f"Extracted {len(raw_text)} characters.")

    with tab2:
        pasted = st.text_area("Paste contract text or a single clause here", height=200)
        if pasted.strip():
            raw_text = pasted

    analyze = st.button("🔍 Analyze Clauses", type="primary", use_container_width=True)

    if analyze and raw_text.strip():
        store = get_store()
        clauses = split_into_clauses(raw_text)

        if not clauses:
            st.warning("Could not find any clauses to analyze. Try pasting plain text.")
            return

        st.info(f"Segmented into **{len(clauses)}** clause(s). Analyzing...")

        results = []
        progress = st.progress(0.0)

        for i, clause in enumerate(clauses):
            entities = extract_entities(clause)
            retrieved = retrieve_similar_clauses(clause, store, top_k=top_k)
            risk_info = score_clause_risk(retrieved)
            explanation = generate_explanation(clause, retrieved, risk_info, groq_api_key=groq_key or None)

            results.append({
                "clause": clause,
                "entities": entities,
                "retrieved": retrieved,
                "risk_info": risk_info,
                "explanation": explanation,
            })
            progress.progress((i + 1) / len(clauses))

        progress.empty()

        # ---- Summary dashboard ----
        st.subheader("📊 Risk Summary")
        risk_counts = pd.Series([r["risk_info"]["risk_level"] for r in results]).value_counts()
        cols = st.columns(4)
        for idx, level in enumerate(["High", "Medium", "Low", "Unknown"]):
            count = int(risk_counts.get(level, 0))
            cols[idx].metric(f"{level} Risk", count)

        st.bar_chart(risk_counts)

        st.divider()
        st.subheader("📋 Clause-by-Clause Analysis")

        for i, r in enumerate(results, start=1):
            risk_info = r["risk_info"]
            with st.expander(f"Clause {i}: {risk_info['clause_type']} — {risk_info['risk_level']} risk"):
                st.markdown(risk_badge(risk_info["risk_level"]), unsafe_allow_html=True)
                st.markdown(f"**Clause text:**\n\n> {r['clause']}")

                ents = r["entities"]
                if any(ents.values()):
                    st.markdown("**Extracted entities:**")
                    if ents["parties"]:
                        st.markdown(f"- Parties/Orgs: {', '.join(ents['parties'])}")
                    if ents["dates"]:
                        st.markdown(f"- Dates: {', '.join(ents['dates'])}")
                    if ents["money"]:
                        st.markdown(f"- Monetary amounts: {', '.join(ents['money'])}")

                st.markdown("**Why this risk level:**")
                st.write(r["explanation"])

                if r["retrieved"]:
                    st.markdown("**Top similar reference clauses (RAG retrieval):**")
                    ref_df = pd.DataFrame(r["retrieved"])[["clause_type", "risk_level", "similarity"]]
                    st.dataframe(ref_df, use_container_width=True, hide_index=True)

    elif analyze:
        st.warning("Please upload a document or paste some text first.")


if __name__ == "__main__":
    main()
