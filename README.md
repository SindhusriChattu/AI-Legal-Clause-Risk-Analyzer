# ⚖️ AI Legal Clause Risk Analyzer

NLP + RAG powered app that reads a contract, splits it into clauses, and flags each clause's risk level (Low / Medium / High) with a plain-English explanation.

## Pipeline
1. **Document ingestion** — PDF (`pdfplumber`) / DOCX (`python-docx`) / TXT
2. **Clause segmentation** — regex on numbered clauses, falls back to spaCy sentence splitting
3. **Legal NER** — spaCy `en_core_web_sm` extracts parties, dates, monetary amounts
4. **Semantic retrieval (RAG)** — `sentence-transformers` (`all-MiniLM-L6-v2`) embeds each clause; **FAISS** retrieves the top-k most similar clauses from a hand-labeled reference knowledge base (34 clauses across 13 clause types: Indemnification, Termination, Limitation of Liability, Confidentiality, Non-Compete, Payment Terms, Arbitration, Force Majeure, IP, Warranty, Renewal, Governing Law, Assignment)
5. **Risk scoring** — similarity-weighted vote across retrieved neighbours → predicted clause type + risk level
6. **Explanation generation** — if a Groq API key is supplied, `llama-3.1-8b-instant` generates a grounded explanation using the retrieved clauses as context (this is the RAG generation step); otherwise a deterministic template explanation is used, so **the app works fully without any API key**.

## Files
```
app.py                   # Streamlit UI
utils.py                 # NLP pipeline (ingestion, segmentation, NER, retrieval, scoring, explanation)
build_vectorstore.py     # Optional: pre-build vectorstore.pkl locally for faster cold starts
data/clause_dataset.csv  # Reference clause knowledge base (RAG source)
requirements.txt
.python-version           # pins Python 3.11 for Streamlit Cloud
```

Note: `vectorstore.pkl` is NOT required in the repo — if it's missing, `app.py` builds
the FAISS index automatically on first run from `data/clause_dataset.csv` (takes a
couple of seconds, cached via `st.cache_resource` for the rest of the session).
If you want faster cold starts, run `python build_vectorstore.py` locally once and
commit the resulting `vectorstore.pkl`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (e.g. `AI-Legal-Clause-Risk-Analyzer`)
2. Go to share.streamlit.io → New app → select repo → main file: `app.py`
3. Deploy. First load will download `en_core_web_sm` + `all-MiniLM-L6-v2` (~100MB total) — this needs internet access, which Streamlit Cloud has.
4. (Optional) Get a free Groq API key at console.groq.com and paste it in the app's sidebar for LLM-generated explanations instead of the rule-based ones.

## Interview talking points
- **Why FAISS + Sentence-Transformers instead of keyword search?** Legal clauses use varied wording for the same risk (e.g. "unlimited liability" vs "no cap on damages") — semantic embeddings capture meaning, not just words.
- **Why is this RAG, not just classification?** The model doesn't just output a label — it *retrieves* real labeled reference clauses as evidence, and the explanation step is *grounded* on that retrieved evidence rather than the LLM's own unguided judgment. This reduces hallucination and makes the risk score auditable/explainable (you can literally show which reference clause drove the score).
- **Why similarity-weighted voting instead of a trained classifier?** With a small labeled set (34 examples), a few-shot retrieval approach generalizes better than training a classifier from scratch, and it's trivially extensible — add more labeled clauses to the CSV and the system improves without retraining.
- **How would you scale this?** Swap `IndexFlatIP` for `IndexIVFFlat` or `HNSW` for larger clause libraries; add a fine-tuned legal-domain embedding model (e.g. Legal-BERT) instead of general MiniLM; expand NER with a custom spaCy pipeline trained on annotated legal entities (parties, jurisdiction, obligations) instead of the generic `en_core_web_sm`.
- **Limitations to be upfront about:** the current knowledge base is a hand-curated sample (34 clauses) for demonstration — a production system would need a much larger, lawyer-reviewed clause library; also this is a decision-support tool, not a substitute for actual legal review.
