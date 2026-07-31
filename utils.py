"""
utils.py
Core NLP pipeline for the AI Legal Clause Risk Analyzer.

Pipeline stages:
1. Document ingestion (PDF / DOCX / TXT)
2. Clause segmentation (regex + spaCy sentence boundaries)
3. Legal NER (parties, dates, monetary amounts, obligations) via spaCy
4. Semantic embedding (Sentence-Transformers)
5. Retrieval (FAISS similarity search against a labeled clause knowledge base) -> RAG
6. Risk scoring (weighted vote of top-k retrieved neighbours)
7. LLM-based plain-English explanation (Groq API, optional; falls back to a
   template-based explanation if no API key is supplied so the app still
   works fully offline / without any paid key)
"""

import io
import os
import re
import pickle

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy-loaded globals (loaded once, cached by Streamlit via st.cache_resource
# in app.py — this module just exposes plain functions).
# ---------------------------------------------------------------------------

_NLP = None
_EMBED_MODEL = None


def get_spacy_model():
    """Load a small spaCy English model for sentence splitting + NER."""
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded yet (first run on Streamlit Cloud) -> fetch it.
            from spacy.cli import download
            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


def get_embedding_model():
    """Load the sentence-transformer embedding model (MiniLM: small & fast)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


# ---------------------------------------------------------------------------
# 1. Document ingestion
# ---------------------------------------------------------------------------

def extract_text_from_file(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF, DOCX, or TXT file."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)

    elif name.endswith(".docx"):
        import docx
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    else:  # plain text / fallback
        return data.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# 2. Clause segmentation
# ---------------------------------------------------------------------------

# Legal documents often number clauses like "1.", "2.1", "(a)", "Section 4:" etc.
_CLAUSE_SPLIT_PATTERN = re.compile(
    r"(?:\n\s*(?:\d+\.\d*|\(\w\)|Section\s+\d+|Article\s+\d+)[\.\):]?\s+)",
    re.IGNORECASE,
)


def split_into_clauses(text: str, min_len: int = 25) -> list:
    """
    Split a raw contract text into individual clauses.
    Strategy: first try numbered-clause boundaries; if that yields too few
    pieces (unstructured document), fall back to spaCy sentence segmentation.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    numbered_split = _CLAUSE_SPLIT_PATTERN.split(text)
    candidates = [c.strip() for c in numbered_split if len(c.strip()) >= min_len]

    if len(candidates) >= 3:
        return candidates

    # Fallback: sentence-level segmentation via spaCy
    nlp = get_spacy_model()
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= min_len]
    return sentences


# ---------------------------------------------------------------------------
# 3. Legal NER
# ---------------------------------------------------------------------------

def extract_entities(clause_text: str) -> dict:
    """Extract key legal entities: parties (ORG/PERSON), dates, money amounts."""
    nlp = get_spacy_model()
    doc = nlp(clause_text)
    entities = {"parties": [], "dates": [], "money": [], "misc": []}
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PERSON"):
            entities["parties"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["money"].append(ent.text)
        else:
            entities["misc"].append((ent.text, ent.label_))

    for key in ("parties", "dates", "money"):
        entities[key] = sorted(set(entities[key]))
    return entities


# ---------------------------------------------------------------------------
# 4 & 5. Embedding + FAISS retrieval (RAG)
# ---------------------------------------------------------------------------

def build_vectorstore_from_csv(csv_path: str = "data/clause_dataset.csv") -> dict:
    """Embed every reference clause in csv_path and build an in-memory FAISS index."""
    import faiss

    df = pd.read_csv(csv_path)
    model = get_embedding_model()
    embeddings = model.encode(
        df["clause_text"].tolist(), normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    return {"index": index, "metadata": df.reset_index(drop=True), "dim": dim}


def load_vectorstore(pkl_path: str = "vectorstore.pkl", csv_path: str = "data/clause_dataset.csv") -> dict:
    """
    Load the FAISS index + metadata. If a pre-built pkl exists (produced by
    build_vectorstore.py), load it directly (fast). Otherwise, build it on
    the fly from the CSV knowledge base (used automatically on first deploy,
    e.g. Streamlit Cloud, so no binary file needs to be committed to git).
    """
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return build_vectorstore_from_csv(csv_path)


def retrieve_similar_clauses(clause_text: str, store: dict, top_k: int = 3):
    """Embed a clause and retrieve the top_k most similar reference clauses."""
    import faiss  # noqa: F401 (ensures faiss is importable at call time)

    model = get_embedding_model()
    query_vec = model.encode([clause_text], normalize_embeddings=True).astype("float32")

    index = store["index"]
    metadata = store["metadata"]

    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = metadata.iloc[idx]
        results.append({
            "clause_text": row["clause_text"],
            "clause_type": row["clause_type"],
            "risk_level": row["risk_level"],
            "risk_reason": row["risk_reason"],
            "similarity": float(score),
        })
    return results


# ---------------------------------------------------------------------------
# 6. Risk scoring
# ---------------------------------------------------------------------------

_RISK_TO_SCORE = {"Low": 1, "Medium": 2, "High": 3}
_SCORE_TO_RISK = {1: "Low", 2: "Medium", 3: "High"}


def score_clause_risk(retrieved: list) -> dict:
    """
    Aggregate the top-k retrieved neighbours into:
      - predicted clause_type (majority vote, weighted by similarity)
      - predicted risk_level (similarity-weighted average, rounded)
    """
    if not retrieved:
        return {"clause_type": "Unclassified", "risk_level": "Unknown", "confidence": 0.0}

    type_weights, risk_weighted_sum, weight_total = {}, 0.0, 0.0
    for r in retrieved:
        w = max(r["similarity"], 0.0001)
        type_weights[r["clause_type"]] = type_weights.get(r["clause_type"], 0.0) + w
        risk_weighted_sum += _RISK_TO_SCORE[r["risk_level"]] * w
        weight_total += w

    predicted_type = max(type_weights, key=type_weights.get)
    avg_risk_score = risk_weighted_sum / weight_total if weight_total else 2
    predicted_risk = _SCORE_TO_RISK[int(round(avg_risk_score))]
    confidence = retrieved[0]["similarity"]  # top-1 similarity as a proxy confidence

    return {
        "clause_type": predicted_type,
        "risk_level": predicted_risk,
        "confidence": round(confidence, 3),
    }


# ---------------------------------------------------------------------------
# 7. Explanation generation (RAG -> LLM), with offline fallback
# ---------------------------------------------------------------------------

def generate_explanation(clause_text: str, retrieved: list, risk_info: dict,
                          groq_api_key: str = None) -> str:
    """
    Build a plain-English risk explanation.
    If a Groq API key is provided, use an LLM (llama-3.1-8b-instant) grounded
    on the retrieved reference clauses (this is the "RAG" step). Otherwise,
    fall back to a deterministic template using the retrieved risk_reason
    fields, so the app remains fully functional without any API key.
    """
    context = "\n".join(
        f"- Similar {r['clause_type']} clause (risk: {r['risk_level']}): {r['risk_reason']}"
        for r in retrieved
    )

    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            prompt = (
                "You are a contract risk analyst. Given the clause below and "
                "similar reference clauses with known risk levels, explain in "
                "2-3 concise sentences WHY this clause carries its risk level "
                "and what a business person should watch out for. Avoid legal "
                "jargon.\n\n"
                f"CLAUSE:\n{clause_text}\n\n"
                f"PREDICTED RISK: {risk_info['risk_level']} ({risk_info['clause_type']})\n\n"
                f"REFERENCE CLAUSES:\n{context}\n"
            )
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # network / auth errors -> fall back gracefully
            return _template_explanation(risk_info, retrieved) + f"\n\n(LLM call failed, showing rule-based explanation: {e})"

    return _template_explanation(risk_info, retrieved)


def _template_explanation(risk_info: dict, retrieved: list) -> str:
    if not retrieved:
        return "No sufficiently similar reference clause was found to assess this clause's risk."
    top = retrieved[0]
    return (
        f"This clause was classified as **{risk_info['clause_type']}** with "
        f"**{risk_info['risk_level']} risk** (similarity {risk_info['confidence']:.2f} to closest reference). "
        f"Reasoning drawn from the closest matching reference clause: {top['risk_reason']}"
    )
