"""
build_vectorstore.py

Run this ONCE (locally, before deploying) to build vectorstore.pkl from
data/clause_dataset.csv. This pre-computes sentence-transformer embeddings
for every reference clause and stores them in a FAISS index, so the
Streamlit app itself never needs to re-embed the knowledge base at runtime.

Usage:
    python build_vectorstore.py
"""

import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def main():
    df = pd.read_csv("data/clause_dataset.csv")
    print(f"Loaded {len(df)} reference clauses across {df['clause_type'].nunique()} clause types.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["clause_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    store = {
        "index": index,
        "metadata": df.reset_index(drop=True),
        "dim": dim,
    }

    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(store, f)

    print(f"Saved vectorstore.pkl (dim={dim}, vectors={index.ntotal})")


if __name__ == "__main__":
    main()
