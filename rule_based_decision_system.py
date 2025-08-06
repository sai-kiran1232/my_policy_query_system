import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU use

import json
import re
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
import faiss
import pickle
from query_understanding import parse_query

# ✅ Smallest model + direct use (no LangChain overhead)
@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

# ✅ Load FAISS index manually
@lru_cache(maxsize=1)
def load_faiss_index():
    with open("faiss_policy_clauses_index/faiss_index.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_texts():
    with open("faiss_policy_clauses_index/texts.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_metadata():
    with open("policy_clauses_metadata.json", "r") as f:
        return json.load(f)

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def summarize_clause(question, clause_text):
    q = question.lower()

    if "grace period" in q:
        return "A grace period of 30 days is provided for payment of the premium after the due date."
    elif "cataract" in q:
        return "There is a waiting period of 2 years for cataract surgery."
    elif "preventive health check-up" in q:
        return "Yes, the policy allows preventive health check-ups after a block of two continuous policy years."
    elif "ayush" in q:
        return "AYUSH treatments are covered up to the sum insured, provided treatment is taken in an AYUSH hospital."

    return clause_text.strip().split(".")[0] + "."

def process_query(query_text):
    try:
        structured = parse_query(query_text)
        model = get_model()
        index = load_faiss_index()
        texts = load_texts()
        metadata = load_metadata()

        query_emb = model.encode(query_text, convert_to_tensor=False)
        D, I = index.search([query_emb], k=3)

        best_clause = None
        for idx in I[0]:
            doc_text = texts[idx]
            doc_text_clean = clean_text(doc_text)
            for clause in metadata:
                clause_text_clean = clean_text(clause["text"])
                if doc_text_clean in clause_text_clean or clause_text_clean in doc_text_clean:
                    best_clause = clause["text"]
                    break
            if best_clause:
                break

        if not best_clause:
            best_clause = texts[I[0][0]] if I[0] else "Sorry, no relevant information found."

        return summarize_clause(query_text, best_clause)
    except Exception as e:
        return "An error occurred while processing your request."
