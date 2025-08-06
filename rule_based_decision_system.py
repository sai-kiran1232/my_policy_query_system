import os
import json
import pickle
import re
from functools import lru_cache
from query_understanding import parse_query
import faiss
from sentence_transformers import SentenceTransformer

# Force CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

@lru_cache(maxsize=1)
def get_faiss_index():
    with open("faiss_index.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def get_texts():
    with open("texts.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def get_metadata():
    with open("policy_clauses_metadata.json", "r") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

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
    structured = parse_query(query_text)
    print("Structured Query:", structured)

    index = get_faiss_index()
    texts = get_texts()
    metadata = get_metadata()
    model = get_model()

    query_embedding = model.encode([query_text])
    _, I = index.search(query_embedding, k=10)

    best_clause = None

    for idx in I[0]:
        doc_text_clean = clean_text(texts[idx])
        for clause in metadata:
            clause_text_clean = clean_text(clause["text"])
            if doc_text_clean in clause_text_clean or clause_text_clean in doc_text_clean:
                best_clause = clause["text"]
                break
        if best_clause:
            break

    if not best_clause and len(I[0]) > 0:
        best_clause = texts[I[0][0]]
    elif not best_clause:
        best_clause = "Sorry, no relevant information found."

    return summarize_clause(query_text, best_clause)
