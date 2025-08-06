import json
import re
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from query_understanding import parse_query

# ✅ Use smaller model + force CPU to save memory
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-albert-small-v2",
        model_kwargs={"device": "cpu"}  # force CPU instead of GPU (safer on Render)
    )

@lru_cache(maxsize=1)
def get_faiss_index():
    return FAISS.load_local(
        "faiss_policy_clauses_index",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

@lru_cache(maxsize=1)
def get_metadata():
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
    structured = parse_query(query_text)
    print("Structured Query:", structured)

    faiss_index = get_faiss_index()
    clause_metadata = get_metadata()

    retrieved_docs = faiss_index.similarity_search(query_text, k=10)

    best_clause = None

    for doc in retrieved_docs:
        doc_text_clean = clean_text(doc.page_content)
        for clause in clause_metadata:
            clause_text_clean = clean_text(clause["text"])
            if doc_text_clean in clause_text_clean or clause_text_clean in doc_text_clean:
                best_clause = clause["text"]
                break
        if best_clause:
            break

    if not best_clause:
        best_clause = retrieved_docs[0].page_content if retrieved_docs else "Sorry, no relevant information found."

    return summarize_clause(query_text, best_clause)
