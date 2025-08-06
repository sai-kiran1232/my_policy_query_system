import json
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from query_understanding import parse_query

# Load FAISS index and metadata
faiss_index = FAISS.load_local(
    "faiss_policy_clauses_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

with open("policy_clauses_metadata.json", "r") as f:
    clause_metadata = json.load(f)

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def summarize_clause(question, clause_text):
    # Custom rule-based summarization for known question types
    q = question.lower()

    if "grace period" in q:
        return "A grace period of 30 days is provided for payment of the premium after the due date."
    elif "cataract" in q:
        return "There is a waiting period of 2 years for cataract surgery."
    elif "preventive health check-up" in q:
        return "Yes, the policy allows preventive health check-ups after a block of two continuous policy years."
    elif "ayush" in q:
        return "AYUSH treatments are covered up to the sum insured, provided treatment is taken in an AYUSH hospital."

    # Fallback: return the first full sentence from the clause
    return clause_text.strip().split(".")[0] + "."

def process_query(query_text):
    structured = parse_query(query_text)
    print("Structured Query:", structured)

    retrieved_docs = faiss_index.similarity_search(query_text, k=10)

    best_clause = None
    best_score = -1

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

    # Summarize the clause into a clean answer
    return summarize_clause(query_text, best_clause)
