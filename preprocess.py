# Step 1: Document Preprocessing & Storage
# Dependencies setup

import os
import fitz  # PyMuPDF
import json
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# Paths
DOC_FOLDER = 'C:/Users/saiki/OneDrive/Documents/my_policy_query_system/docs'

 # Assuming all your doc1, doc2... are stored here

# Load model for embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function: Extract text from PDF and segment into clauses
def extract_and_segment_clauses(pdf_path):
    doc = fitz.open(pdf_path)
    clauses = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        # Basic segmentation by line breaks, can be replaced by regex for more complex structures
        lines = text.split('\n')
        for line in lines:
            if len(line.strip()) > 50:  # Skip short lines (like headers, footers)
                clauses.append({
                    "pdf_name": os.path.basename(pdf_path),
                    "page": page_num + 1,
                    "text": line.strip()
                })
    return clauses

# Step 2: Build Clause Index
all_clauses = []
for filename in os.listdir(DOC_FOLDER):
    if filename.endswith('.pdf'):
        file_path = os.path.join(DOC_FOLDER, filename)
        file_clauses = extract_and_segment_clauses(file_path)
        all_clauses.extend(file_clauses)

print(f"Extracted {len(all_clauses)} clauses")

# Step 3: Generate embeddings and store
texts = [clause['text'] for clause in all_clauses]
embeddings = model.encode(texts, show_progress_bar=True)

# Optional: Save as FAISS index
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_index = FAISS.from_texts(texts, embeddings_model)
faiss_index.save_local("faiss_policy_clauses_index")

# Save metadata
with open("policy_clauses_metadata.json", "w") as f:
    json.dump(all_clauses, f, indent=2)

print("✅ Document preprocessing and index build complete.")
