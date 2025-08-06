import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use the same model as during indexing
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Load the original LangChain FAISS vector store
faiss_store = FAISS.load_local("faiss_policy_clauses_index", embedding, allow_dangerous_deserialization=True)

# Extract:
# 1. The raw FAISS index
# 2. The original document texts
faiss_index = faiss_store.index
texts = [doc.page_content for doc in faiss_store.similarity_search("test")]

# Save the raw FAISS index
with open("faiss_policy_clauses_index/faiss_index.pkl", "wb") as f:
    pickle.dump(faiss_index, f)

# Save the texts list
with open("faiss_policy_clauses_index/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Successfully exported raw FAISS index and texts to .pkl files")
