# verify_index.py  — run this after ingest.py to confirm everything saved
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)

# Load from disk
vectorstore = FAISS.load_local(
    "./faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
    # This flag is required because the index uses pickle internally.
    # Safe here because YOU created this file — it's your own data.
)

# Quick check
results = vectorstore.similarity_search("neural network architecture", k=1)
print("Index loaded successfully!")
print(f"Sample result: {results[0].page_content[:200]}")