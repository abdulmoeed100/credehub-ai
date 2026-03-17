# Libraries import karo
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ─────────────────────────────────────────────────────
# SUBJECTS CONFIG
# Nayi subject add karni ho toh sirf yahan ek entry add karo
# Baaki sab automatic hoga
# ─────────────────────────────────────────────────────
SUBJECTS = {
    "computer_science_9": {
        "pdf":     "data/pdfs/computerclass9.pdf",
        "index":   "data/faiss_index/computer_science_9",
        "subject": "Computer Science",
        "grade":   9
    },
    # Future subjects — PDF ready hone pe uncomment karo:
    # "physics_9": {
    #     "pdf":     "data/pdfs/physics9.pdf",
    #     "index":   "data/faiss_index/physics_9",
    #     "subject": "Physics",
    #     "grade":   9
    # },
    # "chemistry_9": {
    #     "pdf":     "data/pdfs/chemistry9.pdf",
    #     "index":   "data/faiss_index/chemistry_9",
    #     "subject": "Chemistry",
    #     "grade":   9
    # },
}

# Step 1 — PDF load karo + metadata add karo
def load_pdf(pdf_path, subject, grade):
    print(f"  PDF load ho rahi hai: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Har page mein metadata add karo
    for i, page in enumerate(pages):
        page.metadata["page_number"] = i + 1      # Page number
        page.metadata["subject"]     = subject     # Subject name
        page.metadata["grade"]       = grade       # Class number

    print(f"  Total pages loaded: {len(pages)}")
    return pages

# Step 2 — Chunks banao (metadata automatically preserve hoga)
def split_into_chunks(pages):
    print("  Chunks ban rahe hain...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    print(f"  Total chunks: {len(chunks)}")

    # Pehle 2 chunks ka metadata verify karo
    print("  Sample metadata check:")
    for i, chunk in enumerate(chunks[:2]):
        print(f"    Chunk {i+1}: {chunk.metadata}")

    return chunks

# Step 3 — FAISS index banao aur save karo
def create_vector_store(chunks, index_path):
    print("  Embeddings ban rahi hain... (thoda time lagega)")
    os.makedirs(index_path, exist_ok=True)
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    print(f"  Index save ho gaya: {index_path} ✅")
    return vector_store

# Step 4 — Index load karo
def load_vector_store(index_path):
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

# Step 5 — Sawaal se related chunks dhundo
def search_relevant_chunks(query, vector_store, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results

# ─────────────────────────────────────────────────────
# Run karo: python rag.py
# Sab subjects ka index ban jayega
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Credehub RAG Setup ===\n")

    for key, info in SUBJECTS.items():
        print(f"Processing: {info['subject']} Class {info['grade']}")

        if not os.path.exists(info["pdf"]):
            print(f"  PDF nahi mila: {info['pdf']} — skip\n")
            continue

        pages  = load_pdf(info["pdf"], info["subject"], info["grade"])
        chunks = split_into_chunks(pages)
        create_vector_store(chunks, info["index"])
        print(f"  {info['subject']} Class {info['grade']} complete!\n")

    print("=== Setup Complete! ===")
