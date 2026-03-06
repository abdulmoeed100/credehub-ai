# Libraries import karo
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1 — PDF load karne ka function
def load_pdf(pdf_path):
    print(f"PDF load ho rahi hai: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Total pages: {len(pages)}")
    return pages

# Step 2 — PDF ko chunks mein todne ka function
def split_into_chunks(pages):
    print("PDF ko chunks mein tod raha hun...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Har chunk 1000 characters ka
        chunk_overlap=100     # Chunks ke beech 100 characters overlap
    )
    chunks = splitter.split_documents(pages)
    print(f"Total chunks bane: {len(chunks)}")
    return chunks

# Step 3 — Chunks ko FAISS mein save karo
def create_vector_store(chunks):
    print("Embeddings ban rahi hain... (thoda time lagega)")
    
    # Free embeddings model — HuggingFace se
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # FAISS vector store banao
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Local mein save karo
    vector_store.save_local("data/faiss_index")
    print("Vector store ban gaya aur save ho gaya! ✅")
    
    return vector_store

# Step 4 — Vector store load karo
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

# Step 5 — Sawal se related chunks dhundo
def search_relevant_chunks(query, vector_store, k=3):
    results = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

# Test karo — seedha run karne pe
if __name__ == "__main__":
    # PDF load karo
    pages = load_pdf("data/computerclass9.pdf")
    
    # Chunks banao
    chunks = split_into_chunks(pages)
    
    # Vector store banao
    create_vector_store(chunks)
    
    print("\n✅ RAG setup complete!")