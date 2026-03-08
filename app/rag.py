# Libraries import karo
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
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
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    print(f"Total chunks bane: {len(chunks)}")
    return chunks

# Step 3 — Chunks ko FAISS mein save karo
def create_vector_store(chunks):
    print("Embeddings ban rahi hain... (thoda time lagega)")
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("data/faiss_index")
    print("Vector store ban gaya aur save ho gaya! ✅")
    return vector_store

# Step 4 — Vector store load karo
def load_vector_store():
    embeddings = FastEmbedEmbeddings()
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
    pages = load_pdf("data/computerclass9.pdf")
    chunks = split_into_chunks(pages)
    create_vector_store(chunks)
    print("\n✅ RAG setup complete!")