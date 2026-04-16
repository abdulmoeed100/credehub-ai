# Import libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber
import pickle
import re
import os

# ─────────────────────────────────────────────────────
# SUBJECTS CONFIG
# To add a new subject — add one entry here
# ─────────────────────────────────────────────────────
SUBJECTS = {
    "computer_science_9": {
        "pdf":     "data/pdfs/computerclass9.pdf",
        "index":   "data/faiss_index/computer_science_9",
        "chunks":  "data/faiss_index/computer_science_9/chunks.pkl",
        "subject": "Computer Science",
        "grade":   9
    },
    # Uncomment when PDF is ready:
    # "physics_9": {
    #     "pdf":     "data/pdfs/physics9.pdf",
    #     "index":   "data/faiss_index/physics_9",
    #     "chunks":  "data/faiss_index/physics_9/chunks.pkl",
    #     "subject": "Physics",
    #     "grade":   9
    # },
}

# ─────────────────────────────────────────────────────
# UNIT MAP — page ranges to unit names
# ─────────────────────────────────────────────────────
UNIT_MAP = [
    (range(7,  36),  "Unit 1 - Fundamentals of Computer"),
    (range(36, 52),  "Unit 2 - Fundamentals of Operating System"),
    (range(52, 71),  "Unit 3 - Office Automation"),
    (range(71, 101), "Unit 4 - Data Communication and Computer Networks"),
    (range(101,123), "Unit 5 - Computer Security and Ethics"),
    (range(123,148), "Unit 6 - Web Development"),
    (range(148,169), "Unit 7 - Introduction to Database System"),
]

def get_unit(pdf_page_num):
    """Return unit name based on PDF page number."""
    for page_range, unit_name in UNIT_MAP:
        if pdf_page_num in page_range:
            return unit_name
    return "Introduction"

# ─────────────────────────────────────────────────────
# TOPIC DETECTION — font size 14 = heading in this PDF
# ─────────────────────────────────────────────────────
def extract_topics_from_pdf(pdf_path):
    """Scan PDF and detect topic headings using font size analysis."""
    print("  Detecting topics from PDF...")

    topic_map     = {}
    current_topic = "General"

    skip_patterns = [
        r'^(SUMMARY|EXERCISE|ACTIVITIES|WEBLINKS?)$',
        r'^\d+$',
        r'^[A-Z]\.$',
        r'(Normal State|Shift State)',
        r'^(Protocol|Messages|Sender|Receiver|Analog|Digital|Hybrid)$',
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for pg_idx in range(6, len(pdf.pages)):
            pdf_page_num = pg_idx + 1
            page  = pdf.pages[pg_idx]
            words = page.extract_words(extra_attrs=["fontname", "size"])

            lines = {}
            for w in words:
                size = round(w.get('size', 0), 0)
                top  = round(w.get('top',  0), -1)
                if size == 14:
                    if top not in lines:
                        lines[top] = []
                    lines[top].append(w['text'])

            for top in sorted(lines.keys()):
                line_text = ' '.join(lines[top]).strip()

                should_skip = any(
                    re.search(p, line_text, re.IGNORECASE)
                    for p in skip_patterns
                )
                has_topic_num = bool(re.search(r'^\d+\.\d+', line_text))

                if (has_topic_num or len(line_text) > 10) and not should_skip:
                    half = len(line_text) // 2
                    if line_text[:half].strip() == line_text[half:].strip():
                        line_text = line_text[:half].strip()
                    line_text = re.sub(r'\s+\d+\.\d+\s*$', '', line_text).strip()
                    current_topic = line_text
                    break

            topic_map[pdf_page_num] = current_topic

    print(f"  Topics detected: {len(set(topic_map.values()))} unique topics")
    return topic_map

# ─────────────────────────────────────────────────────
# LOAD PDF AND ADD METADATA
# ─────────────────────────────────────────────────────
def load_pdf_with_metadata(pdf_path, subject, grade):
    """Load PDF and add full metadata to every page."""
    print(f"  Loading PDF: {pdf_path}")

    topic_map = extract_topics_from_pdf(pdf_path)

    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"  Total pages: {len(pages)}")

    for i, page in enumerate(pages):
        pdf_page_num = i + 1
        page.metadata["pdf_page_number"] = pdf_page_num
        page.metadata["subject"]         = subject
        page.metadata["grade"]           = grade
        page.metadata["unit"]            = get_unit(pdf_page_num)
        page.metadata["topic"]           = topic_map.get(pdf_page_num, "General")
        page.metadata["board"]           = "Karachi Board"

    print("\n  Sample metadata (first 3 pages):")
    for page in pages[:3]:
        m = page.metadata
        print(f"    Page {m['pdf_page_number']:3d} | {m['unit'][:35]:35s} | {m['topic'][:40]}")

    return pages

# ─────────────────────────────────────────────────────
# SPLIT INTO CHUNKS
# ─────────────────────────────────────────────────────
def split_into_chunks(pages):
    """Split pages into chunks — metadata is preserved automatically."""
    print("\n  Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    print(f"  Total chunks: {len(chunks)}")

    print("\n  Sample chunk metadata:")
    for i, chunk in enumerate(chunks[:3]):
        m = chunk.metadata
        print(f"    Chunk {i+1}: Page {m.get('pdf_page_number','?')} | "
              f"{m.get('unit','?')[:30]} | {m.get('topic','?')[:35]}")

    return chunks

# ─────────────────────────────────────────────────────
# CREATE FAISS INDEX + SAVE CHUNKS FOR BM25
# ─────────────────────────────────────────────────────
def create_vector_store(chunks, index_path, chunks_path):
    """Create FAISS index and also save chunks for BM25 retriever."""
    print("\n  Generating embeddings... (2-3 minutes)")

    os.makedirs(index_path, exist_ok=True)

    # Save FAISS index
    embeddings   = FastEmbedEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    print(f"  FAISS index saved: {index_path} ✅")

    # Save chunks for BM25 — pickle file
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  Chunks saved for BM25: {chunks_path} ✅")

    return vector_store

# ─────────────────────────────────────────────────────
# LOAD EXISTING INDEX
# ─────────────────────────────────────────────────────
def load_vector_store(index_path):
    """Load existing FAISS index from disk."""
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

# ─────────────────────────────────────────────────────
# MAIN — run with: python app/rag.py
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   Credehub RAG Setup")
    print("=" * 50)

    for key, info in SUBJECTS.items():
        print(f"\nProcessing: {info['subject']} Class {info['grade']}\n")

        if not os.path.exists(info["pdf"]):
            print(f"  PDF not found: {info['pdf']} — skipping\n")
            continue

        pages  = load_pdf_with_metadata(info["pdf"], info["subject"], info["grade"])
        chunks = split_into_chunks(pages)
        create_vector_store(chunks, info["index"], info["chunks"])

        print(f"\n  {info['subject']} Class {info['grade']} — Complete!\n")

    print("=" * 50)
    print("   Setup Complete!")
    print("=" * 50)
