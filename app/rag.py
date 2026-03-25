# Libraries import karo
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
import pdfplumber
import re
import os

# ─────────────────────────────────────────────────────
# SUBJECTS CONFIG
# Nayi subject add karni ho toh sirf yahan ek entry add karo
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
}

# ─────────────────────────────────────────────────────
# UNIT MAP — Page ranges se unit name milega
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
    """Page number se unit name nikalo"""
    for page_range, unit_name in UNIT_MAP:
        if pdf_page_num in page_range:
            return unit_name
    return "Introduction"

# ─────────────────────────────────────────────────────
# TOPIC DETECTION — pdfplumber se font size >= 13
# ─────────────────────────────────────────────────────
def extract_topics_from_pdf(pdf_path):
    """
    Poori PDF scan karo aur har page ka topic nikalo.
    Font size 14 = topic heading
    Returns: {pdf_page_num: "topic name"}
    """
    print("  Topics detect ho rahe hain (font analysis)...")
    
    topic_map = {}       # {page_num: topic_name}
    current_topic = "General"
    
    # Skip karne wale patterns
    skip_patterns = [
        r'^(SUMMARY|EXERCISE|ACTIVITIES|WEBLINKS?)$',
        r'^\d+$',                          # sirf numbers
        r'^[A-Z]\.$',                      # single letters
        r'(Normal State|Shift State)',      # keyboard diagrams
        r'^(Protocol|Messages|Sender|Receiver|Analog|Digital|Hybrid)$',  # diagram labels
    ]
    
    with pdfplumber.open(pdf_path) as pdf:
        for pg_idx in range(6, len(pdf.pages)):  # page 7 se shuru
            pdf_page_num = pg_idx + 1
            page = pdf.pages[pg_idx]
            words = page.extract_words(extra_attrs=["fontname", "size"])
            
            # Lines reconstruct karo (same top position = same line)
            lines = {}
            for w in words:
                size = round(w.get('size', 0), 0)
                top  = round(w.get('top',  0), -1)  # 10px tolerance
                
                if size == 14:  # topic heading size
                    key = top
                    if key not in lines:
                        lines[key] = []
                    lines[key].append(w['text'])
            # Lines ko text mein convert karo
            for top in sorted(lines.keys()):
                line_text = ' '.join(lines[top]).strip()
                
                # Skip patterns check karo
                should_skip = any(re.search(p, line_text, re.IGNORECASE) 
                                 for p in skip_patterns)
                
                # Topic number pattern check karo — jaise "1.1", "4.2", "6.9"
                has_topic_num = bool(re.search(r'^\d+\.\d+', line_text))
                
                # Valid topic hai?
                if (has_topic_num or len(line_text) > 10) and not should_skip:
                    # Duplicate text remove karo (PDF rendering issue)
                    # "1.1 INTRODUCTION TO COMPUTER 1.1 INTRODUCTION TO COMPUTER"
                    half = len(line_text) // 2
                    if line_text[:half].strip() == line_text[half:].strip():
                        line_text = line_text[:half].strip()
                    
                    # Topic number suffix remove karo — "1.1 INTRODUCTION 1.1" → "1.1 INTRODUCTION"
                    line_text = re.sub(r'\s+\d+\.\d+\s*$', '', line_text).strip()
                    
                    current_topic = line_text
                    break  # Pehla valid heading lo
            
            topic_map[pdf_page_num] = current_topic
    
    print(f"  Topics detected: {len(set(topic_map.values()))} unique topics")
    return topic_map

# ─────────────────────────────────────────────────────
# PDF LOAD + METADATA ADD
# ─────────────────────────────────────────────────────
def load_pdf_with_metadata(pdf_path, subject, grade):
    """PDF load karo aur har page mein metadata add karo"""
    print(f"  PDF load ho rahi hai: {pdf_path}")
    
    # Topics pehle extract karo
    topic_map = extract_topics_from_pdf(pdf_path)
    
    # LangChain se pages load karo
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"  Total pages: {len(pages)}")
    
    # Har page mein metadata add karo
    for i, page in enumerate(pages):
        pdf_page_num = i + 1
        
        page.metadata["pdf_page_number"] = pdf_page_num
        page.metadata["subject"]         = subject
        page.metadata["grade"]           = grade
        page.metadata["unit"]            = get_unit(pdf_page_num)
        page.metadata["topic"]           = topic_map.get(pdf_page_num, "General")
        page.metadata["board"]           = "Karachi Board"
    
    # Sample check — pehle 3 pages ka metadata dikhao
    print("\n  Sample metadata (first 3 pages):")
    for page in pages[:3]:
        m = page.metadata
        print(f"    Page {m['pdf_page_number']:3d} | {m['unit'][:35]:35s} | {m['topic'][:40]}")
    
    return pages

# ─────────────────────────────────────────────────────
# CHUNKS BANAO
# ─────────────────────────────────────────────────────
def split_into_chunks(pages):
    """Pages ko chunks mein todo — metadata preserve hoga"""
    print("\n  Chunks ban rahe hain...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    print(f"  Total chunks: {len(chunks)}")
    
    # Sample chunk metadata check
    print("\n  Sample chunk metadata:")
    for i, chunk in enumerate(chunks[:3]):
        m = chunk.metadata
        print(f"    Chunk {i+1}: Page {m.get('pdf_page_number','?')} | {m.get('unit','?')[:30]} | {m.get('topic','?')[:35]}")
    
    return chunks

# ─────────────────────────────────────────────────────
# FAISS INDEX BANAO
# ─────────────────────────────────────────────────────
def create_vector_store(chunks, index_path):
    """Chunks se FAISS index banao aur save karo"""
    print("\n  Embeddings ban rahi hain... (2-3 minute lagenge)")
    
    os.makedirs(index_path, exist_ok=True)
    embeddings   = FastEmbedEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    
    print(f"  Index save ho gaya: {index_path} ✅")
    return vector_store

# ─────────────────────────────────────────────────────
# INDEX LOAD KARO
# ─────────────────────────────────────────────────────
def load_vector_store(index_path):
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

# ─────────────────────────────────────────────────────
# SEARCH — query se related chunks nikalo
# ─────────────────────────────────────────────────────
def search_relevant_chunks(query, vector_store, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results

# ─────────────────────────────────────────────────────
# MAIN — python rag.py se run karo
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   Credehub RAG Setup")
    print("=" * 50)
    
    for key, info in SUBJECTS.items():
        print(f"\nProcessing: {info['subject']} Class {info['grade']}\n")
        
        # PDF exist karta hai?
        if not os.path.exists(info["pdf"]):
            print(f"  ⚠️  PDF nahi mila: {info['pdf']} — skip\n")
            continue
        
        # Steps
        pages  = load_pdf_with_metadata(info["pdf"], info["subject"], info["grade"])
        chunks = split_into_chunks(pages)
        create_vector_store(chunks, info["index"])
        
        print(f"\n  ✅ {info['subject']} Class {info['grade']} complete!\n")
    
    print("=" * 50)
    print("   Setup Complete!")
    print("=" * 50)
