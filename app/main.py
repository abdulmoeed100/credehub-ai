# Import libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import pickle
import os
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Credehub AI API",
    description="Karachi Board Class 9 & 10 AI Assistant",
    version="1.0.0"
)

# CORS — allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================
# EMBEDDINGS — BAAI/bge-small-en-v1.5 (Better for retrieval)
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ============================================================
# LOAD FAISS + BM25
# ============================================================
def load_retrievers(index_path, chunks_path):
    """Load FAISS and BM25 retrievers separately."""
    
    # Load FAISS
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Load chunks for BM25
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4
    
    return vector_store, bm25_retriever, chunks

def hybrid_search(query, vector_store, bm25_retriever, chunks, k=8):
    """Combine FAISS (semantic) + BM25 (keyword) search manually."""
    
    # FAISS semantic search
    semantic_results = vector_store.similarity_search(query, k=k)
    
    # BM25 search
    bm25_results = bm25_retriever.invoke(query)
    
    # Combine results (remove duplicates by content)
    all_docs = semantic_results + bm25_results
    unique_docs = []
    seen = set()
    
    for doc in all_docs:
        key = doc.page_content[:300]
        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)
    
    return unique_docs[:k]

# Load Computer Science
CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS = load_retrievers(
    "data/faiss_index/computer_science_9",
    "data/faiss_index/computer_science_9/chunks.pkl"
)

# ============================================================
# PAGE TO UNIT MAPPING (Actual book pages)
# ============================================================
def get_unit_from_page(page_num):
    if 1 <= page_num <= 29:
        return "Unit 1 - Fundamentals of Computer"
    elif 30 <= page_num <= 45:
        return "Unit 2 - Fundamentals of Operating System"
    elif 46 <= page_num <= 64:
        return "Unit 3 - Office Automation"
    elif 65 <= page_num <= 94:
        return "Unit 4 - Data Communication and Computer Networks"
    elif 95 <= page_num <= 116:
        return "Unit 5 - Computer Security and Ethics"
    elif 117 <= page_num <= 141:
        return "Unit 6 - Web Development"
    elif 142 <= page_num <= 162:
        return "Unit 7 - Introduction to Database System"
    return None

# ============================================================
# UNIT KEYWORD DETECTION
# ============================================================
UNIT_KEYWORDS = {
    "unit 1": "Unit 1 - Fundamentals of Computer",
    "unit 2": "Unit 2 - Fundamentals of Operating System",
    "unit 3": "Unit 3 - Office Automation",
    "unit 4": "Unit 4 - Data Communication and Computer Networks",
    "unit 5": "Unit 5 - Computer Security and Ethics",
    "unit 6": "Unit 6 - Web Development",
    "unit 7": "Unit 7 - Introduction to Database System",
    "computer": "Unit 1 - Fundamentals of Computer",
    "hardware": "Unit 1 - Fundamentals of Computer",
    "software": "Unit 1 - Fundamentals of Computer",
    "cpu": "Unit 1 - Fundamentals of Computer",
    "ram": "Unit 1 - Fundamentals of Computer",
    "generation": "Unit 1 - Fundamentals of Computer",
    "operating system": "Unit 2 - Fundamentals of Operating System",
    "windows": "Unit 2 - Fundamentals of Operating System",
    "ms word": "Unit 3 - Office Automation",
    "ms excel": "Unit 3 - Office Automation",
    "excel": "Unit 3 - Office Automation",
    "network": "Unit 4 - Data Communication and Computer Networks",
    "topology": "Unit 4 - Data Communication and Computer Networks",
    "transmission": "Unit 4 - Data Communication and Computer Networks",
    "security": "Unit 5 - Computer Security and Ethics",
    "malware": "Unit 5 - Computer Security and Ethics",
    "virus": "Unit 5 - Computer Security and Ethics",
    "ethics": "Unit 5 - Computer Security and Ethics",
    "html": "Unit 6 - Web Development",
    "web": "Unit 6 - Web Development",
    "website": "Unit 6 - Web Development",
    "hyperlink": "Unit 6 - Web Development",
    "database": "Unit 7 - Introduction to Database System",
    "dbms": "Unit 7 - Introduction to Database System",
    "sql": "Unit 7 - Introduction to Database System",
}

def detect_unit(question: str):
    """Detect which unit the question is about."""
    q = question.lower()
    
    # First: Check for page number
    page_match = re.search(r'page\s+(\d+)', q)
    if page_match:
        page_num = int(page_match.group(1))
        unit = get_unit_from_page(page_num)
        if unit:
            return unit
    
    # Second: Check for chapter number
    chapter_match = re.search(r'chapter\s+(\d+)', q)
    if chapter_match:
        chapter_num = int(chapter_match.group(1))
        chapter_to_unit = {
            1: "Unit 1 - Fundamentals of Computer",
            2: "Unit 2 - Fundamentals of Operating System",
            3: "Unit 3 - Office Automation",
            4: "Unit 4 - Data Communication and Computer Networks",
            5: "Unit 5 - Computer Security and Ethics",
            6: "Unit 6 - Web Development",
            7: "Unit 7 - Introduction to Database System",
        }
        return chapter_to_unit.get(chapter_num)
    
    # Third: Check for topic keywords
    for keyword, unit in UNIT_KEYWORDS.items():
        if keyword in q:
            return unit
    
    return None

# ============================================================
# REQUEST FORMAT
# ============================================================
class ChatRequest(BaseModel):
    question: str
    history: List[Dict] = []
    subject: str = "Computer Science"
    grade: int = 9

# ============================================================
# ENDPOINT 1 — Health check
# ============================================================
@app.get("/")
def home():
    return {
        "status": "Credehub AI is running! ✅",
        "version": "1.0.0"
    }

# ============================================================
# ENDPOINT 2 — Chat
# ============================================================
@app.post("/chat")
def chat(request: ChatRequest):
    
    question_lower = request.question.lower()
    
    # CHECK FOR PAGE NUMBER
    page_match = re.search(r'page\s+(\d+)', question_lower)
    page_num = int(page_match.group(1)) if page_match else None
    
    if page_num:
        # Try 1: FAISS with metadata filter (actual_page_number)
        results = CS_VECTOR_STORE.similarity_search(
            "", k=50, filter={"actual_page_number": page_num}
        )
        
        # Try 2: If no results, try pdf_page_number (with offset +6)
        if not results:
            pdf_page = page_num + 6
            results = CS_VECTOR_STORE.similarity_search(
                "", k=50, filter={"pdf_page_number": pdf_page}
            )
        
        # Try 3: If still no results, use BM25 keyword search
        if not results:
            results = CS_BM25_RETRIEVER.invoke(f"page {page_num}")
        
        # Try 4: Last resort — normal similarity search
        if not results:
            results = hybrid_search(
                f"page {page_num} content", 
                CS_VECTOR_STORE, 
                CS_BM25_RETRIEVER, 
                CS_CHUNKS, 
                k=10
            )
        
        if results:
            # Combine all chunks of this page
            full_content = "\n\n".join([doc.page_content for doc in results])
            unit_name = get_unit_from_page(page_num) or "Computer Science"
            
            context = f"""[{unit_name} | Class 9 | Page {page_num} - COMPLETE PAGE CONTENT]

{full_content}"""
        else:
            context = f"No content found for page {page_num}. Available pages are 1 to 162."
    
    else:
        # Normal query — check for unit detection
        detected_unit = detect_unit(request.question)
        
        if detected_unit:
            # Unit detected — use FAISS with metadata filter
            results = CS_VECTOR_STORE.similarity_search(
                request.question,
                k=6,
                filter={"unit": detected_unit}
            )
            if not results:
                results = hybrid_search(request.question, CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS, k=8)
        else:
            # No unit detected — use hybrid search
            results = hybrid_search(request.question, CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS, k=8)
        
        # Build context with metadata
        context_parts = []
        for doc in results:
            m = doc.metadata
            page_display = m.get('actual_page_number', m.get('pdf_page_number', '?'))
            context_parts.append(
                f"[{m.get('unit', '?')} | Page {page_display}]\n{doc.page_content}"
            )
        context = "\n\n".join(context_parts)
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": f"""You are Credehub AI — an assistant for Karachi Board Class 9 and 10 students.

STRICT RULES:

1. Answer ONLY from the curriculum content below.
2. If the answer is not found, say: "Is topic ka jawab curriculum mein nahi hai. Apne teacher se poochein."
3. If user asked about a specific page, give COMPLETE information from that page.
4. Give detailed answer in 6-8 lines with examples where possible.
5. At the end of every answer, mention the source:
   📚 Source: [Unit Name] | Page [Number]
6. NEVER use your own knowledge — only what is in the content below.

CURRICULUM CONTENT:
{context}"""
        }
    ]
    
    # Add chat history — last 8 messages only
    for msg in request.history[-8:]:
        messages.append(msg)
    
    # Add new question
    messages.append({
        "role": "user",
        "content": request.question
    })
    
    # Send to Groq
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        max_tokens=800,
        messages=messages
    )
    
    # Remove think tags
    answer = response.choices[0].message.content
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()
    
    return {
        "question": request.question,
        "answer": answer,
        "subject": request.subject,
        "grade": request.grade
    }

# ============================================================
# ENDPOINT 3 — Subjects list
# ============================================================
@app.get("/subjects")
def get_subjects():
    return {
        "subjects": ["Computer Science", "Physics", "Chemistry", "English"],
        "grades": [9, 10]
    }