# Import libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
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

# ─────────────────────────────────────────────────────
# Load FAISS + BM25 for each subject at startup
# ─────────────────────────────────────────────────────
embeddings = FastEmbedEmbeddings()

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
    bm25_retriever.k = 3
    
    return vector_store, bm25_retriever, chunks

def hybrid_search(query, vector_store, bm25_retriever, chunks, k=6):
    """Combine FAISS (semantic) + BM25 (keyword) search manually."""
    
    # FAISS semantic search
    semantic_results = vector_store.similarity_search(query, k=k)
    
    # BM25 search
    bm25_results = bm25_retriever.invoke(query)
    
    # Combine results (remove duplicates by content)
    all_docs = semantic_results + bm25_results
    unique_docs = []
    seen_content = set()
    
    for doc in all_docs:
        # Use first 500 chars as unique identifier
        content_key = doc.page_content[:500]
        if content_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_key)
    
    return unique_docs[:k]

# Load Computer Science
CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS = load_retrievers(
    "data/faiss_index/computer_science_9",
    "data/faiss_index/computer_science_9/chunks.pkl"
)

# ─────────────────────────────────────────────────────
# Unit keyword detection
# ─────────────────────────────────────────────────────
UNIT_KEYWORDS = {
    "unit 1": "Unit 1 - Fundamentals of Computer",
    "unit 2": "Unit 2 - Fundamentals of Operating System",
    "unit 3": "Unit 3 - Office Automation",
    "unit 4": "Unit 4 - Data Communication and Computer Networks",
    "unit 5": "Unit 5 - Computer Security and Ethics",
    "unit 6": "Unit 6 - Web Development",
    "unit 7": "Unit 7 - Introduction to Database System",
    # Topic keywords
    "computer":           "Unit 1 - Fundamentals of Computer",
    "hardware":           "Unit 1 - Fundamentals of Computer",
    "software":           "Unit 1 - Fundamentals of Computer",
    "cpu":                "Unit 1 - Fundamentals of Computer",
    "ram":                "Unit 1 - Fundamentals of Computer",
    "generation":         "Unit 1 - Fundamentals of Computer",
    "operating system":   "Unit 2 - Fundamentals of Operating System",
    "windows":            "Unit 2 - Fundamentals of Operating System",
    "ms word":            "Unit 3 - Office Automation",
    "ms excel":           "Unit 3 - Office Automation",
    "excel":              "Unit 3 - Office Automation",
    "network":            "Unit 4 - Data Communication and Computer Networks",
    "topology":           "Unit 4 - Data Communication and Computer Networks",
    "transmission":       "Unit 4 - Data Communication and Computer Networks",
    "security":           "Unit 5 - Computer Security and Ethics",
    "malware":            "Unit 5 - Computer Security and Ethics",
    "virus":              "Unit 5 - Computer Security and Ethics",
    "ethics":             "Unit 5 - Computer Security and Ethics",
    "html":               "Unit 6 - Web Development",
    "web":                "Unit 6 - Web Development",
    "website":            "Unit 6 - Web Development",
    "hyperlink":          "Unit 6 - Web Development",
    "database":           "Unit 7 - Introduction to Database System",
    "dbms":               "Unit 7 - Introduction to Database System",
    "sql":                "Unit 7 - Introduction to Database System",
}

def detect_unit(question: str):
    """Detect which unit the question is about."""
    q = question.lower()
    for keyword, unit in UNIT_KEYWORDS.items():
        if keyword in q:
            return unit
    return None

# ─────────────────────────────────────────────────────
# Request format
# ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history:  List[Dict] = []
    subject:  str = "Computer Science"
    grade:    int = 9

# ─────────────────────────────────────────────────────
# Endpoint 1 — Health check
# ─────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "Credehub AI is running! ✅",
        "version": "1.0.0"
    }

# ─────────────────────────────────────────────────────
# Endpoint 2 — Chat
# ─────────────────────────────────────────────────────
@app.post("/chat")
def chat(request: ChatRequest):
    
    # Detect unit from question
    detected_unit = detect_unit(request.question)
    
    if detected_unit:
        # Unit detected — use FAISS with metadata filter for precision
        results = CS_VECTOR_STORE.similarity_search(
            request.question,
            k=5,
            filter={"unit": detected_unit}
        )
        # Fallback — if filter returns nothing, use hybrid search
        if not results:
            results = hybrid_search(request.question, CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS, k=6)
    else:
        # No unit detected — use hybrid search (FAISS + BM25)
        results = hybrid_search(request.question, CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS, k=6)
    
    # Build context with full metadata
    context_parts = []
    for doc in results:
        m = doc.metadata
        context_parts.append(
            f"[{m.get('subject','?')} Class {m.get('grade','?')} | "
            f"{m.get('unit','?')} | "
            f"Topic: {m.get('topic','?')} | "
            f"Page {m.get('pdf_page_number','?')}]\n"
            f"{doc.page_content}"
        )
    context = "\n\n".join(context_parts)
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": f"""/no_think
You are Credehub AI — an assistant for Karachi Board Class 9 and 10 students in Pakistan.

STRICT RULES — FOLLOW EXACTLY:

1. Answer ONLY from the curriculum content provided below in the brackets [].
   - NEVER use your own knowledge to fill gaps.
   - NEVER mention topics, units, or concepts not present in the content below.
   - If the answer is not found, say: "This topic is not found in the curriculum. Please ask your teacher."

2. LANGUAGE DETECTION:
   - Student writes in ENGLISH or says "in english"          → reply in English only.
   - Student writes in ROMAN URDU or says "urdu mein batao"  → reply in Roman Urdu only.
   - Student writes in URDU SCRIPT                           → reply in Roman Urdu only.
   - Student writes in ROMAN PUNJABI or says "punjabi mein"  → reply in Roman Punjabi only.
   - DEFAULT is English if language cannot be detected.
   - NEVER say "I cannot respond in this language."
   - NEVER mix two languages in one reply.

3. ROMAN URDU STYLE:
   Example: "Computer ek electronic machine hai jo data process karta hai."

4. ROMAN PUNJABI STYLE:
   Example: "Computer ik electronic machine hai jo data process karda hai."

5. Give a detailed answer in 8-10 lines with examples where possible.

6. At the end of every answer mention the source:
   📚 Source: [Unit Name] | Topic: [Topic Name] | Page [Number]

7. Spellings must always be accurate in all languages.

Curriculum Content:
{context}"""
        }
    ]
    
    # Add chat history — last 10 messages only
    for msg in request.history[-10:]:
        messages.append(msg)
    
    # Add new question
    messages.append({
        "role": "user",
        "content": request.question
    })
    
    # Send to Groq
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        max_tokens=600,
        messages=messages
    )
    
    # Remove think tags
    answer = response.choices[0].message.content
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    answer = re.sub(r'<think>.*',          '', answer, flags=re.DOTALL).strip()
    
    return {
        "question": request.question,
        "answer":   answer,
        "subject":  request.subject,
        "grade":    request.grade
    }

# ─────────────────────────────────────────────────────
# Endpoint 3 — Subjects list
# ─────────────────────────────────────────────────────
@app.get("/subjects")
def get_subjects():
    return {
        "subjects": ["Computer Science", "Physics", "Chemistry", "English"],
        "grades":   [9, 10]
    }