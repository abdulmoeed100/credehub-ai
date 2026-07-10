# Import libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import Groq, AsyncGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import pickle
import os
import re

# Assessment module — add app/ dir to sys.path so import works
# regardless of where uvicorn is launched from
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))

from assessment import (
    AssessmentRequest,
    AssessmentResponse,
    ReportRequest,
    AssessmentReport,
    generate_assessment,
    generate_report,
)

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
from api_rotator import RotatingAsyncGroq
client = RotatingAsyncGroq()

# ============================================================
# EMBEDDINGS — BAAI/bge-small-en-v1.5 (Better for retrieval)
# Matches the committed FAISS index database
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
    """Load FAISS and BM25 retrievers. Frees chunks list after BM25
    is built to save ~100MB RAM on free-tier deployments."""

    # Load FAISS
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Load chunks for BM25 then immediately free the list
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    # Keep a small reference for hybrid_search dedup; free the big list
    chunks_sample = chunks[:5]   # only used for type hints — not stored
    del chunks                   # free ~100MB RAM

    return vector_store, bm25_retriever, chunks_sample

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
# BOOK OVERVIEW METADATA (Karachi Board Class 9 CS)
# ============================================================
BOOK_OVERVIEW_CONTEXT = """
[Book Structure | Class 9 Computer Science Textbook (Karachi Board)]
This textbook has exactly 7 chapters (units):
- Unit 1: Fundamentals of Computer (Syllabus: Introduction to computer, history, generations of computers, hardware, software, CPU, RAM, ROM, input/output devices) (Pages 1 to 29)
- Unit 2: Fundamentals of Operating System (Syllabus: OS introduction, functions, types of OS like CLI and GUI, Windows OS, file management) (Pages 30 to 45)
- Unit 3: Office Automation (Syllabus: MS Word, MS Excel, MS PowerPoint, word processing, spreadsheets, presentations) (Pages 46 to 64)
- Unit 4: Data Communication and Computer Networks (Syllabus: Data communication components, transmission media, computer networks, LAN, WAN, P2P, Client-Server, network topologies) (Pages 65 to 94)
- Unit 5: Computer Security and Ethics (Syllabus: Cybercrime, malware, virus, worms, antivirus, computer ethics, intellectual property rights, privacy) (Pages 95 to 116)
- Unit 6: Web Development (Syllabus: HTML introduction, basic tags, text formatting, hyperlinks, images, tables, forms, CSS basics) (Pages 117 to 141)
- Unit 7: Introduction to Database System (Syllabus: Database concepts, DBMS, relational database, tables, keys, SQL queries) (Pages 142 to 162)
"""

def is_book_structure_query(question: str) -> bool:
    """Detect if the query is asking about the book syllabus, outlines, or chapters."""
    q = question.lower()
    keywords = [
        "chapter", "chapters", "unit", "units", "syllabus", "table of content", 
        "table of contents", "how many unit", "how many chapter", "book structure",
        "book outline", "outline", "cs book", "computer science book", "chapters name", 
        "units name", "chapters list", "units list", "syllabus check", "course outline"
    ]
    return any(kw in q for kw in keywords)


# ============================================================
# FOLLOW-UP QUERY DETECTION + CONTEXT ENRICHMENT
# ============================================================
# Short follow-up phrases that don't contain a topic themselves
_FOLLOWUP_PATTERNS = [
    r"^(explain|samjhao|samjha|bata|batao|tell me|describe|describe it|isko|is ko|yeh|ye)\b",
    r"\b(in roman urdu|roman urdu mein|urdu mein|in english|english mein|again|dobara|phir se|detail mein|detail se|more detail|aur detail|detail)\b",
    r"^(this|yeh|ye|isko|is ko|is baray mein|is topic|is chapter|this topic|this chapter|what about this|is ke baray)$",
    r"^(can you explain|explain this|is ko explain|samjha do|samjha dena|please explain|please samjhao)\b",
    r"^(aur batao|more|more info|elaborate|expand|go on|continue|jari rakho|aur|or bhi)$",
]

def _is_followup(question: str) -> bool:
    """Return True if the current question looks like a follow-up without a topic."""
    q = question.strip().lower()
    # If less than 6 words and no detected unit/page, likely a follow-up
    words = q.split()
    if len(words) <= 5:
        for pattern in _FOLLOWUP_PATTERNS:
            if re.search(pattern, q):
                return True
    # Also catch anything whose main verb is just a language instruction with no noun
    if re.fullmatch(r'[\w\s]*(roman urdu|urdu|english)[\w\s]*', q) and len(words) <= 8:
        return True
    return False


def _enrich_query_from_history(question: str, history: list) -> str:
    """If query is a follow-up, extract the last meaningful topic from history."""
    if not _is_followup(question) or not history:
        return question
    
    # Walk history backwards: find the last user question that had a real topic
    for msg in reversed(history):
        if msg.get("role") == "user":
            prev_q = msg.get("content", "").strip()
            if prev_q and not _is_followup(prev_q):
                # Combine previous topic with current instruction
                enriched = f"{prev_q} — {question}"
                return enriched
        # Also try the last assistant response first sentence as context
        if msg.get("role") == "assistant":
            prev_answer = msg.get("content", "").strip()
            if prev_answer:
                # Take first 200 chars of last assistant answer as topic hint
                topic_hint = prev_answer[:200].split("\n")[0]
                enriched = f"{topic_hint} — {question}"
                return enriched
    return question


# ============================================================
# CS TYPO / ABBREVIATION NORMALIZER
# ============================================================
# Maps common spelling mistakes, Roman Urdu variations, and abbreviations
# to their correct English CS terms so FAISS vector search finds the right chunks.
_CS_TYPO_MAP = {
    # Unit 1 — Fundamentals of Computer
    "compter": "computer", "computr": "computer", "comuter": "computer",
    "computor": "computer", "cumputer": "computer",
    "hardwre": "hardware", "hardwar": "hardware", "hadware": "hardware",
    "softwre": "software", "sofware": "software", "sotware": "software",
    "processer": "processor", "prcessor": "processor",
    "generaion": "generation", "genration": "generation",
    "inpput": "input", "inut": "input", "outpput": "output", "ouput": "output",
    "memry": "memory", "memmory": "memory",
    "strage": "storage", "stroage": "storage",
    "prnter": "printer", "priner": "printer",
    "moniitor": "monitor", "monitr": "monitor",
    # Unit 2 — Operating System
    "oprating": "operating", "operting": "operating", "operatig": "operating",
    "opearting": "operating", "operatng": "operating",
    "os": "operating system",
    "windwos": "windows", "widnows": "windows", "windoes": "windows",
    "flie": "file", "fils": "file",
    "diretory": "directory", "directry": "directory",
    # Unit 3 — Office Automation
    "ms wrod": "ms word", "msword": "ms word", "word": "ms word",
    "ms excell": "ms excel", "msexcel": "ms excel", "excell": "excel",
    "ms pwrpoint": "ms powerpoint", "powerpiont": "powerpoint", "ppt": "powerpoint",
    "spredsheet": "spreadsheet", "spreedsheet": "spreadsheet",
    "wodrprocessing": "word processing", "wrod processing": "word processing",
    "formating": "formatting", "formmating": "formatting",
    # Unit 4 — Networks
    "nework": "network", "netwrk": "network", "netwok": "network",
    "toplogy": "topology", "topollogy": "topology",
    "protocl": "protocol", "protocal": "protocol",
    "bandwith": "bandwidth", "bandwdth": "bandwidth",
    "lan": "local area network", "wan": "wide area network",
    "transimission": "transmission", "transmision": "transmission",
    # Unit 5 — Security
    "secuirty": "security", "securty": "security", "scurity": "security",
    "malwre": "malware", "malwear": "malware",
    "vrius": "virus", "viurs": "virus", "virsus": "virus",
    "anivirus": "antivirus", "anti vrus": "antivirus",
    "ethic": "ethics", "etics": "ethics",
    "hackar": "hacker", "hackr": "hacker",
    # Unit 6 — Web Development
    "htlm": "html", "httml": "html", "htnl": "html",
    "csss": "css", "cs3": "css",
    "hyprlink": "hyperlink", "hyperliink": "hyperlink",
    "tabel": "table", "tabl": "table",
    "webstie": "website", "webite": "website",
    "intenet": "internet", "internt": "internet",
    # Unit 7 — Database
    "databse": "database", "dtabase": "database", "databas": "database",
    "db": "database",
    "dbms": "database management system",
    "realtional": "relational", "relatinal": "relational",
    "qurey": "query", "qurey": "query", "sqql": "sql",
    "tabel": "table",
    # Roman Urdu common variations
    "netswork": "network", "databess": "database",
    "compyuter": "computer", "softwear": "software",
}

def _normalize_query(q: str) -> str:
    """Replace known CS typos/abbreviations with correct terms for better RAG retrieval."""
    normalized = q.lower()
    for wrong, correct in _CS_TYPO_MAP.items():
        # Whole-word replacement to avoid partial matches
        normalized = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, normalized)
    return normalized

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
async def chat(request: ChatRequest):
    
    question_lower = request.question.lower()
    is_meta = is_book_structure_query(request.question)
    
    # Enrich follow-up queries with context from history
    # e.g., "explain this in roman urdu" → "operating system kya hai — explain this in roman urdu"
    rag_query = _enrich_query_from_history(request.question, request.history)
    
    # Normalize typos/abbreviations for better RAG retrieval
    # e.g., "oprating systm" → "operating system", "db" → "database"
    rag_query = _normalize_query(rag_query)
    
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
        # Use enriched query (topic from history for follow-ups) for RAG retrieval
        detected_unit = detect_unit(rag_query)
        
        if detected_unit:
            # Unit detected — use FAISS with metadata filter (increased k for richer context)
            results = CS_VECTOR_STORE.similarity_search(
                rag_query,
                k=10,
                filter={"unit": detected_unit}
            )
            if not results:
                results = hybrid_search(rag_query, CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS, k=12)
        else:
            # No unit detected — use hybrid search with enriched query
            results = hybrid_search(rag_query, CS_VECTOR_STORE, CS_BM25_RETRIEVER, CS_CHUNKS, k=12)
        
        # Build context with metadata — deduplicate by page to avoid repetition
        context_parts = []
        seen_pages = set()
        for doc in results:
            m = doc.metadata
            page_display = m.get('actual_page_number', m.get('pdf_page_number', '?'))
            page_key = f"{m.get('unit', '?')}_{page_display}"
            if page_key in seen_pages:
                # Already have content from this page — append new content only
                for i, existing in enumerate(context_parts):
                    if existing.startswith(f"[{m.get('unit', '?')} | Page {page_display}]"):
                        context_parts[i] += f"\n{doc.page_content}"
                        break
            else:
                seen_pages.add(page_key)
                context_parts.append(
                    f"[{m.get('unit', '?')} | Page {page_display}]\n{doc.page_content}"
                )
        context = "\n\n".join(context_parts)

    # Prepend Book Overview Context for general book structure queries
    if is_meta:
        context = BOOK_OVERVIEW_CONTEXT + "\n\n" + context
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": f"""You are Credehub AI — a smart, friendly, and highly effective study assistant for Karachi Board Class 9 and 10 students studying Computer Science.

Your goal is to help students understand concepts deeply and score well in their exams. Always be encouraging, patient, and supportive — like a great teacher would be.

RULES (follow strictly):

1. ANSWER ONLY from the curriculum content provided below. Do NOT use outside knowledge or general world knowledge.

2. SPELLING & TYPO TOLERANCE: Students often make spelling mistakes. You MUST try to understand their intent:
   - If a word looks like a misspelling of a CS term (e.g., "oprating system", "compter", "nework"), treat it as the correct term and answer accordingly.
   - NEVER say "I don't understand your question" just because of a spelling mistake.
   - Figure out what topic they are asking about from context and curriculum content.
   - Only say the default error message if the topic is genuinely not in the curriculum — not because of a typo.

3. If the answer is truly not found anywhere in the curriculum content provided, say exactly: "Is topic ka jawab curriculum mein nahi hai. Apne teacher se poochein."

4. If the user asked about a specific page, give COMPLETE information from that page — do not skip anything.

5. RESPONSE STRUCTURE — Always structure your answer in this order:
   a) **Definition** — Start with a clear, simple one-sentence definition of the topic.
   b) **Explanation** — Elaborate in 3-5 sentences covering key points, types, or working.
   c) **Example or Analogy** — Give a real-world example or a simple analogy the student can easily remember.
   d) **Key Points (if applicable)** — Use a bullet list to summarize the most important exam-ready facts.
   This structure makes answers easy to read, understand, and remember for exams.

6. TONE & STYLE:
   - Be encouraging: use phrases like "Bilkul!", "Bohat acha sawal hai!", "Great question!", "Yeh concept samjhna zaruri hai!"
   - Be concise but complete — do not over-repeat, do not under-explain.
   - Use bullet points or numbered lists when listing items, types, or steps.
   - Use bold text (**like this**) to highlight important terms and definitions.

7. CITATION RULE (MUST FOLLOW EXACTLY):
   - At the very end of your response, you MUST add a horizontal line `---` followed by the source in a blockquote format exactly like this:
     ---
     > 📚 **Source:** **`[Unit Name]`** | **`Page [Number]`**
   - Example of how the citation MUST look:
     ---
     > 📚 **Source:** **`Unit 1 - Fundamentals of Computer`** | **`Page 16`**
   - The [Unit Name] and [Number] MUST match the metadata header of the specific content block from which you extracted the answer.
   - Each content block in the curriculum content is prefixed with `[Unit Name | Page Number]`. Find the block that contains the answer and copy the Unit Name and Page Number from its bracketed prefix.
   - Do NOT guess page ranges. Cite only the main page where the core answer is found.

8. NEVER use your own knowledge — only what is in the content below.

9. LANGUAGE RULES:
   - If the student writes in ENGLISH → you MUST reply in English.
   - If the student writes in ROMAN URDU → you MUST reply in Roman Urdu.
   - If the student writes in URDU SCRIPT (Arabic characters) → you MUST reply in Roman Urdu.
   - DEFAULT language is English.
   - NEVER write in Hindi (Devanagari script) or any script other than Latin/English characters.
   - Match the same language/tone throughout your full response — do not mix languages mid-answer.

10. Format all keyboard shortcuts (e.g., **`Ctrl + C`**), HTML tags (e.g., **`<html>`**), and CLI commands (e.g., **`dir`**) in bold inline code so they stand out visually.

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
    
    # Send to Groq — increased max_tokens for complete, detailed answers
    response = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=1600,
        reasoning_format="hidden",
        messages=messages
    )
    
    answer = response.choices[0].message.content.strip()
    
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


# ============================================================
# ENDPOINT 4 — Generate MCQ Assessment
# POST /assessment/generate
# Body: { "num_questions": 20, "subject": "Computer Science", "grade": 9 }
# ============================================================
@app.post("/assessment/generate", response_model=AssessmentResponse)
async def assessment_generate(request: AssessmentRequest):
    """
    Generate a fresh set of MCQs from across all book chapters.
    AI creates new questions every time — never cached.
    """
    try:
        result = await generate_assessment(
            request=request,
            vector_store=CS_VECTOR_STORE,
            bm25_retriever=CS_BM25_RETRIEVER,
            chunks=CS_CHUNKS,
            client=client,
        )
        if result.total_questions == 0:
            raise HTTPException(
                status_code=500,
                detail="MCQ generation failed. Please try again."
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# ENDPOINT 5 — Generate Assessment Report
# POST /assessment/report
# Body: { "questions": [...], "student_answers": [...], "student_name": "" }
# ============================================================
@app.post("/assessment/report", response_model=AssessmentReport)
async def assessment_report(request: ReportRequest):
    """
    Score the student's answers and generate a full AI assessment report.
    Returns unit-wise breakdown, strong/weak topics, and AI narrative.
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided.")
    if not request.student_answers:
        raise HTTPException(status_code=400, detail="No answers provided.")

    try:
        report = await generate_report(request=request, client=client)
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))