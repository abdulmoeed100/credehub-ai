# Libraries import karo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import os

# .env file load karo
load_dotenv()

# FastAPI app banao
app = FastAPI(
    title="Credehub AI API",
    description="Karachi Board Class 9 & 10 AI Assistant",
    version="1.0.0"
)

# CORS — Co-founder ki website ko allow karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client banao
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Vector store load karo — ek baar
embeddings = FastEmbedEmbeddings()
vector_store = FAISS.load_local(
    "data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Request ka format define karo
class ChatRequest(BaseModel):
    question: str
    subject: str = "Computer Science"
    grade: int = 9

# Endpoint 1 — Health check
@app.get("/")
def home():
    return {
        "status": "Credehub AI chal raha hai! ✅",
        "version": "1.0.0"
    }

# Endpoint 2 — Chat
@app.post("/chat")
def chat(request: ChatRequest):
    
    # PDF se relevant context dhundo
    results = vector_store.similarity_search(request.question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Groq ko bhejo
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        extra_body={{"thinking": {{"type": "disabled"}}}},
        messages=[
            {
                "role": "system",
                "content": f"""You are Credehub AI Assistant for Karachi Board Class 9 and 10 students.

STRICT RULES — FOLLOW EXACTLY:
1. Answer ONLY from the curriculum content provided below. Do NOT use outside knowledge.
2. If the answer is not found in the curriculum content, respond exactly: "Is topic ka jawab curriculum mein nahi mila. Apne teacher se poochein."
3. LANGUAGE RULE — VERY IMPORTANT:
   - Format MUST be exactly like this
4. By default output language will be only english but if user asked in specific language (English, Roman Urdu, Roman Punjabi) then answer in the same language.
5. If student writes in Urdu script — still respond in English + Roman Urdu only
6. Keep answers simple, short and student friendly
7. Never make up information
8. 6. Spellings hamesha bilkul accurate rakhna chahay regional language ho ya English.


Curriculum Content:
{context}"""
            },
            {
                "role": "user",
                "content": request.question
            }
        ]
    )
    
    return {
        "question": request.question,
        "answer": response.choices[0].message.content,
        "subject": request.subject,
        "grade": request.grade
    }

# Endpoint 3 — Subjects list
@app.get("/subjects")
def get_subjects():
    return {
        "subjects": [
            "Computer Science",
            "Physics",
            "Chemistry",
            "English"
        ],
        "grades": [9, 10]
    }