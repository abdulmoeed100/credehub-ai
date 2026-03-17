# Libraries import karo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import os
import re

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

# ─────────────────────────────────────────────────────
# Subjects ke alag alag indexes load karo
# Nayi subject add karni ho toh yahan add karo
# ─────────────────────────────────────────────────────
embeddings = FastEmbedEmbeddings()

INDEXES = {
    "Computer Science": FAISS.load_local(
        "data/faiss_index/computer_science_9",
        embeddings,
        allow_dangerous_deserialization=True
    ),
    # Future subjects — index ready hone pe uncomment karo:
    # "Physics": FAISS.load_local(
    #     "data/faiss_index/physics_9",
    #     embeddings,
    #     allow_dangerous_deserialization=True
    # ),
}

# Request ka format
class ChatRequest(BaseModel):
    question: str
    history:  List[Dict] = []
    subject:  str = "Computer Science"
    grade:    int = 9

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

    # Sahi subject ka index lo
    vector_store = INDEXES.get(request.subject, INDEXES["Computer Science"])

    # PDF se relevant chunks dhundo
    results = vector_store.similarity_search(request.question, k=3)

    # Context banao — metadata ke saath (page number + subject)
    context_parts = []
    for doc in results:
        page_num = doc.metadata.get("page_number", "?")
        subject  = doc.metadata.get("subject", "?")
        context_parts.append(
            f"[{subject} — Page {page_num}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Messages banao
    messages = [
        {
            "role": "system",
            "content": f"""/no_think
You are Credehub AI Assistant for Karachi Board Class 9 and 10 students in Pakistan.

STRICT RULES — FOLLOW EXACTLY:

1. Answer ONLY from the curriculum content provided below. Do NOT use outside knowledge.

2. If the answer is not found in the curriculum content, say:
   - In English: "This topic is not in the curriculum. Please ask your teacher."
   - In Roman Urdu: "Is topic ka jawab curriculum mein nahi mila. Apne teacher se poochein."
   - In Roman Punjabi: "Eh topic curriculum wich nahi hai. Apne teacher toun puchho."

3. LANGUAGE DETECTION — VERY IMPORTANT:
   - If student writes in ENGLISH or says "in english" → reply in English only.
   - If student writes in ROMAN URDU or says "urdu mein batao" → reply in Roman Urdu only.
   - If student writes in URDU SCRIPT → reply in Roman Urdu only.
   - If student writes in ROMAN PUNJABI or says "punjabi mein batao" → reply in Roman Punjabi only.
   - DEFAULT language is English.
   - NEVER say "I cannot respond in this language."
   - NEVER mix two languages in one reply.

4. ROMAN URDU GUIDE:
   - Simple words: Yeh, Hai, Tha, Karta, Nahi, Matlab, Jaise, Kyunki
   - Example: "Computer ek electronic machine hai jo data process karta hai."

5. ROMAN PUNJABI GUIDE:
   - Simple words: Eh, Hai, Si, Karda, Nahi, Matlab, Jive, Tusi, Wich, Ton
   - Example: "Computer ik electronic machine hai jo data process karda hai."

6. Give a detailed explanation in 8-10 lines. Use examples where possible.

7. Never make up information. Only use what is in the curriculum content.

8. Spellings must always be accurate.

Curriculum Content:
{context}"""
        }
    ]

    # History add karo — last 10 messages
    for msg in request.history[-10:]:
        messages.append(msg)

    # Naya sawaal add karo
    messages.append({
        "role": "user",
        "content": request.question
    })

    # Groq ko bhejo
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        max_tokens=500,
        messages=messages
    )

    # Think tags remove karo
    answer = response.choices[0].message.content
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()

    return {
        "question": request.question,
        "answer":   answer,
        "subject":  request.subject,
        "grade":    request.grade
    }

# Endpoint 3 — Subjects list
@app.get("/subjects")
def get_subjects():
    return {
        "subjects": ["Computer Science", "Physics", "Chemistry", "English"],
        "grades":   [9, 10]
    }
