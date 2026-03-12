# Libraries import karo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
        messages=[
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
   - Use whichever matches the student's language.

3. LANGUAGE DETECTION — VERY IMPORTANT:
   - If student writes in ENGLISH or says "in english / tell me in english" → reply in English only.
   - If student writes in ROMAN URDU or says "urdu mein / roman urdu mein batao" → reply in Roman Urdu only.
   - If student writes in URDU SCRIPT → reply in Roman Urdu only. Never use Urdu script in reply.
   - If student writes in ROMAN PUNJABI or says "punjabi mein / in punjabi / roman punjabi mein batao" → reply in Roman Punjabi only.
   - DEFAULT language is English if you cannot detect the language.
   - NEVER say "I cannot respond in this language." Always try your best.
   - NEVER mix two languages in one reply.

4. ROMAN URDU GUIDE — follow this style:
   - Simple words: Yeh, Hai, Tha, Karta, Nahi, Matlab, Jaise, Kyunki
   - Example answer: "Computer ek electronic machine hai jo data process karta hai aur results deta hai."

5. ROMAN PUNJABI GUIDE — follow this style exactly:
   - Simple words: Eh, Hai, Si, Karda, Nahi, Matlab, Jive, Kyunki, Tusi, Karo, Wich, Ton
   - Example answer: "Computer ik electronic machine hai jo data process karda hai te results dinda hai."
   - Do NOT use complex or literary Punjabi — keep it simple like everyday speech.

6. Keep answers simple and student friendly. Give a detailed explanation in 8-10 lines. Use examples where possible.
7. Never make up information. Only use what is in the curriculum content.

8. Spellings must always be accurate — whether English, Roman Urdu, or Roman Punjabi.

Curriculum Content:
{context}"""
            },
            {
                "role": "user",
                "content": request.question
            }
        ]
    )

    # Think tags remove karo — regex se
    answer = response.choices[0].message.content
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()

    return {
        "question": request.question,
        "answer": answer,
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
