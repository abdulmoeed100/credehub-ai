# Libraries import karo
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# .env file se API key load karo
load_dotenv()

# Groq client banao
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Vector store load karo — PDF ka data
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ PDF ka data load ho gaya!")
    return vector_store

# PDF se relevant chunks dhundo
def get_relevant_context(question, vector_store):
    results = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

# RAG se chat karo
def chat_with_ai(question, vector_store):
    
    # Step 1 — PDF se relevant context dhundo
    context = get_relevant_context(question, vector_store)
    
    # Step 2 — Groq ko context + sawal bhejo
    response = client.chat.completions.create(
        # model="qwen/qwen3-32b",
        model="qwen/qwen3-32b",
        extra_body={"thinking": {"type": "disabled"}}, 
        messages=[
           {
                "role": "system",
             "content": f"""You are Credehub AI assistant for Karachi Board Class 9 and 10 students.
STRICT RULES — FOLLOW EXACTLY:
1. Answer ONLY from the curriculum content provided below. Do NOT use outside knowledge.
2. If the answer is not found in the curriculum content, respond exactly: "Is topic ka jawab curriculum mein nahi mila. Apne teacher se poochein."
3. If student writes in ENGLISH → reply in English only.
   - If student writes in ROMAN URDU → reply in Roman Urdu only.
   - If student writes in URDU SCRIPT → reply in Roman Urdu only.
   - DEFAULT language is English.
   - Never mix languages.
6. Keep answers simple, short and student friendly
7. Never make up information
8. Spellings hamesha bilkul accurate rakhna chahay regional language ho ya English.


Curriculum Content:
{context}
"""
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    return response.choices[0].message.content


# Main program
if __name__ == "__main__":
    print("Credehub AI load ho raha hai...")
    
    # Vector store ek baar load karo
    vector_store = load_vector_store()
    
    print("🤖 Credehub AI ready hai!")
    print("'exit' likho band karne ke liye\n")
    
    while True:
        question = input("Tum: ")
        
        if question.lower() == "exit":
            print("Allah Hafiz! 👋")
            break
        
        print("\nCredehub AI: ", end="")
        answer = chat_with_ai(question, vector_store)
        print(answer)
        print()