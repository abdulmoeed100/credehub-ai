FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the Hugging Face Embeddings model in the Docker image to make startup instant
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

COPY . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port 7860