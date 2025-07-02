from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os
import re

def load_chunks(chunk_file):
    with open(chunk_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by chunk headers like [Chunk 1]
    chunks = re.split(r"\[Chunk \d+\]\n", content)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks

def create_documents(chunks):
    return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

def embed_and_store(docs, persist_dir=os.path.join("..", "data", "vectorstore")):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(persist_dir, exist_ok=True)
    db.save_local(persist_dir)
    print(f"[✓] Vectorstore saved to: {persist_dir}")

if __name__ == "__main__":
    chunk_file = os.path.join("data", "cleaned", "chunks.txt")
    chunks = load_chunks(chunk_file)
    docs = create_documents(chunks)
    embed_and_store(docs)
