#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv

from groq import Groq
import chromadb
import PyPDF2

from sentence_transformers import SentenceTransformer


# In[3]:


# ==================================
# Load API Key
# ==================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# ==================================
# Groq Client
# ==================================

client = Groq(
    api_key=groq_api_key
)

# ==================================
# Embedding Model
# Free and Accurate
# ==================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==================================
# Persistent ChromaDB
# ==================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="cricket_rag"
)

# ==================================
# Load PDF
# ==================================

pdf_path = r"E:\Apps\Chat_Bot\RAG_chatbot\IPL_Strategy_PDF_Chatbot_Demo.pdf"

pdf_reader = PyPDF2.PdfReader(pdf_path)

text = ""

for page in pdf_reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text

print("PDF Loaded Successfully")

# ==================================
# Chunking
# ==================================

chunk_size = 500
chunk_overlap = 100

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start += chunk_size - chunk_overlap

print(f"Total Chunks Created: {len(chunks)}")

# ==================================
# Create Embeddings Once
# ==================================

if collection.count() == 0:

    print("Creating embeddings...")

    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk
        ).tolist()

        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )

    print("Embeddings Stored Successfully")

else:

    print("Existing Embeddings Found")

# ==================================
# Chat Loop
# ==================================

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    # Query Embedding

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # Similarity Search

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are a Cricket Knowledge Assistant.

Strict Rules:

1. Answer only using the provided context.
2. Do not use external knowledge.
3. If answer is not found, say:
   "I could not find that information in the document."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Answer using only the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    print("\nAnswer:")
    print(answer)


# In[ ]:




