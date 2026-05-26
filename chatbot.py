# ==================================
# Import Libraries
# ==================================

import os
import streamlit as st
from dotenv import load_dotenv

from groq import Groq
from sentence_transformers import SentenceTransformer

import chromadb
import PyPDF2


# ==================================
# Streamlit Page Config
# ==================================

st.set_page_config(
    page_title="Cricket RAG Chatbot",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Cricket RAG Chatbot")

st.write("Ask questions from the IPL Strategy PDF")


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
# ==================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ==================================
# ChromaDB
# ==================================

@st.cache_resource
def load_chroma():

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="cricket_rag"
    )

    return collection


collection = load_chroma()


# ==================================
# Load PDF
# ==================================

pdf_path = r"E:\Apps\Chat_Bot\RAG_chatbot\IPL_Strategy_PDF_Chatbot_Demo.pdf"


@st.cache_data
def load_pdf_text():

    pdf_reader = PyPDF2.PdfReader(pdf_path)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


text = load_pdf_text()


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


# ==================================
# Create Embeddings
# ==================================

if collection.count() == 0:

    with st.spinner("Creating embeddings..."):

        for i, chunk in enumerate(chunks):

            embedding = embedding_model.encode(
                chunk
            ).tolist()

            collection.add(
                ids=[str(i)],
                documents=[chunk],
                embeddings=[embedding]
            )

    st.success("Embeddings Stored Successfully")

else:

    st.success("Existing Embeddings Found")


# ==================================
# User Question
# ==================================

query = st.text_input(
    "Ask a Question from the PDF"
)


# ==================================
# Generate Answer
# ==================================

if st.button("Get Answer"):

    if query:

        with st.spinner("Generating Answer..."):

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

            # Prompt

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

            # LLM Response

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

            # Display Answer

            st.subheader("Answer")

            st.write(answer)

    else:

        st.warning("Please enter a question.")