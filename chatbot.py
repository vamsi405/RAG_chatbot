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
st.write("Type 'exit' to stop the chatbot.")


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
# Load Embedding Model
# ==================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ==================================
# Load ChromaDB
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
# Store Embeddings in ChromaDB
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
# Session State for Chat History
# ==================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================
# Display Previous Chat Messages
# ==================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================
# Chat Input
# ==================================

query = st.chat_input(
    "Ask a question from the PDF..."
)


# ==================================
# Process User Query
# ==================================

if query:

    # ----------------------------------
    # Exit Condition
    # ----------------------------------

    if query.lower() == "exit":

        st.warning("Chatbot session ended.")

        st.stop()

    # ----------------------------------
    # Store User Message
    # ----------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # ----------------------------------
    # Display User Message
    # ----------------------------------

    with st.chat_message("user"):

        st.markdown(query)

    # ----------------------------------
    # Generate Assistant Response
    # ----------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Generating Answer..."):

            # Create Query Embedding

            query_embedding = embedding_model.encode(
                query
            ).tolist()

            # Similarity Search

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            # Retrieved Chunks

            retrieved_chunks = results["documents"][0]

            # Create Context

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

            # Generate LLM Response

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer only using the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            # Extract Answer

            answer = response.choices[0].message.content

            # Display Answer

            st.markdown(answer)

    # ----------------------------------
    # Store Assistant Message
    # ----------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )