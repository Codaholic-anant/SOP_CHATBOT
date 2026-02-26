# ==============================
# main.py
# SOP Chatbot Backend
# Supports: PDF, DOCX, TXT
# Secure Admin Upload
# Append to existing FAISS DB
# 100% Local (Ollama + HF Embeddings)
# ==============================

from typing import List
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

# ==============================
# App Config
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠ Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ADMIN_TOKEN = "Token@123"
UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "vector_db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# Load Embeddings (GLOBAL)
# ==============================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# Load LLM (Ollama)
# ==============================

llm = Ollama(model="mistral")

# ==============================
# Health Check Route
# ==============================

@app.get("/")
def read_root():
    return {"message": "SOP Chatbot Running 🚀"}

# ==============================
# Upload Multiple Documents
# ==============================

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    token: str = Header(...)
):

    # 🔐 Security Check
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")

    all_docs = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file locally
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Detect file type
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)

        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path)

        else:
            continue

        docs = loader.load()
        all_docs.extend(docs)

    if not all_docs:
        return {"message": "No valid documents uploaded"}

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(all_docs)

    # Append to existing DB instead of overwrite
    if os.path.exists(VECTOR_DB_PATH):
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(split_docs)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local(VECTOR_DB_PATH)

    return {"message": "Documents uploaded & database updated successfully 🚀"}


# ==============================
# Chat Endpoint
# ==============================

class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(data: Question):

    if not os.path.exists(VECTOR_DB_PATH):
        return {"answer": "No documents uploaded yet."}

    # Always reload latest DB
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    answer = qa_chain.run(data.question)

    return {"answer": answer}