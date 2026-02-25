from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import os

app = FastAPI()

ADMIN_TOKEN = "supersecret123"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vector_store = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load LLM
llm = Ollama(model="mistral")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever()
)

@app.get("/")
def home():
    return {"message": "SOP Chatbot Running 🚀"}

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(data: Question):
    answer = qa_chain.run(data.question)
    return {"answer": answer}


# 🔐 Secure Upload Endpoint
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    x_admin_token: str = Header(None)
):
    global vector_store, qa_chain

    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")

    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(file_location)
    documents = loader.load()

    # Create new vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("vector_db")

    # 🔥 IMPORTANT: Reload QA chain with new retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    os.remove(file_location)

    return {"message": "PDF uploaded successfully ✅"}