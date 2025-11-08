import os
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
from deta import Deta

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load ENV
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Deta Setup
deta = Deta()
index_drive = deta.Drive("pdf-index")
FAISS_INDEX_NAME = "faiss_index"

# FastAPI App Init
app = FastAPI(title="PDF RAG API", description="Q&A over PDFs.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QuestionRequest(BaseModel):
    query: str


# LLM + Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# ✅ Helper Functions
def get_top_docs_from_retriever(retriever, query, k=3):
    try:
        return retriever.get_relevant_documents(query)
    except Exception:
        return []


def build_context_prompt(docs, question, max_chars=2500):
    text = ""
    for d in docs:
        if hasattr(d, "page_content"):
            if len(text) + len(d.page_content) > max_chars:
                break
            text += d.page_content + "\n\n"
    return f"""
You are a helpful assistant. Answer ONLY using the context below.

CONTEXT:
{text}

QUESTION: {question}
"""


# ✅ Upload PDF → Build Index → Save to Deta
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(split_docs, embedding_model)

    # Save index locally → zip → upload
    with tempfile.TemporaryDirectory() as temp_dir:
        local_index = os.path.join(temp_dir, FAISS_INDEX_NAME)
        vector_store.save_local(local_index)

        shutil.make_archive(local_index, "zip", local_index)
        with open(local_index + ".zip", "rb") as f:
            index_drive.put(f"{FAISS_INDEX_NAME}.zip", f)

    os.remove(tmp_path)

    return {"status": "success", "message": "PDF indexed successfully"}


# ✅ Ask Question → Load index → Query RAG
@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    index_file = index_drive.get(f"{FAISS_INDEX_NAME}.zip")

    if index_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "index.zip")
            with open(zip_path, "wb") as f:
                f.write(index_file.read())

            local_index = os.path.join(temp_dir, FAISS_INDEX_NAME)
            shutil.unpack_archive(zip_path, local_index)

            vector_store = FAISS.load_local(
                local_index,
                embedding_model,
                allow_dangerous_deserialization=True
            )

            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = get_top_docs_from_retriever(retriever, request.query)
            prompt = build_context_prompt(docs, request.query)
            response = llm.invoke([HumanMessage(content=prompt)])
            return {"answer": response.content}

    # If no pdf uploaded yet → normal chatbot mode
    response = llm.invoke([HumanMessage(content=request.query)])
    return {"answer": response.content}


# ✅ Health Check
@app.get("/health")
def health():
    return {"status": "alive"}
