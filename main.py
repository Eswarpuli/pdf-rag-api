# --- imports ---
import os
from dotenv import load_dotenv
import tempfile
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # This import is important
import uvicorn
from typing import List, Any

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ----------------------------
# Load .env and set API key
# ----------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(
    title="PDF Q&A RAG API",
    description="API for uploading PDFs and asking questions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Global Variables
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# This will hold our vector store in memory
global_vector_store: FAISS = None 

# --- NEW: Define the path on Render's persistent disk ---
FAISS_INDEX_PATH = "/var/data/my_faiss_index"

# ----------------------------
# Helper Functions (Your code, unchanged)
# ----------------------------
def get_top_docs_from_retriever(retriever, query, k=3):
    """
    Try several common method names to get relevant docs. Return list of doc objects.
    """
    for method in ("get_relevant_documents", "get_relevant_documents_by_text", "get_documents", "retrieve", "get_relevant_items"):
        fn = getattr(retriever, method, None)
        if callable(fn):
            try:
                docs = fn(query)
                return docs
            except TypeError:
                try:
                    docs = fn(query, k)
                    return docs
                except Exception:
                    continue
            except Exception:
                continue
    try:
        docs = retriever.get_relevant_documents(query)
        return docs
    except Exception:
        return []

def build_context_prompt(docs, user_question, max_chars=2500):
    chunks = []
    total = 0
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        if not text:
            continue
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                chunks.append(text[:remaining])
                total += remaining
            break
        chunks.append(text)
        total += len(text)
    context = "\n\n---\n\n".join(chunks) if chunks else ""
    system = (
        "You are an assistant that answers user questions based only on the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    prompt = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_question}\n\nAnswer concisely."
    return prompt

# ----------------------------
# API Endpoints
# ----------------------------

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    A simple endpoint to check if the server is awake and active.
    """
    print("Health check successful.")
    return {"status": "active"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global global_vector_store, embedding_model # Make sure embedding_model is global
    
    print(f"Processing PDF: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)
        
        # Create and store in the global variable
        global_vector_store = FAISS.from_documents(split_docs, embedding_model)
        
        # --- NEW: Save the index to the persistent disk ---
        print(f"Saving index to disk at {FAISS_INDEX_PATH}...")
        global_vector_store.save_local(FAISS_INDEX_PATH)
        # --------------------------------------------------
        
        print(f"✅ PDF '{file.filename}' processed and saved to disk!")
        return {"status": "success", "filename": file.filename}
    
    except Exception as e:
        print(f"PDF Processing Error: {e}")
        return {"status": "error", "message": str(e)}, 500
    
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


#  Define the QuestionRequest class ---
class QuestionRequest(BaseModel):
    query: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    global global_vector_store, llm, embedding_model # Need all three
    user_input = request.query
    bot_reply = ""

    if global_vector_store is None:
        if os.path.exists(FAISS_INDEX_PATH):
            print("Server woke up. Loading index from disk...")
            try:
                global_vector_store = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    embedding_model,
                    allow_dangerous_deserialization=True 
                )
                print("Index loaded successfully from disk.")
            except Exception as e:
                print(f"Error loading index from disk: {e}")
        else:
            print("No index found on disk. Ready for general chat.")

    try:
        # Check if PDF is loaded (either from upload or disk)
        if global_vector_store is not None:
            print("Querying with RAG...")
            retriever = global_vector_store.as_retriever(search_kwargs={"k": 3})
            docs = get_top_docs_from_retriever(retriever, user_input, k=3)
            prompt = build_context_prompt(docs, user_input, max_chars=2500)
            
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                bot_reply = getattr(response, "content", None) or (response[0].content if isinstance(response, list) and len(response)>0 else str(response))
            except Exception:
                response = llm.invoke(prompt)
                bot_reply = getattr(response, "content", None) or str(response)

        else:
            # No PDF loaded - regular chat
            print("Querying as simple chatbot...")
            response = llm.invoke([HumanMessage(content=user_input)])
            bot_reply = getattr(response, "content", None) or str(response)

        if not bot_reply:
            bot_reply = "Sorry, I couldn't generate a response."

    except Exception as e:
        print(f"Unexpected error: {e}")
        bot_reply = f"Unexpected error: {e}"

    return {"answer": bot_reply}

# This allows you to run locally with "python main.py"
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

