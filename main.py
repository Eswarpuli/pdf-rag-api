import os
from dotenv import load_dotenv
import tempfile
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # Used to define request body
import uvicorn
from typing import List, Any # For type hinting

# --- All your existing LangChain imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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

# --- IMPORTANT: Add CORS middleware ---
# This allows your Lovable frontend (on a different domain) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ----------------------------
# Global Variables (Replaces st.session_state)
# ----------------------------
# These are loaded once when the API starts
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# This will hold our vector store in memory
# Note: This is simple. For a real app, you'd save this to disk/DB
global_vector_store: FAISS = None 

# ----------------------------
# Helper Functions (Copied DIRECTLY from your Streamlit app)
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

#### This is your first endpoint: PDF Upload ####
# It replaces the "if uploaded_file is not None:" logic
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global global_vector_store # Declare we're modifying the global variable
    
    print(f"Processing PDF: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read()) # 'await' is needed for async
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)
        
        # Create and store in the global variable
        global_vector_store = FAISS.from_documents(split_docs, embedding_model)
        
        print(f"✅ PDF '{file.filename}' processed and indexed!")
        # Send a success message back to Lovable
        return {"status": "success", "filename": file.filename}
    
    except Exception as e:
        print(f"PDF Processing Error: {e}")
        # Send an error message back to Lovable
        return {"status": "error", "message": str(e)}, 500
    
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


#### This is your second endpoint: Chat ####
# It replaces the "if user_input := st.chat_input(...)" logic

# We define a Pydantic model for the request
class QuestionRequest(BaseModel):
    query: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    global global_vector_store, llm
    user_input = request.query
    bot_reply = ""
    
    try:
        # Check if PDF is loaded (same as your 'if st.session_state.vector_store')
        if global_vector_store is not None:
            print("Querying with RAG...")
            # --- This is your RAG logic, copied directly ---
            retriever = global_vector_store.as_retriever(search_kwargs={"k": 3})
            docs = get_top_docs_from_retriever(retriever, user_input, k=3)
            if not docs:
                for method in ("similarity_search", "search"):
                    fn = getattr(global_vector_store, method, None)
                    if callable(fn):
                        try:
                            docs = fn(user_input, 3)
                            break
                        except Exception:
                            continue
            
            prompt = build_context_prompt(docs, user_input, max_chars=2500)
            
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                bot_reply = getattr(response, "content", None) or (response[0].content if isinstance(response, list) and len(response)>0 else str(response))
            except Exception:
                response = llm.invoke(prompt)
                bot_reply = getattr(response, "content", None) or str(response)
            # --- End of your RAG logic ---

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

    # Return the answer in a JSON object
    return {"answer": bot_reply}

# This allows you to run the API locally with "python main.py"
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)