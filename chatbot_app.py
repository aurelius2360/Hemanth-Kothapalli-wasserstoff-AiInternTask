import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import PyPDF2
import pytesseract
from PIL import Image
import io
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

load_dotenv()

nltk.download('punkt')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

#chromadb and groq
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

#uploads directory to steal pdfs
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    selected_documents: List[str] = []

class DocumentResponse:
    def __init__(self, doc_id: str, answer: str, citation: str):
        self.doc_id = doc_id
        self.answer = answer
        self.citation = citation

class Theme:
    def __init__(self, name: str, description: str, doc_ids: List[str]):
        self.name = name
        self.description = description
        self.doc_ids = doc_ids

# text extraction
def extract_text(file: UploadFile) -> str:
    content = file.file.read()
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += f"\nPage {page_num + 1}:\n{page_text}"
        return text
    else:  
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
        return f"Page 1:\n{text}"

#document room
def store_document(doc_id: str, text: str):
    sentences = sent_tokenize(text)
    embeddings = embedding_function(sentences)
    collection.add(
        documents=sentences,
        metadatas=[{"doc_id": doc_id, "sentence_index": i} for i in range(len(sentences))],
        ids=[f"{doc_id}_{i}" for i in range(len(sentences))],
        embeddings=embeddings
    )

#llm
async def query_llm(context: str, query: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a research assistant. Provide concise answers with citations."},
            {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
        ]
    )
    return response.choices[0].message.content


def identify_themes(responses: List[DocumentResponse]) -> List[Theme]:
    theme_prompt = "Identify common themes across the following responses:\n" + "\n".join(
        [f"Doc {r.doc_id}: {r.answer}" for r in responses]
    )
    theme_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Identify and summarize common themes with document references."},
            {"role": "user", "content": theme_prompt}
        ]
    )
    themes = []
    lines = theme_response.choices[0].message.content.split("\n")
    for line in lines:
        if line.startswith("Theme"):
            parts = line.split(":")
            theme_name = parts[0].strip()
            description = parts[1].strip()
            doc_ids = [r.doc_id for r in responses]
            themes.append(Theme(theme_name, description, doc_ids))
    return themes

# backend
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    text = extract_text(file)
    store_document(doc_id, text)
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return {"doc_id": doc_id, "filename": file.filename}

@app.get("/documents")
async def list_documents():
    files = os.listdir(UPLOAD_DIR)
    return [{"doc_id": f.split("_")[0], "filename": "_".join(f.split("_")[1:])} for f in files]

@app.post("/query")
async def process_query(request: QueryRequest):
    query_embedding = embedding_function([request.query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"doc_id": {"$in": request.selected_documents}} if request.selected_documents else None
    )
    
    responses = []
    for doc_id, doc_text, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
        answer = await query_llm(doc_text, request.query)
        citation = f"Page {metadata.get('page', 1)}, Sentence {metadata['sentence_index']}"
        responses.append(DocumentResponse(metadata["doc_id"], answer, citation))
    
    themes = identify_themes(responses)
    
    return {
        "responses": [{"doc_id": r.doc_id, "answer": r.answer, "citation": r.citation} for r in responses],
        "themes": [{"name": t.name, "description": t.description, "doc_ids": t.doc_ids} for t in themes]
    }

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)