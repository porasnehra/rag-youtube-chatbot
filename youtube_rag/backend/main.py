from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag_logic import ingest_youtube_video, ask_question

app = FastAPI(title="YouTube RAG API")

class IngestRequest(BaseModel):
    url: str

class AskRequest(BaseModel):
    query: str
    video_id: str

@app.post("/ingest")
def ingest(request: IngestRequest):
    result = ingest_youtube_video(request.url)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/ask")
def ask(request: AskRequest):
    result = ask_question(request.query, request.video_id)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result
