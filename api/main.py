"""Clausula REST API.

Run: uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader

from clausula.detector import analyze, load_detector

MAX_CHARS = 200_000
detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request.
    global detector
    detector = load_detector()
    yield


app = FastAPI(title="Clausula API", description="Detect potentially unfair clauses in Terms of Service.",
              version="0.1.0", lifespan=lifespan)


def get_detector():
    return detector


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, description="Full Terms of Service text")


class Label(BaseModel):
    category: str
    score: float
    explanation: str


class Clause(BaseModel):
    text: str
    labels: list[Label]


class AnalyzeResponse(BaseModel):
    model: str
    n_sentences: int
    n_flagged: int
    risk_level: str
    categories: dict[str, int]
    clauses: list[Clause]


@app.get("/health")
def health(det=Depends(get_detector)):
    return {"status": "ok", "model": det.name if det else None}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest, det=Depends(get_detector)):
    return analyze(det, req.text)


@app.post("/analyze/pdf", response_model=AnalyzeResponse)
async def analyze_pdf(file: UploadFile = File(...), det=Depends(get_detector)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")
    try:
        reader = PdfReader(BytesIO(await file.read()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the PDF") from None
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text found in the PDF (is it a scanned image?)")
    return analyze(det, text[:MAX_CHARS])
