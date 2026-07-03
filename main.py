"""
main.py

FastAPI backend for the AI HR Recruitment Assistant.

Endpoints:
  GET  /        - health check
  POST /analyze - upload resumes + job description, run the 4-agent pipeline
  POST /query   - search previously indexed resumes for a free-text query
"""

from __future__ import annotations

import asyncio
import logging
import os
from io import BytesIO
from typing import List

from docx import Document as DocxDocument
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from models import FinalReport
from tasks import run_pipeline
from tools import get_query_engine, index_resumes, reset_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANALYZE_TIMEOUT_SECONDS = 300
MAX_CANDIDATES = 5
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

app = FastAPI(
    title="AI HR Recruitment Assistant",
    description="Multi-agent resume screening and candidate evaluation API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(raw_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(raw_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text).strip()


def _extract_docx_text(raw_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(raw_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs).strip()


def _extract_txt_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore").strip()


def extract_text(file_name: str, raw_bytes: bytes) -> str:
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        return _extract_pdf_text(raw_bytes)
    if ext == ".docx":
        return _extract_docx_text(raw_bytes)
    if ext == ".txt":
        return _extract_txt_text(raw_bytes)
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "service": "AI HR Recruitment Assistant",
        "model": "groq/llama-3.3-70b-versatile",
    }


@app.post("/analyze", response_model=FinalReport)
async def analyze(
    job_description: str = Form(..., description="The job description text"),
    resumes: List[UploadFile] = File(..., description="1-5 resume files (PDF/DOCX/TXT)"),
):
    """
    Run the full multi-agent pipeline: screen, rank, generate interview
    questions for the top 3, and produce final hiring recommendations.
    """
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description must not be empty.")

    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume file is required.")

    if len(resumes) > MAX_CANDIDATES:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_CANDIDATES} resumes are supported per request.",
        )

    resume_texts = {}
    skipped_files: List[str] = []

    for upload in resumes:
        ext = os.path.splitext(upload.filename or "")[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped_files.append(upload.filename or "unknown")
            continue
        try:
            raw_bytes = await upload.read()
            text = extract_text(upload.filename, raw_bytes)
            if not text.strip():
                skipped_files.append(upload.filename)
                continue
            resume_texts[upload.filename] = text
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse file %s", upload.filename)
            skipped_files.append(upload.filename or "unknown")

    if not resume_texts:
        raise HTTPException(
            status_code=400,
            detail="No valid resume text could be extracted from the uploaded files.",
        )

    try:
        reset_index()
        index_resumes(resume_texts)

        report = await asyncio.wait_for(
            asyncio.to_thread(run_pipeline, job_description, resume_texts, skipped_files),
            timeout=ANALYZE_TIMEOUT_SECONDS,
        )
        return report
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timed out after {ANALYZE_TIMEOUT_SECONDS} seconds.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline execution failed.")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/query")
async def query_resumes(request: QueryRequest):
    """Search the currently indexed resumes for relevant content."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")

    try:
        engine = get_query_engine(similarity_top_k=request.top_k)
        nodes = engine.retrieve(request.query)
        results = [
            {
                "file_name": n.node.metadata.get("file_name", "unknown"),
                "score": getattr(n, "score", None),
                "text": n.node.get_content(),
            }
            for n in nodes
        ]
        return {"query": request.query, "results": results}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed.")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)