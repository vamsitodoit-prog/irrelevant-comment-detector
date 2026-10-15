"""FastAPI server for the Irrelevant Comment Detection web app."""
import csv
import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .detector import IrrelevantCommentDetector

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
MAX_COMMENTS = 2000

app = FastAPI(title="Irrelevant Comment Detector", version="1.0.0",
              description="Flags off-topic, spam and low-effort comments for a given post.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

_detectors: dict[tuple, IrrelevantCommentDetector] = {}


def get_detector(threshold: float, spam_threshold: float) -> IrrelevantCommentDetector:
    key = (round(threshold, 3), round(spam_threshold, 3))
    if key not in _detectors:
        _detectors[key] = IrrelevantCommentDetector(threshold=threshold, spam_threshold=spam_threshold)
    return _detectors[key]


class AnalyzeRequest(BaseModel):
    post: str = Field(..., min_length=3, description="Body/description of the post, video or article")
    title: str = Field("", description="Optional title (weighted higher)")
    comments: list[str] = Field(..., min_length=1)
    threshold: float = Field(0.10, ge=0.0, le=1.0, description="Min relevance to count as on-topic")
    spam_threshold: float = Field(0.5, ge=0.0, le=1.0)


@app.get("/api/health")
def health():
    metrics_file = ROOT / "model" / "metrics.json"
    return {"status": "ok", "model_loaded": get_detector(0.10, 0.5).model is not None,
            "model_metrics": json.loads(metrics_file.read_text()) if metrics_file.exists() else None}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    comments = [c.strip() for c in req.comments if c and c.strip()]
    if not comments:
        raise HTTPException(400, "No non-empty comments supplied")
    if len(comments) > MAX_COMMENTS:
        raise HTTPException(413, f"Too many comments (max {MAX_COMMENTS})")
    return get_detector(req.threshold, req.spam_threshold).analyze(req.post, comments, req.title)


@app.post("/api/analyze-csv")
async def analyze_csv(post: str = Form(...), file: UploadFile = File(...), title: str = Form(""),
                      threshold: float = Form(0.10), column: str = Form("comment")):
    """Upload a CSV; comments are read from `column` (falls back to the first column)."""
    text = (await file.read()).decode("utf-8-sig", errors="ignore")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise HTTPException(400, "Empty CSV")
    header = [h.strip().lower() for h in rows[0]]
    idx = header.index(column.lower()) if column.lower() in header else 0
    body = rows[1:] if column.lower() in header or header[0] in ("comment", "comments", "text") else rows
    comments = [r[idx].strip() for r in body if len(r) > idx and r[idx].strip()][:MAX_COMMENTS]
    if not comments:
        raise HTTPException(400, "No comments found in CSV")
    return get_detector(threshold, 0.5).analyze(post, comments, title)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")
