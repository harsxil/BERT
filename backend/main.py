"""
FastAPI backend for "Self-Attention: Understanding Context Inside BERT" demo.

Endpoints
---------
GET  /api/health            – liveness and model status check
POST /api/analyze-context   – run BERT on two sentences, extract contextual
                              vectors, cosine similarity, and real self-attention tensors
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.bert_service import bert_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing BERT model for Self-Attention Context Demo...")
    try:
        bert_service.load()
    except Exception as exc:
        logger.error("Failed to preload BERT model: %s", exc)
    yield
    logger.info("Shutting down backend.")


app = FastAPI(
    title="Self-Attention Inside BERT API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeContextRequest(BaseModel):
    sentence_a: str = Field(..., min_length=2, max_length=512)
    sentence_b: str = Field(..., min_length=2, max_length=512)
    target_word: str = Field(..., min_length=1)
    layer: Optional[int] = Field(0, ge=0, le=11)
    head: Optional[int] = Field(0, ge=0, le=11)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "bert_loaded": bert_service._loaded,
        "device": bert_service.device,
    }


@app.post("/api/analyze-context")
def analyze_context(req: AnalyzeContextRequest):
    if not bert_service._loaded:
        raise HTTPException(status_code=503, detail="BERT model is still loading.")
    try:
        result = bert_service.analyze_context(
            sentence_a=req.sentence_a,
            sentence_b=req.sentence_b,
            target_word=req.target_word,
            layer=req.layer if req.layer is not None else 0,
            head=req.head if req.head is not None else 0,
        )
        return result
    except Exception as exc:
        logger.exception("Context analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Context analysis error: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
