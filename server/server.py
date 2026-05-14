"""kor-minish HTTP 임베딩 서버.

uvicorn 으로 실행:
    uv run uvicorn server.server:app --host 0.0.0.0 --port 8000

환경변수:
    KOR_MINISH_MODEL  모델 경로 또는 HF repo. default: kor-minish-bge-m3-ko
    KOR_MINISH_BATCH  encode 배치 한도. default: 256
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from model2vec import StaticModel
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("KOR_MINISH_MODEL", "kor-minish-bge-m3-ko")
MAX_BATCH = int(os.environ.get("KOR_MINISH_BATCH", "256"))

resolved = Path(MODEL_PATH)
if not resolved.exists() and "/" not in MODEL_PATH:
    raise FileNotFoundError(f"model path not found: {resolved.resolve()}")

model = StaticModel.from_pretrained(MODEL_PATH)
DIM = int(model.embedding.shape[1])
VOCAB = int(model.embedding.shape[0])

app = FastAPI(title="kor-minish embedding API", version="0.1.0")


class EncodeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    normalize: bool = True


class EncodeResponse(BaseModel):
    dim: int
    embeddings: list[list[float]]


class SimilarityRequest(BaseModel):
    query: str
    docs: list[str] = Field(..., min_length=1)


class SimilarityResponse(BaseModel):
    scores: list[float]
    order: list[int]


def _encode(texts: list[str], normalize: bool) -> np.ndarray:
    if len(texts) > MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"batch too large: {len(texts)} > {MAX_BATCH}",
        )
    vecs = model.encode(texts)
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
    return vecs


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH, "dim": DIM, "vocab": VOCAB}


@app.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest) -> EncodeResponse:
    vecs = _encode(req.texts, req.normalize)
    return EncodeResponse(dim=DIM, embeddings=vecs.tolist())


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest) -> SimilarityResponse:
    all_vecs = _encode([req.query, *req.docs], normalize=True)
    q, docs = all_vecs[0], all_vecs[1:]
    scores = (docs @ q).tolist()
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return SimilarityResponse(scores=scores, order=order)
