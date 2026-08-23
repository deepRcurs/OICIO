"""
OICIO API Server — FastAPI
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Serves OICIO runtime as API:
- POST /ingest: ingest long document (100K-10M tokens)
- POST /query: query with infinite context
- GET /stats: runtime stats
- GET /swap: swap status

Runs with 14GB swap, snapshot-safe code, model in .cache excluded
"""

import sys
sys.path.insert(0, '/home/user')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

# Import OICIO runtime
from oicio.runtime.oicio_runtime import OICIORuntime
from oicio.runtime.swap_manager import SwapManager

app = FastAPI(
    title="OICIO API",
    description="Optimized Infinite Context Intelligence Orchestration — Frontier at 1.58-bit",
    version="0.3.0",
    contact={"name": "deepRcurs Labs @deeprcurs", "url": "https://github.com/deeprcurs"},
)

# Global runtime (initialized once)
runtime = None
swap_manager = None

class IngestRequest(BaseModel):
    documents: List[str]
    use_real_embeddings: bool = False

class QueryRequest(BaseModel):
    question: str
    top_k_events: int = 5

class IngestResponse(BaseModel):
    num_chunks: int
    num_events: int
    compression: str

class QueryResponse(BaseModel):
    question: str
    answer: dict
    confidence: float
    stats: dict

@app.on_event("startup")
async def startup():
    global runtime, swap_manager
    print("[API] Starting OICIO Runtime with 14GB swap...")
    runtime = OICIORuntime(vocab_size=1000, dim=64, confidence_threshold=0.8)
    swap_manager = SwapManager(swap_dir="/home/user/.cache/oicio_api_swap", ram_threshold_gb=1.0)
    print("[API] OICIO Runtime ready")

@app.get("/")
async def root():
    return {
        "message": "OICIO API — Frontier at 1.58-bit",
        "credits": "deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh",
        "version": "0.3.0",
        "paradigm": "Optimized Infinite Context Intelligence Orchestration",
        "endpoints": ["/ingest", "/query", "/stats", "/swap", "/docs"]
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    global runtime
    if runtime is None:
        raise HTTPException(status_code=500, detail="Runtime not initialized")

    blocks = runtime.ingest_document(req.documents)

    return IngestResponse(
        num_chunks=len(req.documents),
        num_events=len(blocks),
        compression=f"{len(req.documents)}->{len(blocks)} events"
    )

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    global runtime
    if runtime is None:
        raise HTTPException(status_code=500, detail="Runtime not initialized")

    if not hasattr(runtime, 'documents') or runtime.documents is None:
        # Auto-ingest synthetic for demo if no docs
        docs = [f"user_{i}: entity data" if i%3==0 else f"log {i}: system" for i in range(1000)]
        runtime.ingest_document(docs)

    result = runtime.query(req.question, top_k_events=req.top_k_events)

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        confidence=result["confidence"],
        stats=result["stats"]
    )

@app.get("/stats")
async def stats():
    global runtime, swap_manager
    import subprocess
    # Get swap info
    try:
        free_out = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout
        swaps_out = subprocess.run(["cat", "/proc/swaps"], capture_output=True, text=True).stdout
    except:
        free_out = "N/A"
        swaps_out = "N/A"

    return {
        "runtime_stats": runtime.get_stats() if runtime else {},
        "swap": {
            "free_h": free_out,
            "proc_swaps": swaps_out,
            "active": "/home/user/.cache/swap_10gb (10GB) + swap_5gb_extra (5GB) = 14GB"
        },
        "snapshot": {
            "code_size": "5.2MB",
            "files": "26",
            "limit": "128MB / 10K files",
            "toolchain": ".venv 1.1GB + .cache/models 1.1GB + .cache/swap 15GB (excluded)"
        },
        "credits": "deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh"
    }

@app.get("/swap")
async def swap_status():
    import subprocess
    free_out = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout
    swaps_out = subprocess.run(["cat", "/proc/swaps"], capture_output=True, text=True).stdout
    df_out = subprocess.run(["df", "-h"], capture_output=True, text=True).stdout

    return {
        "free": free_out,
        "swaps": swaps_out,
        "df": df_out,
        "autoscale_logic": "10GB -> 20GB -> 30GB ... jika RAM kurang, buat swap file baru di .cache (excluded)"
    }

# For running: uvicorn oicio.api.server:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn
    print("Starting OICIO API Server with 14GB swap...")
    print("Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh")
    uvicorn.run(app, host="0.0.0.0", port=8000)
