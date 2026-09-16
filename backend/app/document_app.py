"""DBS KYC document API: Jina classification and extraction only."""

from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .services.document_processing import (
    MAX_BYTES,
    MAX_PAGES,
    OCR_MODEL,
    OMNI_MODEL,
    VLM_MODEL,
    ExtractionInput,
    PageInput,
    classify,
    config,
    extract,
    ocr_configured,
    prepare_document,
)

app = FastAPI(title="DBS Document Intelligence — Jina AI", version="1.0.0")
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "docs" / "sample-docs"


@app.middleware("http")
async def boundary(request: Request, call_next):
    """Bound bodies before JSON parsing; block cross-site browser API posts."""
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).hostname not in ("localhost", "127.0.0.1"):
            return JSONResponse(
                {"detail": "Cross-site requests are not permitted."}, status_code=403
            )
        limit = MAX_BYTES if request.url.path.endswith("/prepare") else 26 * 1024 * 1024
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > limit:
                return JSONResponse(
                    {"detail": "Request exceeds the document size limit."},
                    status_code=413,
                )
        request._body = bytes(data)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/documents/health")
async def health():
    """Configuration presence only; no upstream requests."""
    return {
        "configured": bool(config("JINA_API_KEY", default="").strip()),
        "ocr_configured": ocr_configured(),
        "models": {
            "embedding": OMNI_MODEL,
            "extraction": VLM_MODEL,
            "ocr": OCR_MODEL,
        },
        "max_bytes": MAX_BYTES,
        "max_pages": MAX_PAGES,
    }


@app.post("/api/documents/prepare")
async def prepare(request: Request):
    """Decode locally, without inference."""
    return await run_in_threadpool(prepare_document, await request.body())


@app.post("/api/documents/classify")
async def classify_document(payload: PageInput):
    """Classification and reusable page/document embeddings."""
    return await classify(payload.pages)


@app.post("/api/documents/extract")
async def extract_document(payload: ExtractionInput):
    """OCR and validated schema-specific fields."""
    return await extract(payload.pages, payload.document_type, payload.engine)


def samples():
    """Enumerate only allowed sample files under the configured directory."""
    return sorted(
        p
        for p in SAMPLE_DIR.glob("*/*")
        if p.is_file()
        and not p.is_symlink()
        and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf")
        and p.resolve().is_relative_to(SAMPLE_DIR.resolve())
    )


@app.get("/api/documents/samples")
async def list_samples():
    """Local sample selection; filenames are never classifier inputs."""
    return [
        {"id": i, "name": p.name, "category": p.parent.name, "bytes": p.stat().st_size}
        for i, p in enumerate(samples())
    ]


@app.get("/api/documents/samples/{sample_id}")
async def get_sample(sample_id: int):
    """Serve a selected local sample without copying to public assets."""
    files = samples()
    if sample_id < 0 or sample_id >= len(files):
        raise HTTPException(404, "Sample not found")
    return FileResponse(files[sample_id], filename=files[sample_id].name)
