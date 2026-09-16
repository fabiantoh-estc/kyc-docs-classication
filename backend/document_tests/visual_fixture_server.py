"""Visual test server ONLY; synthetic outputs are explicitly labelled, no live calls.

Run with python -m uvicorn document_tests.visual_fixture_server:app --port 8002.
Never use this server for POC inference or accuracy validation.
"""

import io
import math

from fastapi.responses import Response
from PIL import Image, ImageDraw

from app import document_app
from app.services.document_processing import OMNI_MODEL, VLM_MODEL


async def fixture_classify(pages):
    """Return obvious synthetic vectors for layout verification."""
    v = [math.sin(i) / 23 for i in range(1024)]
    return {
        "document_type": "invoice_receipt",
        "requires_review": False,
        "mixed_pages": False,
        "scores": [
            {"document_type": "invoice_receipt", "similarity": 0.8},
            {"document_type": "passport", "similarity": 0.2},
        ],
        "margin": 0.6,
        "model": OMNI_MODEL + " (TEST FIXTURE)",
        "elapsed_seconds": 0,
        "embeddings": {
            "model": OMNI_MODEL + " (TEST FIXTURE)",
            "dimensions": 1024,
            "task": "text-matching",
            "pooling": "synthetic visual test values, NOT inference",
            "document": v,
            "pages": [v],
        },
    }


async def fixture_extract(pages, document_type, engine="vlm"):
    """Return synthetic invoice fields, labelled independently of real samples."""
    return {
        "document_type": document_type,
        "model": VLM_MODEL + " (TEST FIXTURE)",
        "elapsed_seconds": 0,
        "ocr_text": "VISUAL TEST FIXTURE — NOT LIVE INFERENCE\nExample Merchant\nPaper 2 x 5.00 = 10.00\nTax 1.00\nTotal 11.00",
        "review_notes": [
            "Synthetic visual test output. Not a prediction or accuracy result."
        ],
        "fields": {
            "vendor_name": "VISUAL TEST FIXTURE — NOT LIVE INFERENCE",
            "vendor_address": "Example street",
            "vendor_phone": None,
            "vendor_email": None,
            "invoice_number": "TEST-001",
            "transaction_date_time": "14/09/2026",
            "currency": "SGD",
            "items": [
                {
                    "sku": "00001",
                    "description": "Paper / 纸张",
                    "quantity": "2",
                    "unit_price": "5.00",
                    "line_total": "10.00",
                }
            ],
            "subtotal": "10.00",
            "taxes": [{"label": "Example tax", "rate": "10%", "amount": "1.00"}],
            "total": "11.00",
            "payments": [
                {"method": "Cash", "amount": "5.00"},
                {"method": "Credit", "amount": "6.00"},
            ],
            "card_last_four": "0010",
        },
    }


document_app.classify = fixture_classify
document_app.extract = fixture_extract
app = document_app.app
# Replace only sample routes, ensuring this server never shows user documents.
app.router.routes[:] = [
    r
    for r in app.router.routes
    if not getattr(r, "path", "").startswith("/api/documents/samples")
]


@app.get("/api/documents/samples")
async def sample_list():
    """Synthetic source only."""
    return [
        {
            "id": 0,
            "name": "VISUAL-TEST-ONLY.jpg",
            "category": "synthetic",
            "bytes": 12000,
        }
    ]


@app.get("/api/documents/samples/{sample_id}")
async def sample_image(sample_id: int):
    """Create a clearly labelled test document without customer data."""
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    for y, text in enumerate(
        [
            "VISUAL TEST FIXTURE",
            "NOT LIVE INFERENCE",
            "",
            "EXAMPLE MERCHANT",
            "Invoice: TEST-001",
            "14/09/2026",
            "",
            "Paper       2 x 5.00 = 10.00",
            "Subtotal             10.00",
            "Tax                   1.00",
            "Total                11.00",
            "",
            "Cash                  5.00",
            "Credit                6.00",
        ]
    ):
        draw.text((45, 40 + y * 42), text, fill="black", font_size=24)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return Response(output.getvalue(), media_type="image/jpeg")
