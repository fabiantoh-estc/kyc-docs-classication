"""Isolated document tests; never load the template or contact live model APIs."""

import asyncio
import base64
import io
import json
import math
import sys

import httpx
import pymupdf
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.document_app import app
from app.services import document_processing as service


def jpeg():
    output = io.BytesIO()
    Image.new("RGB", (80, 60), "white").save(output, format="JPEG")
    return output.getvalue()


def large_jpeg():
    output = io.BytesIO()
    Image.new("RGB", (1800, 1400), "white").save(output, format="JPEG", quality=95)
    return output.getvalue()


def pdf(count=1, encrypt=False):
    with pymupdf.open() as doc:
        for _ in range(count):
            page = doc.new_page()
            page.insert_image(page.rect, stream=jpeg())
        return (
            doc.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner",
                user_pw="secret",
            )
            if encrypt
            else doc.tobytes()
        )


def null_passport():
    return {key: None for key in service.PassportFields.model_fields}


def test_preparation_and_passport_pdf():
    with TestClient(app) as client:
        for data, fmt in [(jpeg(), "JPEG"), (pdf(), "PDF")]:
            result = client.post("/api/documents/prepare", content=data)
            assert result.status_code == 200
            assert result.json()["format"] == fmt
            service.validate_pages(result.json()["pages"])
            assert result.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "data,status", [(b"", 413), (b"garbage", 422), (b"%PDF-broken", 422)]
)
def test_bad_uploads(data, status):
    assert (
        TestClient(app).post("/api/documents/prepare", content=data).status_code
        == status
    )


def test_limits_and_cross_site():
    client = TestClient(app)
    assert (
        client.post(
            "/api/documents/prepare", content=b"x" * (service.MAX_BYTES + 1)
        ).status_code
        == 413
    )
    assert client.post("/api/documents/prepare", content=pdf(6)).status_code == 422
    assert (
        client.post("/api/documents/prepare", content=pdf(encrypt=True)).status_code
        == 422
    )
    assert (
        client.post(
            "/api/documents/prepare",
            content=jpeg(),
            headers={"origin": "https://untrusted.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/documents/classify",
            json={"pages": ["https://untrusted.example/image.jpg"]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/documents/extract", json={"pages": [], "document_type": "random"}
        ).status_code
        == 422
    )


def test_vlm_pages_are_compacted_before_extraction():
    pages = service.prepare_document(large_jpeg())["pages"]
    compacted = service.compact_pages_for_vlm(pages)
    assert len(compacted) == 1
    assert len(compacted[0]) < len(pages[0])
    raw = compacted[0].removeprefix("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(raw))) as image:
        assert max(image.size) <= service.VLM_MAX_EDGE


def test_strict_extraction_and_original_scripts():
    fields = null_passport()
    fields.update(
        passport_number="001234", given_names="中文 हिन्दी", country_code="IND"
    )
    data = {"ocr_text": "Page 1\n中文 हिन्दी", "fields": fields, "review_notes": []}
    result = service.parse_extraction(
        "```json\n" + json.dumps(data) + "\n```", "passport"
    )
    assert result["fields"]["passport_number"] == "001234"
    assert result["fields"]["surname"] is None
    assert result["fields"]["given_names"] == "中文 हिन्दी"
    data["fields"]["passport_number"] = 1234
    result = service.parse_extraction(json.dumps(data), "passport")
    assert result["fields"]["passport_number"] == "1234"
    empty = service.parse_extraction('{"fields":{}}', "passport")
    assert empty["fields"]["passport_number"] is None
    with pytest.raises(HTTPException):
        service.parse_extraction('{"fields":[]}', "passport")


def test_invoice_arrays_and_fractional_quantity():
    fields = {k: None for k in service.InvoiceFields.model_fields}
    fields.update(
        items=[
            {
                "sku": "00012",
                "description": "Sample",
                "quantity": "0.61",
                "unit_price": None,
                "line_total": "8.98",
            }
        ],
        taxes=[{"label": "Sales Tax", "rate": "6.5%", "amount": "3.31"}],
        payments=[
            {"method": "Cash", "amount": "50.00"},
            {"method": "Credit", "amount": "4.23"},
        ],
        card_last_four="0010",
    )
    result = service.parse_extraction(
        json.dumps({"ocr_text": "Example", "fields": fields, "review_notes": []}),
        "invoice_receipt",
    )
    assert result["fields"]["items"][0]["quantity"] == "0.61"
    assert result["fields"]["card_last_four"] == "0010"
    assert len(result["fields"]["payments"]) == 2


def test_vlm_json_variants_are_normalized_without_fabrication():
    passport = {
        "ocr_text": "page text",
        "fields": {
            "type": "P",
            "country": "IND",
            "passportNo": "001234",
            "family_name": "EXAMPLE",
            "given_name": "ALICE MAY",
            "gender": "F",
            "dob": "23/09/1959",
            "expiry_date": "10/10/2021",
        },
        "notes": "surname visible",
    }
    wrapped = "Here is the JSON:\n```json\n" + json.dumps(passport) + "\n```"
    result = service.parse_extraction(wrapped, "passport")
    assert result["fields"]["document_type"] == "P"
    assert result["fields"]["country_code"] == "IND"
    assert result["fields"]["passport_number"] == "001234"
    assert result["fields"]["surname"] == "EXAMPLE"
    assert result["fields"]["given_names"] == "ALICE MAY"
    assert result["fields"]["date_of_birth"] == "23/09/1959"
    assert result["fields"]["date_of_issue"] is None
    assert result["review_notes"] == ["surname visible"]

    nested = {
        "ocr_text": {
            "fields": {
                "document_type": "P",
                "country_code": "SGP",
                "passport_number": "E1234567",
                "department": "ICA",
            }
        }
    }
    nested_result = service.parse_extraction(json.dumps(nested), "passport")
    assert nested_result["fields"]["passport_number"] == "E1234567"
    assert nested_result["fields"]["country_code"] == "SGP"
    assert nested_result["ocr_text"] == ""
    assert "department" not in nested_result["fields"]

    invoice = {
        "vendor_merchant_name": "Total BusinessWare Inc.",
        "vendor_contact_details": {
            "phone": "(952) 447-6624",
            "email": "support@totalbusinessware.com",
        },
        "receipt_number": "1-10628",
        "line_items": [
            {
                "product_code": "603361000025",
                "item_description": "Stapler",
                "qty": 1,
                "amount": 6.5,
            }
        ],
        "tax": {"label": "Sales Tax", "rate": "6.5%", "amount": 3.31},
        "total_amount": 58.29,
        "payment_methods_used": {"Cash": "50.00", "Credit": "4.23"},
        "card_last_4": "****0010",
    }
    result = service.parse_extraction(json.dumps(invoice), "invoice_receipt")
    assert result["fields"]["vendor_name"] == "Total BusinessWare Inc."
    assert result["fields"]["vendor_phone"] == "(952) 447-6624"
    assert result["fields"]["vendor_email"] == "support@totalbusinessware.com"
    assert result["fields"]["invoice_number"] == "1-10628"
    assert result["fields"]["items"][0]["sku"] == "603361000025"
    assert result["fields"]["items"][0]["quantity"] == "1"
    assert result["fields"]["items"][0]["line_total"] == "6.5"
    assert result["fields"]["taxes"][0]["amount"] == "3.31"
    assert result["fields"]["payments"][1]["method"] == "Credit"
    assert result["fields"]["card_last_four"] == "0010"


def test_vlm_json_repairs_quotes_and_truncation():
    quoted = (
        '{"fields":{"vendor_name":"Acme","items":[{"sku":"1","description":'
        '"Staple 8/0"PK","quantity":"1","unit_price":"1.00","line_total":"1.00"}],'
        '"taxes":[],"payments":[]},"review_notes":[],"ocr_text":""}'
    )
    result = service.parse_extraction(quoted, "invoice_receipt")
    assert result["fields"]["vendor_name"] == "Acme"
    assert result["fields"]["items"][0]["description"] == 'Staple 8/0"PK'

    truncated = (
        '{"fields":{"document_type":"P","country_code":"ZAF","passport_number":"A123",'
        '"surname":"DOE","given_names":"JANE","nationality":"RSA","sex":"F",'
        '"date_of_birth":"01 Jan 1990","place_of_birth":"CPT","place_of_issue":"CPT",'
        '"date_of_issue":"01 Jan 2015","date_of_expiry":"01 Jan 2025"},'
        '"review_notes":[],"ocr_text": "Passport page with unterminated'
    )
    result = service.parse_extraction(truncated, "passport")
    assert result["fields"]["passport_number"] == "A123"
    assert result["fields"]["surname"] == "DOE"

    utilities = (
        '{\n  "fields": {\n    "account_holder_name": "A CUSTOMER",\n'
        '    "account_number": "001",\n    "address": "1 Example Street"\n  },\n'
        '  "review_notes": [],\n  "ocr_text": "Page 1\n\n# Example heading\n1-800-000-0000"\n}'
    )
    result = service.parse_extraction(utilities, "utilities_bill")
    assert result["fields"]["account_holder_name"] == "A CUSTOMER"
    assert result["fields"]["account_number"] == "001"
    assert "Example heading" in result["ocr_text"]


def test_utilities_bill_extraction_and_kyc_name_match():
    payload = {
        "fields": {
            "customer_name": "EXAMPLE ALICE MAY",
            "account_no": "00123-45678",
            "service_address": "1 Example Street, Example City",
            "account_type": "Electricity",
            "bill_date": "10/9/24",
            "payment_due": "10/9/24",
            "billing_period": "Sep 12 - Sep 18",
        }
    }
    result = service.parse_extraction(json.dumps(payload), "utilities_bill")
    assert result["fields"]["account_holder_name"] == "EXAMPLE ALICE MAY"
    assert result["fields"]["account_number"] == "00123-45678"
    assert result["fields"]["address"].startswith("1 Example Street")
    identity = {"given_names": "ALICE MAY", "surname": "EXAMPLE"}
    match = service.compare_kyc_names(
        service.identity_full_name(identity), result["fields"]["account_holder_name"]
    )
    assert match["matched"] is True and match["status"] == "match"
    mismatch = service.compare_kyc_names(
        service.identity_full_name({"given_names": "BOB", "surname": "SPECIMEN"}),
        result["fields"]["account_holder_name"],
    )
    assert mismatch["matched"] is False and mismatch["status"] == "mismatch"
    incomplete = service.compare_kyc_names(None, "EXAMPLE ALICE MAY")
    assert incomplete["status"] == "incomplete"


def test_zero_shot_vectors_and_other_review(monkeypatch):
    calls = []

    async def fake(url, payload):
        calls.append((url, payload))
        # Page aligns to passport; then passport/invoice/utilities/other prototypes.
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return {
            "data": [
                {"index": i, "embedding": v + [0.0] * 1020}
                for i, v in reversed(list(enumerate(vectors)))
            ]
        }

    monkeypatch.setattr(service, "jina_post", fake)
    c = asyncio.run(service.classify(service.prepare_document(jpeg())["pages"]))
    assert c["document_type"] == "passport" and not c["requires_review"]
    assert len(c["embeddings"]["document"]) == 1024
    assert math.isclose(sum(v * v for v in c["embeddings"]["document"]), 1.0)
    assert calls[0][0] == service.EMBEDDING_URL
    assert calls[0][1]["model"] == service.OMNI_MODEL
    assert len(calls) == 1
    assert "utilities_bill" in [text for text in service.CLASS_DESCRIPTIONS]


def test_ambiguous_and_mixed_pages_require_review(monkeypatch):
    async def ambiguous(url, payload):
        return {
            "data": [
                {"index": i, "embedding": [1.0] + [0.0] * 1023}
                for i in range(len(payload["input"]))
            ]
        }

    monkeypatch.setattr(service, "jina_post", ambiguous)
    assert asyncio.run(service.classify(service.prepare_document(jpeg())["pages"]))[
        "requires_review"
    ]

    async def mixed(url, payload):
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return {
            "data": [
                {"index": i, "embedding": v + [0.0] * 1020}
                for i, v in enumerate(vectors)
            ]
        }

    monkeypatch.setattr(service, "jina_post", mixed)
    page = service.prepare_document(jpeg())["pages"][0]
    result = asyncio.run(service.classify([page, page]))
    assert result["requires_review"] and result["mixed_pages"]


def test_invalid_embeddings_fail_closed(monkeypatch):
    async def invalid(url, payload):
        return {"data": [{"index": 0, "embedding": [float("nan")]}]}

    monkeypatch.setattr(service, "jina_post", invalid)
    with pytest.raises(HTTPException):
        asyncio.run(service.classify(service.prepare_document(jpeg())["pages"]))


def test_extraction_uses_only_jina_vlm(monkeypatch):
    async def fake(url, payload):
        assert url == service.VLM_URL and payload["model"] == "jina-vlm"
        assert payload["max_tokens"] == service.vlm_max_tokens("passport")
        prompt = payload["messages"][0]["content"][0]["text"]
        assert prompt.index("fields") < prompt.index("ocr_text")
        assert "complete visible transcription" not in prompt
        assert "ocr_text must be an empty string" in prompt
        assert "240 characters" not in prompt
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
            assert max(image.size) <= service.VLM_MAX_EDGE
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "ocr_text": "Page 1",
                                "fields": null_passport(),
                                "review_notes": [],
                            }
                        )
                    },
                }
            ]
        }

    async def ocr_must_not_run(*args, **kwargs):
        raise AssertionError("Jina OCR must not run for the VLM engine")

    monkeypatch.setattr(service, "jina_post", fake)
    monkeypatch.setattr(service, "jina_ocr_parse", ocr_must_not_run)
    result = asyncio.run(
        service.extract(service.prepare_document(large_jpeg())["pages"], "passport")
    )
    assert result["model"] == "jina-vlm"


def test_vlm_token_budget_matches_schema():
    assert service.vlm_max_tokens("passport") == 512
    assert service.vlm_max_tokens("utilities_bill") == 512
    assert service.vlm_max_tokens("invoice_receipt") == 1024


def test_extraction_ocr_uses_harness_then_text_vlm(monkeypatch):
    pages = service.prepare_document(jpeg())["pages"]

    async def fake_ocr(received):
        assert received == pages
        return {
            "status": "SUCCESS",
            "model": "jina-ocr-v1",
            "markdown_full": "PASSPORT\nSurname EXAMPLE\nGiven names TEST",
            "failed_pages": [],
        }

    async def fake_vlm(url, payload):
        assert url == service.VLM_URL and payload["model"] == "jina-vlm"
        content = payload["messages"][0]["content"]
        assert payload["max_tokens"] == service.vlm_max_tokens("passport")
        assert isinstance(content, str)
        assert "OCR markdown" in content
        assert "ocr_text must be an empty string" in content
        assert "Surname EXAMPLE" in content
        assert "image_url" not in content
        fields = null_passport()
        fields["surname"] = "EXAMPLE"
        fields["given_names"] = "TEST"
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "ocr_text": "short",
                                "fields": fields,
                                "review_notes": [],
                            }
                        )
                    },
                }
            ]
        }

    monkeypatch.setattr(service, "jina_ocr_parse", fake_ocr)
    monkeypatch.setattr(service, "jina_post", fake_vlm)
    result = asyncio.run(service.extract(pages, "passport", engine="ocr"))
    assert result["model"] == "jina-ocr-v1"
    assert result["ocr_text"].startswith("PASSPORT")
    assert result["fields"]["surname"] == "EXAMPLE"


def test_ocr_parse_posts_multipart_to_harness(monkeypatch):
    captured = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["files"] = kwargs["files"]
            captured["data"] = kwargs["data"]
            return httpx.Response(
                200,
                json={
                    "status": "SUCCESS",
                    "markdown_full": "Hello",
                    "markdown": {"pages": [{"page": 1, "markdown": "Hello"}]},
                    "failed_pages": [],
                    "model": "jina-ocr-v1",
                },
            )

    monkeypatch.setattr(
        service, "ocr_endpoint", lambda: "https://ocr.example.test/api/parse"
    )
    monkeypatch.setattr(service.httpx, "AsyncClient", lambda **kwargs: Client())
    result = asyncio.run(service.jina_ocr_parse(service.prepare_document(jpeg())["pages"]))
    assert captured["url"] == "https://ocr.example.test/api/parse"
    assert captured["data"]["model"] == service.OCR_MODEL
    assert captured["files"]["file"][0] == "document.jpg"
    assert result["markdown_full"] == "Hello"


def test_extract_rejects_unknown_engine():
    pages = service.prepare_document(jpeg())["pages"]
    result = TestClient(app).post(
        "/api/documents/extract",
        json={"pages": pages, "document_type": "passport", "engine": "tesseract"},
    )
    assert result.status_code == 422


def test_no_template_modules_or_credentials_in_health():
    assert "app.main" not in sys.modules
    assert "app.elasticsearch.client" not in sys.modules
    result = TestClient(app).get("/api/documents/health")
    assert result.status_code == 200
    assert set(result.json()) == {
        "configured",
        "ocr_configured",
        "models",
        "max_bytes",
        "max_pages",
    }


@pytest.mark.parametrize(
    "status,expected", [(401, 502), (403, 502), (429, 429), (503, 503), (500, 502)]
)
def test_upstream_failures_do_not_leak_bodies(monkeypatch, status, expected):
    """Return actionable errors without copying sensitive upstream response bodies."""

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            return httpx.Response(status, json={"secret": "do-not-expose"})

    monkeypatch.setattr(service.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(service, "api_key", lambda: "test-only")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(service.jina_post(service.VLM_URL, {"model": service.VLM_MODEL}))
    assert caught.value.status_code == expected
    assert "do-not-expose" not in caught.value.detail


def test_missing_key_and_non_jina_endpoint(monkeypatch):
    """Missing credentials fail locally; other providers cannot be called."""
    monkeypatch.setattr(service, "config", lambda *args, **kwargs: "")
    with pytest.raises(HTTPException):
        service.api_key()
    with pytest.raises(ValueError):
        asyncio.run(service.jina_post("https://another-model.example", {}))


def test_ocr_url_comes_from_env(monkeypatch):
    monkeypatch.setattr(service, "config", lambda *args, **kwargs: "")
    with pytest.raises(HTTPException) as missing:
        service.ocr_endpoint()
    assert missing.value.status_code == 503
    monkeypatch.setattr(
        service, "config", lambda *args, **kwargs: "http://insecure.example/api/parse"
    )
    with pytest.raises(HTTPException) as insecure:
        service.ocr_endpoint()
    assert insecure.value.status_code == 503
    monkeypatch.setattr(
        service, "config", lambda *args, **kwargs: "https://ocr.example.test/api/parse/"
    )
    assert service.ocr_endpoint() == "https://ocr.example.test/api/parse"
    assert service.ocr_configured() is True
