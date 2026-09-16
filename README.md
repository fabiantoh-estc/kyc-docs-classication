# DBS customer onboarding KYC — document intelligence

Proof of concept for **DBS customer onboarding KYC**: classify a passport, utilities bill or invoice, extract structured fields, and check whether the passport name matches the utilities account holder.

Inference uses **Jina AI only**. There is no Elasticsearch, Kibana, or Agent Builder in this package.

## Architecture

The browser keeps the uploaded file in memory. The FastAPI service rasterizes pages, calls Jina, validates JSON, and returns results. Uploads are **not written to disk** by the API. The Jina API key never leaves the server.

```
Browser (React, :3000)
    POST /api/documents/prepare | classify | extract
FastAPI (backend/.env, :8001)
    JINA_API_KEY  -->  api.jina.ai  (Omni embeddings)
                  -->  api-beta-vlm.jina.ai  (jina-vlm)
    JINA_OCR_URL  -->  optional OCR /api/parse  (no Jina key sent)
```

| Stage | Model | Input | Output |
| --- | --- | --- | --- |
| Prepare | none (local) | JPEG, PNG or PDF | JPEG data URLs, max 5 pages, 8 MB, 2048px edge |
| Classify | `jina-embeddings-v5-omni-small` | page images + class texts | document type + 1024-d vectors |
| Extract (VLM) | `jina-vlm` | compacted page images | schema fields |
| Extract (OCR) | harness then `jina-vlm` | same pages, then OCR markdown | transcription + schema fields |
| KYC name check | none (deterministic) | passport name vs bill holder | match / mismatch / incomplete |

## Classification

`POST /api/documents/classify` embeds each prepared page and four **text** class descriptions with Omni (`task=text-matching`, 1024 dimensions, normalized). Classes are:

- passport
- invoice_receipt
- utilities_bill
- other

The predicted label is the class with the highest cosine similarity to the **mean page vector**. Filenames and folder names are never sent to the model.

Review is required when:

- the top class is `other`
- top similarity &lt; `0.15`
- margin to the next class &lt; `0.025`
- pages disagree with each other (mixed document)

Those thresholds are **pilot settings**, not calibrated confidence.

## Field extraction

The UI radio chooses the extract engine (`engine` on `POST /api/documents/extract`).

**Jina VLM (default).** Compacted page images (896px JPEG) go to `jina-vlm` with a schema prompt. The model must return JSON `fields`, `review_notes`, and empty `ocr_text`. Pydantic rejects extra keys and non-string scalars. Missing values stay `null`; nothing is invented.

**Jina OCR.** Requires `JINA_OCR_URL` in `backend/.env` (https, typically `…/api/parse`, multipart field `file`). The harness returns markdown; VLM maps that markdown onto the same schemas. The OCR tab shows the harness transcript. If `JINA_OCR_URL` is unset, the OCR radio is disabled.

Schemas:

- **Passport:** type, alpha-3 country code, number, surname, given names, nationality, sex, dates and places of birth/issue/expiry.
- **Utilities bill:** account holder, account number, service address, service type, issue date, due date, billing period.
- **Invoice / receipt:** merchant, contact, number, datetime, currency, line items, taxes, totals, payments, card last four.

## KYC name match

After a passport extract, the UI stores given names + surname. After a utilities extract (without clearing the pair), it stores `account_holder_name` and address. Matching is **token overlap** after NFKC/uppercase and dropping honorifics. It is not a calibrated identity score. Invoices do not update the KYC pair.

## Run locally

Python 3.12 and Node 20+ are required.

```sh
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit backend/.env: set JINA_API_KEY from https://jina.ai
# Optional: set JINA_OCR_URL to your OCR parse endpoint

cd ../frontend
npm install

cd ..
chmod +x scripts/dev.sh
./scripts/dev.sh start
```

- UI: http://127.0.0.1:3000
- API: http://127.0.0.1:8001

```sh
./scripts/dev.sh status
./scripts/dev.sh stop
```

Override ports with `DOCUMENT_UI_PORT` and `DOCUMENT_API_PORT`.

## Samples

Public-domain demo files are under `docs/sample-docs/` (`passport/`, `utilities/`, `invoices/`). They appear in **Use a local sample**. You can add further JPEG/PNG/PDF files in those folders; filenames are not used as classifier labels.

## Security

- Put secrets only in `backend/.env`. That file is gitignored. Never use `VITE_` for the Jina key.
- Do not commit `backend/.env`.
- The API binds to loopback and rejects non-localhost browser `Origin` on POST.
- Upstream Jina error bodies are not forwarded; the UI shows generic status messages.
- Provider retention of images sent to Jina is governed by your Jina account terms.

## Tests

```sh
cd backend
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest

cd ../frontend
npm test
```
