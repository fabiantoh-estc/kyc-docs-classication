import { useEffect, useRef, useState } from "react";
import {
  documentRequest,
  postDocument,
  sampleFile,
  type Classification,
  type DocumentType,
  type Extraction,
  type ExtractionEngine,
  type Fields,
  type PreparedDocument,
  type Sample,
} from "../services/documentApi";
import {
  compareKycNames,
  identityFullName,
  type KycMatch,
} from "../utils/kycMatch";
import "./DocumentProcessingPage.css";

type Phase =
  | "idle"
  | "preparing"
  | "ready"
  | "classifying"
  | "review"
  | "extracting"
  | "complete"
  | "error";
type ResultTab = "Fields" | "OCR text" | "Embeddings" | "JSON";
type KycSlot = {
  source: string;
  name: string | null;
  address?: string | null;
  document_type: DocumentType;
};
const EXTRACTABLE: DocumentType[] = [
  "passport",
  "invoice_receipt",
  "utilities_bill",
];
const names: Record<string, string> = {
  passport: "Passport",
  invoice_receipt: "Invoice / receipt",
  utilities_bill: "Utilities bill",
  other: "Other document",
};
const labels: Record<string, string> = {
  document_type: "Document type (Type / P)",
  country_code: "Issuing country code (alpha-3)",
  passport_number: "Passport number",
  surname: "Surname",
  given_names: "Given name(s)",
  nationality: "Nationality",
  sex: "Sex / gender",
  date_of_birth: "Date of birth",
  place_of_birth: "Place of birth",
  place_of_issue: "Place of issue",
  date_of_issue: "Date of issue",
  date_of_expiry: "Date of expiry",
  vendor_name: "Vendor / merchant",
  vendor_address: "Vendor address",
  vendor_phone: "Vendor phone",
  vendor_email: "Vendor email",
  invoice_number: "Invoice / receipt / ticket number",
  transaction_date_time: "Transaction date & time",
  currency: "Currency",
  subtotal: "Subtotal",
  total: "Total due / paid",
  card_last_four: "Card last 4 digits",
  sku: "SKU / code",
  description: "Description",
  quantity: "Quantity",
  unit_price: "Unit price",
  line_total: "Line total",
  label: "Tax",
  rate: "Rate",
  amount: "Amount",
  method: "Payment method",
  account_holder_name: "Account holder name",
  account_number: "Account number",
  address: "Service address",
  service_type: "Service type",
  bill_issue_date: "Bill / issue date",
  due_date: "Due date",
  billing_period: "Billing period",
};
const tableFields: Record<string, string[]> = {
  items: ["sku", "description", "quantity", "unit_price", "line_total"],
  taxes: ["label", "rate", "amount"],
  payments: ["method", "amount"],
};
const tableNames: Record<string, string> = {
  items: "Line items",
  taxes: "Taxes",
  payments: "Payment breakdown",
};

function FieldResults({ fields }: { fields: Fields }) {
  return (
    <>
      <dl className="doc-fields">
        {Object.entries(fields)
          .filter(([, v]) => !Array.isArray(v))
          .map(([key, value]) => (
            <div key={key} className={value === null ? "missing-field" : ""}>
              <dt>{labels[key] || key}</dt>
              <dd>
                {value === null ? (
                  <span className="missing">Not found / unreadable</span>
                ) : (
                  String(value)
                )}
              </dd>
            </div>
          ))}
      </dl>
      {Object.entries(tableFields)
        .filter(([key]) => key in fields)
        .map(([key, columns]) => {
          const rows = fields[key] as Record<string, string | null>[];
          return (
            <section className="doc-table-section" key={key}>
              <h3>
                {tableNames[key]} <span>{rows.length}</span>
              </h3>
              {rows.length ? (
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label={`${tableNames[key]} table`}
                >
                  <table>
                    <thead>
                      <tr>
                        {columns.map((c) => (
                          <th key={c}>{labels[c]}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => (
                        <tr key={i}>
                          {columns.map((c) => (
                            <td key={c}>{row[c] ?? "—"}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="small-note">
                  No {tableNames[key].toLowerCase()} could be extracted.
                </p>
              )}
            </section>
          );
        })}
    </>
  );
}

function fieldText(fields: Fields, key: string): string | null {
  const value = fields[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function KycMatchCard({
  identity,
  utilities,
  match,
  onReset,
}: {
  identity: KycSlot | null;
  utilities: KycSlot | null;
  match: KycMatch;
  onReset: () => void;
}) {
  const verdict =
    match.status === "match"
      ? "KYC name match"
      : match.status === "mismatch"
        ? "KYC name mismatch"
        : "Awaiting both documents";
  return (
    <section className="kyc-panel" aria-label="Customer onboarding KYC">
      <div className="kyc-panel-title">
        <div>
          <p className="eyebrow">CUSTOMER ONBOARDING KYC</p>
          <h2>Identity and proof of address</h2>
        </div>
        <span className={`kyc-verdict kyc-${match.status}`}>{verdict}</span>
      </div>
      <div className="kyc-grid">
        <article className={identity ? "filled" : ""}>
          <p className="eyebrow">01 / IDENTITY</p>
          <strong>{identity?.name || "Process a passport"}</strong>
          <small>
            {identity
              ? identity.source
              : "Surname and given names from the identity document."}
          </small>
        </article>
        <article className={utilities ? "filled" : ""}>
          <p className="eyebrow">02 / UTILITIES BILL</p>
          <strong>{utilities?.name || "Process a utilities bill"}</strong>
          <small>
            {utilities?.address ||
              "Account holder name and service address from the bill."}
          </small>
        </article>
        <article className={`kyc-result kyc-${match.status}`}>
          <p className="eyebrow">NAME CHECK</p>
          <strong>
            {match.status === "match"
              ? "Names match"
              : match.status === "mismatch"
                ? "Names do not match"
                : "Not ready"}
          </strong>
          <small>{match.reason}</small>
        </article>
      </div>
      {(identity || utilities) && (
        <button className="text-button" onClick={onReset}>
          Reset KYC pair
        </button>
      )}
    </section>
  );
}

export function DocumentProcessingPage() {
  const [dark, setDark] = useState(true);
  const [phase, setPhase] = useState<Phase>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [prepared, setPrepared] = useState<PreparedDocument | null>(null);
  const [classification, setClassification] = useState<Classification | null>(
    null,
  );
  const [extraction, setExtraction] = useState<Extraction | null>(null);
  const [schema, setSchema] = useState<DocumentType>("passport");
  const [engine, setEngine] = useState<ExtractionEngine>("vlm");
  const [reviewed, setReviewed] = useState(false);
  const [error, setError] = useState("");
  const [connection, setConnection] = useState("Connecting to document API…");
  const [configured, setConfigured] = useState(false);
  const [ocrConfigured, setOcrConfigured] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [page, setPage] = useState(0);
  const [zoom, setZoom] = useState(false);
  const [activeTab, setActiveTab] = useState<ResultTab>("Fields");
  const [dragging, setDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [identitySlot, setIdentitySlot] = useState<KycSlot | null>(null);
  const [utilitiesSlot, setUtilitiesSlot] = useState<KycSlot | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const pending = useRef<AbortController | null>(null);
  const busy = ["preparing", "classifying", "extracting"].includes(phase);
  const kycMatch = compareKycNames(
    identitySlot?.name,
    utilitiesSlot?.name,
  );

  function rememberKyc(type: DocumentType, result: Extraction, source: string) {
    if (type === "passport") {
      setIdentitySlot({
        source,
        document_type: type,
        name: identityFullName({
          given_names: fieldText(result.fields, "given_names"),
          surname: fieldText(result.fields, "surname"),
        }),
      });
    }
    if (type === "utilities_bill") {
      setUtilitiesSlot({
        source,
        document_type: type,
        name: fieldText(result.fields, "account_holder_name"),
        address: fieldText(result.fields, "address"),
      });
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    documentRequest<{ configured: boolean; ocr_configured?: boolean }>("health", {
      signal: controller.signal,
    })
      .then((h) => {
        setConfigured(h.configured);
        setOcrConfigured(Boolean(h.ocr_configured));
        if (!h.ocr_configured) setEngine("vlm");
        setConnection(
          h.configured ? "Jina API key configured" : "Jina API key required",
        );
      })
      .catch((e) => {
        if (!controller.signal.aborted)
          setConnection(e instanceof Error ? e.message : "API unavailable");
      });
    documentRequest<Sample[]>("samples", { signal: controller.signal })
      .then(setSamples)
      .catch(() => {});
    return () => {
      controller.abort();
      pending.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (!busy) return;
    setElapsed(0);
    const start = Date.now();
    const timer = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - start) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [busy, phase]);

  function newRequest() {
    pending.current?.abort();
    const c = new AbortController();
    pending.current = c;
    return c;
  }
  function fail(e: unknown, controller: AbortController) {
    if (!controller.signal.aborted) {
      setError(
        e instanceof Error ? e.message : "Processing failed. Try again.",
      );
      setPhase("error");
    }
  }
  function reset() {
    pending.current?.abort();
    setFile(null);
    setPrepared(null);
    setClassification(null);
    setExtraction(null);
    setError("");
    setPhase("idle");
    setPage(0);
    setZoom(false);
    setReviewed(false);
    setActiveTab("Fields");
    if (input.current) input.current.value = "";
  }
  async function prepareFile(selected: File, controller = newRequest()) {
    setFile(selected);
    setPrepared(null);
    setClassification(null);
    setExtraction(null);
    setError("");
    setPage(0);
    setZoom(false);
    setReviewed(false);
    setActiveTab("Fields");
    if (!selected.size || selected.size > 8 * 1024 * 1024) {
      setError("Choose a non-empty file of at most 8 MB.");
      setPhase("error");
      return;
    }
    setPhase("preparing");
    try {
      const data = await documentRequest<PreparedDocument>("prepare", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: selected,
        signal: controller.signal,
      });
      if (!controller.signal.aborted) {
        setPrepared(data);
        setPhase("ready");
      }
    } catch (e) {
      fail(e, controller);
    }
  }
  async function chooseSample(sample: Sample) {
    const c = newRequest();
    setPhase("preparing");
    setError("");
    setFile(null);
    setPrepared(null);
    setClassification(null);
    setExtraction(null);
    try {
      const selected = await sampleFile(sample, c.signal);
      if (!c.signal.aborted) await prepareFile(selected, c);
    } catch (e) {
      fail(e, c);
    }
  }
  async function runExtraction(type: DocumentType, c: AbortController) {
    if (!prepared) return;
    setPhase("extracting");
    setError("");
    setExtraction(null);
    setActiveTab("Fields");
    try {
      const result = await postDocument<Extraction>(
        "extract",
        { pages: prepared.pages, document_type: type, engine },
        c.signal,
      );
      if (!c.signal.aborted) {
        setExtraction(result);
        rememberKyc(type, result, file?.name || "Uploaded document");
        setPhase("complete");
      }
    } catch (e) {
      fail(e, c);
    }
  }
  async function processDocument() {
    if (!prepared) return;
    const c = newRequest();
    setError("");
    setClassification(null);
    setExtraction(null);
    setReviewed(false);
    setPhase("classifying");
    try {
      const result = await postDocument<Classification>(
        "classify",
        { pages: prepared.pages },
        c.signal,
      );
      if (c.signal.aborted) return;
      setClassification(result);
      if (!EXTRACTABLE.includes(result.document_type as DocumentType)) {
        setPhase("review");
        return;
      }
      const type = result.document_type as DocumentType;
      setSchema(type);
      if (result.requires_review) {
        setPhase("review");
        return;
      }
      await runExtraction(type, c);
    } catch (e) {
      fail(e, c);
    }
  }
  const exportData = {
    filename: file?.name,
    page_count: prepared?.page_count,
    classification,
    extraction,
    kyc: {
      identity: identitySlot,
      utilities: utilitiesSlot,
      match: kycMatch,
    },
    schema_selection: {
      document_type: extraction?.document_type ?? schema,
      engine,
      source: reviewed ? "reviewer" : "automatic",
    },
    status: phase,
    note: "Similarity scores are not calibrated probabilities. Review extracted fields against the original.",
  };
  function download() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(exportData, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${file?.name.replace(/\.[^.]+$/, "") || "document"}-results.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const stageStates = [
    classification
      ? "Complete"
      : phase === "classifying"
        ? "Running"
        : phase === "error" && prepared
          ? "Failed"
          : "Waiting",
    extraction
      ? "Complete"
      : phase === "extracting"
        ? "Running"
        : phase === "review"
          ? "Review"
          : phase === "error" && classification
            ? "Failed"
            : "Waiting",
    classification ? "Available" : "Waiting",
  ];

  return (
    <div className="doc-app" data-theme={dark ? "dark" : "light"}>
      <header className="doc-header">
        <a href="./" className="doc-wordmark">
          DBS <span>Customer onboarding KYC</span>
        </a>
        <div>
          <span className="doc-pill">PROOF OF CONCEPT</span>
          <button onClick={() => setDark(!dark)}>
            {dark ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </header>
      <main>
        <section className="doc-hero">
          <div>
            <p className="eyebrow">CUSTOMER ONBOARDING / KYC</p>
            <h1>
              Verify identity
              <br />
              and proof of address.
            </h1>
            <p>
              Match a passport to a utilities bill during onboarding.
              <br />
              Invoices still classify and extract as usual.
            </p>
          </div>
          <img
            loading="lazy"
            src={`${import.meta.env.BASE_URL}images/document-hero.svg`}
            alt="Illustrated passport and utilities bill used for customer onboarding KYC"
          />
        </section>
        <div className="doc-stage-grid" aria-label="Processing stages">
          {[
            "Classify document",
            "Extract information",
            "Document embeddings",
          ].map((name, i) => (
            <div
              className={`doc-stage ${stageStates[i].toLowerCase()}`}
              key={name}
            >
              <span className="stage-number">
                {["Complete", "Available"].includes(stageStates[i])
                  ? "✓"
                  : `0${i + 1}`}
              </span>
              <div>
                <strong>{name}</strong>
                <small>
                  {i === 1
                    ? engine === "ocr"
                      ? "Jina OCR · transcription then schema fields"
                      : "Jina VLM · schema-guided extraction"
                    : "Jina Omni · multimodal embeddings"}
                </small>
              </div>
              <span className="stage-state">{stageStates[i]}</span>
            </div>
          ))}
        </div>
        <div className="connection-line">
          <span className={configured ? "configured" : ""}>●</span> {connection}
          <span className="language-note">
            English · 中文 / 繁體 · Melayu · Indonesia · हिन्दी
          </span>
        </div>
        <KycMatchCard
          identity={identitySlot}
          utilities={utilitiesSlot}
          match={kycMatch}
          onReset={() => {
            setIdentitySlot(null);
            setUtilitiesSlot(null);
          }}
        />
        <div className="doc-workspace">
          <section
            className="doc-panel"
            onDragOver={(e) => {
              e.preventDefault();
              if (!busy) setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (!busy && e.dataTransfer.files[0])
                void prepareFile(e.dataTransfer.files[0]);
            }}
          >
            <div className="panel-title">
              <h2>Source document</h2>
              {file ? (
                <button className="text-button" onClick={reset}>
                  Clear
                </button>
              ) : (
                <span>01 / UPLOAD</span>
              )}
            </div>
            <input
              ref={input}
              type="file"
              aria-label="Upload document"
              accept=".jpg,.jpeg,.png,.pdf"
              className="visually-hidden"
              disabled={busy}
              onChange={(e) => {
                if (e.target.files?.[0]) void prepareFile(e.target.files[0]);
                e.target.value = "";
              }}
            />
            {prepared ? (
              <>
                <div className="doc-file-info">
                  <strong>{file?.name}</strong>
                  <span>
                    {prepared.format} · {prepared.page_count}{" "}
                    {prepared.page_count === 1 ? "page" : "pages"}
                  </span>
                </div>
                <div className={`doc-preview ${zoom ? "zoomed" : ""}`}>
                  <img
                    src={prepared.pages[page]}
                    alt={`Uploaded document, page ${page + 1}`}
                  />
                </div>
                <div className="doc-page-controls">
                  <button
                    disabled={page === 0}
                    onClick={() => setPage(page - 1)}
                    aria-label="Previous page"
                  >
                    ←
                  </button>
                  <span>
                    Page {page + 1} of {prepared.page_count}
                  </span>
                  <button
                    disabled={page === prepared.page_count - 1}
                    onClick={() => setPage(page + 1)}
                    aria-label="Next page"
                  >
                    →
                  </button>
                  <button onClick={() => setZoom(!zoom)}>
                    {zoom ? "Fit page" : "Zoom in"}
                  </button>
                </div>
                <fieldset className="engine-toggle" disabled={busy}>
                  <legend>Extraction engine</legend>
                  <div className="engine-toggle-options">
                    <label>
                      <input
                        type="radio"
                        name="extraction-engine"
                        value="vlm"
                        checked={engine === "vlm"}
                        onChange={() => setEngine("vlm")}
                      />
                      Jina VLM
                    </label>
                    <label>
                      <input
                        type="radio"
                        name="extraction-engine"
                        value="ocr"
                        checked={engine === "ocr"}
                        disabled={!ocrConfigured}
                        onChange={() => setEngine("ocr")}
                      />
                      Jina OCR
                    </label>
                  </div>
                  {!ocrConfigured && (
                    <p className="small-note">
                      Set JINA_OCR_URL in backend/.env to enable Jina OCR.
                    </p>
                  )}
                </fieldset>
                <div className="doc-action-row">
                  <button
                    className="primary"
                    disabled={busy || !configured}
                    onClick={() => void processDocument()}
                  >
                    {busy
                      ? "Processing…"
                      : classification
                        ? "Process again"
                        : "Process document"}
                  </button>
                  <button
                    disabled={busy}
                    onClick={() => input.current?.click()}
                  >
                    Replace file
                  </button>
                </div>
                <p className="small-note inset">
                  {engine === "ocr"
                    ? "Processing sends rendered pages to Jina OCR for transcription, then VLM for schema fields. Results stay in this session unless you download them."
                    : "Processing sends rendered pages to Jina VLM for schema-guided extraction. Results stay in this session unless you download them."}
                </p>
              </>
            ) : (
              <div
                className={`doc-empty drop-zone ${dragging ? "dragging" : ""}`}
              >
                <div className="document-symbol">↥</div>
                <h3>
                  {phase === "preparing"
                    ? "Preparing your document…"
                    : "Drop your document here"}
                </h3>
                <p>
                  JPEG, PNG or scanned PDF
                  <br />
                  Up to 8 MB · 5 pages · 25 megapixels
                </p>
                <button
                  className="primary"
                  disabled={busy}
                  onClick={() => input.current?.click()}
                >
                  Choose document
                </button>
              </div>
            )}
            {!!samples.length && (
              <details className="doc-samples">
                <summary>
                  Use a local sample{" "}
                  <span>
                    {samples.length}{" "}
                    {samples.length === 1 ? "document" : "documents"}
                  </span>
                </summary>
                <div>
                  {samples.map((s) => (
                    <button
                      disabled={busy}
                      onClick={() => void chooseSample(s)}
                      key={s.id}
                    >
                      <span>{s.name}</span>
                      <small>
                        {s.category} · {(s.bytes / 1024).toFixed(0)} KB
                      </small>
                    </button>
                  ))}
                </div>
              </details>
            )}
          </section>
          <section className="doc-panel">
            <div className="panel-title">
              <h2>Document results</h2>
              {classification ? (
                <button className="text-button" onClick={download}>
                  Download JSON ↓
                </button>
              ) : (
                <span>READY WHEN YOU ARE</span>
              )}
            </div>
            {error && (
              <div className="doc-alert" role="alert">
                <strong>Processing needs attention</strong>
                <p>{error}</p>
                {classification && !extraction && (
                  <button
                    disabled={busy}
                    onClick={() => {
                      setReviewed(true);
                      void runExtraction(schema, newRequest());
                    }}
                  >
                    Retry extraction
                  </button>
                )}
              </div>
            )}
            {busy && (
              <div className="doc-progress" role="status" aria-live="polite">
                <span className="spinner" />
                {phase === "preparing"
                  ? "Preparing preview locally"
                  : phase === "classifying"
                    ? "Jina Omni is classifying the document"
                    : engine === "ocr"
                      ? "Jina OCR is transcribing the document"
                      : "Jina VLM is reading the document"}
                <span>{elapsed}s</span>
              </div>
            )}
            {classification && (
              <div className="classification-card">
                <div className="classification-heading">
                  <div>
                    <p className="eyebrow">DETECTED DOCUMENT</p>
                    <h3>{names[classification.document_type]}</h3>
                  </div>
                  <span
                    className={`doc-pill ${classification.requires_review ? "review-pill" : ""}`}
                  >
                    {classification.requires_review
                      ? "REVIEW REQUIRED"
                      : "SCHEMA SELECTED"}
                  </span>
                </div>
                <div className="score-list">
                  {classification.scores.map((s) => (
                    <div className="score" key={s.document_type}>
                      <span>{names[s.document_type]}</span>
                      <meter
                        min={-1}
                        max={1}
                        value={s.similarity}
                        aria-label={`${names[s.document_type]} similarity`}
                      />
                      <code>{s.similarity.toFixed(3)}</code>
                    </div>
                  ))}
                </div>
                <p className="small-note">
                  Cosine similarity, not confidence. Pilot thresholds are not
                  calibrated. {classification.elapsed_seconds}s
                </p>
                {(phase === "review" ||
                  (!busy && classification && extraction)) && (
                  <div className="schema-review">
                    <label htmlFor="schema">
                      {classification.mixed_pages
                        ? "Pages disagree — split mixed documents or select a schema"
                        : "Review extraction schema"}
                    </label>
                    <div>
                      <select
                        id="schema"
                        value={schema}
                        onChange={(e) =>
                          setSchema(e.target.value as DocumentType)
                        }
                      >
                        <option value="passport">Passport</option>
                        <option value="utilities_bill">Utilities bill</option>
                        <option value="invoice_receipt">
                          Invoice / receipt
                        </option>
                      </select>
                      <button
                        onClick={() => {
                          setReviewed(true);
                          void runExtraction(schema, newRequest());
                        }}
                      >
                        {extraction ? "Re-extract" : "Confirm & extract"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
            {classification && (
              <>
                <div
                  className="doc-tabs"
                  role="tablist"
                  aria-label="Result views"
                >
                  {(
                    ["Fields", "OCR text", "Embeddings", "JSON"] as ResultTab[]
                  ).map((t) => (
                    <button
                      role="tab"
                      aria-selected={activeTab === t}
                      aria-controls="result-panel"
                      id={`tab-${t.replace(" ", "-")}`}
                      tabIndex={activeTab === t ? 0 : -1}
                      key={t}
                      onClick={() => setActiveTab(t)}
                      onKeyDown={(e) => {
                        const tabs: ResultTab[] = [
                          "Fields",
                          "OCR text",
                          "Embeddings",
                          "JSON",
                        ];
                        const direction =
                          e.key === "ArrowRight"
                            ? 1
                            : e.key === "ArrowLeft"
                              ? -1
                              : 0;
                        if (direction) {
                          e.preventDefault();
                          const next =
                            tabs[
                              (tabs.indexOf(t) + direction + tabs.length) %
                                tabs.length
                            ];
                          setActiveTab(next);
                          document
                            .getElementById(`tab-${next.replace(" ", "-")}`)
                            ?.focus();
                        }
                      }}
                    >
                      {t}
                    </button>
                  ))}
                </div>
                <div
                  className="doc-result-content"
                  id="result-panel"
                  role="tabpanel"
                  aria-labelledby={`tab-${activeTab.replace(" ", "-")}`}
                >
                  {activeTab === "Fields" &&
                    (extraction ? (
                      <>
                        <div className="result-meta">
                          <strong>
                            {names[extraction.document_type]} fields
                          </strong>
                          <span>
                            {extraction.elapsed_seconds}s ·{" "}
                            {reviewed
                              ? "Reviewer-selected schema"
                              : "Automatic schema"}
                          </span>
                        </div>
                        <FieldResults fields={extraction.fields} />
                        {!!extraction.review_notes.length && (
                          <aside className="review-notes">
                            <h3>Review notes</h3>
                            <ul>
                              {extraction.review_notes.map((n, i) => (
                                <li key={i}>{n}</li>
                              ))}
                            </ul>
                          </aside>
                        )}
                        <p className="small-note">
                          Extracted by {extraction.model}. Verify against the
                          original before use.
                        </p>
                      </>
                    ) : (
                      <div className="tab-empty">
                        <h3>
                          {phase === "review"
                            ? "Confirm the document type"
                            : "Fields will appear here"}
                        </h3>
                        <p>
                          {phase === "review"
                            ? "Choose the correct schema above to begin extraction."
                            : engine === "ocr"
                              ? "Review the classification while Jina OCR transcribes the document."
                              : "Review the classification while Jina VLM reads the document."}
                        </p>
                      </div>
                    ))}
                  {activeTab === "OCR text" && (
                    <>
                      <p className="small-note">
                        Original-language transcription from{" "}
                        {extraction?.model === "jina-ocr-v1"
                          ? "Jina OCR"
                          : "Jina VLM"}
                        .
                      </p>
                      <pre className="ocr-text">
                        {extraction?.ocr_text ||
                          "Transcription is available after successful extraction."}
                      </pre>
                    </>
                  )}
                  {activeTab === "Embeddings" && (
                    <>
                      <div className="vector-metrics">
                        <div>
                          <strong>
                            {classification.embeddings.dimensions}
                          </strong>
                          <span>Dimensions</span>
                        </div>
                        <div>
                          <strong>
                            {classification.embeddings.pages.length}
                          </strong>
                          <span>Page vectors</span>
                        </div>
                        <div>
                          <strong>Text-matching</strong>
                          <span>Embedding task</span>
                        </div>
                      </div>
                      <h3 className="vector-title">
                        Document vector · first 64 dimensions
                      </h3>
                      <div
                        className="vector-bars"
                        role="img"
                        aria-label="Preview of first 64 document embedding dimensions"
                      >
                        {classification.embeddings.document
                          .slice(0, 64)
                          .map((v, i) => (
                            <span
                              key={i}
                              title={`${i}: ${v}`}
                              style={{
                                height: `${Math.max(3, Math.min(100, Math.abs(v) * 1000))}%`,
                                opacity: v < 0 ? 0.45 : 1,
                              }}
                            />
                          ))}
                      </div>
                      <p className="small-note">
                        {classification.embeddings.model}
                        <br />
                        {classification.embeddings.pooling}. Reused from
                        classification; no second embedding call.
                      </p>
                      <details>
                        <summary>Inspect vector values</summary>
                        <pre>
                          {JSON.stringify(
                            classification.embeddings.document,
                            null,
                            2,
                          )}
                        </pre>
                      </details>
                    </>
                  )}
                  {activeTab === "JSON" && (
                    <pre>{JSON.stringify(exportData, null, 2)}</pre>
                  )}
                </div>
              </>
            )}
            {!classification && !busy && (
              <div className="doc-empty">
                <div className="document-symbol">▤</div>
                <h3>Start customer onboarding KYC.</h3>
                <p>
                  Process a passport, then a utilities bill, to check the name.
                  <br />
                  Invoices still extract as a separate operations document.
                </p>
                <div className="supported-types">
                  <span>Passport</span>
                  <span>Utilities bill</span>
                  <span>Invoice / receipt</span>
                </div>
              </div>
            )}
          </section>
        </div>
        <footer>
          Jina AI models only <span>•</span> No Elasticsearch <span>•</span> DBS
          customer onboarding KYC
        </footer>
      </main>
    </div>
  );
}
