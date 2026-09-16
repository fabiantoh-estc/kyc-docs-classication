export type DocumentType = "passport" | "invoice_receipt" | "utilities_bill";
export type ExtractionEngine = "vlm" | "ocr";
export type PreparedDocument = {
  pages: string[];
  format: string;
  page_count: number;
};
export type Classification = {
  document_type: DocumentType | "other";
  requires_review: boolean;
  mixed_pages: boolean;
  scores: { document_type: string; similarity: number }[];
  margin: number;
  model: string;
  elapsed_seconds: number;
  embeddings: {
    model: string;
    task: string;
    dimensions: number;
    pooling: string;
    document: number[];
    pages: number[][];
  };
};
export type Fields = Record<
  string,
  string | null | Record<string, string | null>[]
>;
export type Extraction = {
  fields: Fields;
  ocr_text: string;
  review_notes: string[];
  model: string;
  document_type: DocumentType;
  elapsed_seconds: number;
};
export type Sample = {
  id: number;
  name: string;
  category: string;
  bytes: number;
};
const ROOT = `${import.meta.env.BASE_URL}api/documents/`;

export async function documentRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${ROOT}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `Document API returned HTTP ${response.status}. Check that the Jina-only API is running.`,
    );
  }
  return response.json() as Promise<T>;
}
export const postDocument = <T>(
  path: string,
  data: unknown,
  signal: AbortSignal,
) =>
  documentRequest<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
    signal,
  });
export async function sampleFile(sample: Sample, signal: AbortSignal) {
  const response = await fetch(`${ROOT}samples/${sample.id}`, { signal });
  if (!response.ok)
    throw new Error(
      "Sample file is unavailable. Choose a local document instead.",
    );
  return new File([await response.blob()], sample.name);
}
