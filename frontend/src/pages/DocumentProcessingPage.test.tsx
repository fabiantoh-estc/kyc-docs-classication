import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { DocumentProcessingPage } from "./DocumentProcessingPage";
import { documentRequest, postDocument } from "../services/documentApi";
vi.mock("../services/documentApi", () => ({
  documentRequest: vi.fn(),
  postDocument: vi.fn(),
  sampleFile: vi.fn(),
}));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
const vector = {
  model: "jina-embeddings-v5-omni-small",
  task: "text-matching",
  dimensions: 1024,
  pooling: "mean",
  pages: [[1]],
  document: [1],
};
function setup(review = false) {
  vi.mocked(documentRequest).mockImplementation(async (path) => {
    if (path === "health")
      return { configured: true, ocr_configured: true } as never;
    if (path === "samples") return [] as never;
    return {
      pages: ["data:image/jpeg;base64,example"],
      page_count: 1,
      format: "JPEG",
    } as never;
  });
  vi.mocked(postDocument).mockImplementation(async (path) => {
    if (path === "classify")
      return {
        document_type: "passport",
        requires_review: review,
        scores: [{ document_type: "passport", similarity: 0.8 }],
        model: vector.model,
        elapsed_seconds: 1,
        embeddings: vector,
      } as never;
    return {
      fields: { passport_number: "00123", surname: null },
      ocr_text: "Example transcription",
      review_notes: [],
      model: "jina-vlm",
      document_type: "passport",
      elapsed_seconds: 2,
    } as never;
  });
  render(<DocumentProcessingPage />);
}
async function upload() {
  fireEvent.change(screen.getByLabelText("Upload document"), {
    target: {
      files: [new File(["example"], "input.jpg", { type: "image/jpeg" })],
    },
  });
  await waitFor(() =>
    expect(
      (
        screen.getByRole("button", {
          name: "Process document",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false),
  );
}
describe("Document workflow", () => {
  it("classifies then extracts and clears results", async () => {
    setup();
    await upload();
    fireEvent.click(screen.getByRole("button", { name: "Process document" }));
    await screen.findByText("00123");
    expect(
      screen.getByRole("heading", { name: "Identity and proof of address" }),
    ).toBeTruthy();
    expect(screen.queryByText("KYC name match")).toBeNull();
    expect(vi.mocked(postDocument).mock.calls.map((c) => c[0])).toEqual([
      "classify",
      "extract",
    ]);
    expect(vi.mocked(postDocument).mock.calls[1][1]).toMatchObject({
      document_type: "passport",
      engine: "vlm",
    });
    expect(screen.queryByText("Not found / unreadable")).not.toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "OCR text" }));
    expect(screen.queryByText("Example transcription")).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^Clear$/ }));
    expect(screen.queryByText("Example transcription")).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Download JSON ↓" }),
    ).toBeNull();
  });
  it("waits for reviewer confirmation on uncertain classification", async () => {
    setup(true);
    await upload();
    fireEvent.click(screen.getByRole("button", { name: "Process document" }));
    await screen.findByText("REVIEW REQUIRED");
    expect(postDocument).toHaveBeenCalledTimes(1);
    fireEvent.change(screen.getByLabelText("Review extraction schema"), {
      target: { value: "invoice_receipt" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Confirm & extract" }));
    await waitFor(() => expect(postDocument).toHaveBeenCalledTimes(2));
    expect(vi.mocked(postDocument).mock.calls[1][1]).toMatchObject({
      document_type: "invoice_receipt",
      engine: "vlm",
    });
  });
  it("sends Jina OCR when that extraction engine is selected", async () => {
    setup();
    await upload();
    fireEvent.click(screen.getByRole("radio", { name: "Jina OCR" }));
    fireEvent.click(screen.getByRole("button", { name: "Process document" }));
    await screen.findByText("00123");
    expect(vi.mocked(postDocument).mock.calls[1][0]).toBe("extract");
    expect(vi.mocked(postDocument).mock.calls[1][1]).toMatchObject({
      engine: "ocr",
    });
  });
  it("retains embeddings after extraction timeout", async () => {
    setup();
    const original = vi.mocked(postDocument).getMockImplementation()!;
    vi.mocked(postDocument).mockImplementation(async (...args) => {
      if (args[0] === "extract") throw new Error("Jina inference timed out.");
      return original(...args);
    });
    await upload();
    fireEvent.click(screen.getByRole("button", { name: "Process document" }));
    await screen.findByRole("alert");
    expect(screen.queryByText("Jina inference timed out.")).not.toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "Embeddings" }));
    expect(screen.queryByText("1024")).not.toBeNull();
    expect(
      (
        screen.getByRole("button", {
          name: "Retry extraction",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
  });
});
