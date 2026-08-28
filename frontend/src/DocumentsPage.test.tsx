import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import DocumentsPage from "./DocumentsPage";
import { listDocuments, reindexDocument, scanDocuments } from "./api/client";
import type { DocumentRow } from "./api/types";

vi.mock("./api/client", () => ({
  listDocuments: vi.fn(),
  reindexDocument: vi.fn(),
  scanDocuments: vi.fn(),
}));

const mockedListDocuments = vi.mocked(listDocuments);
const mockedReindexDocument = vi.mocked(reindexDocument);
const mockedScanDocuments = vi.mocked(scanDocuments);

describe("DocumentsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedListDocuments.mockResolvedValue([
      {
        document_id: 1,
        file_name: "护理制度.docx",
        version_id: "v1",
        status: "active",
        updated_at: "2026-08-23T01:02:03Z",
        failure_reason: null,
      },
      {
        document_id: 2,
        file_name: "虚构失败样例.pdf",
        version_id: "v2",
        status: "failed",
        updated_at: "2026-08-22T01:02:03Z",
        failure_reason: "INDEX_FAILED",
      },
    ]);
    mockedScanDocuments.mockResolvedValue({ results: [], deactivated: [], snapshot_complete: true });
    mockedReindexDocument.mockResolvedValue({
      document_id: 1,
      file_name: "护理制度.docx",
      version_id: "v1",
      status: "active",
      updated_at: "2026-08-23T01:02:03Z",
      failure_reason: null,
    });
  });

  it("renders safe metadata and maps failure reasons without exposing paths or raw content", async () => {
    render(<DocumentsPage />);

    expect(await screen.findByText("护理制度.docx")).toBeInTheDocument();
    expect(screen.getByText("2026-08-23 01:02:03")).toBeInTheDocument();
    expect(screen.getByText("索引失败")).toBeInTheDocument();
    expect(screen.queryByText("fictional source path")).not.toBeInTheDocument();
    expect(screen.queryByText("完整原始正文")).not.toBeInTheDocument();
  });

  it("refreshes after scanning and reindexes a selected document", async () => {
    const user = userEvent.setup();
    render(<DocumentsPage />);
    await screen.findByText("护理制度.docx");

    await user.click(screen.getByRole("button", { name: "扫描资料" }));
    await waitForListCall(2);
    expect(mockedScanDocuments).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "重建 护理制度.docx" }));
    expect(await screen.findByText("资料已更新")).toBeInTheDocument();
    expect(mockedReindexDocument).toHaveBeenCalledWith(1);
  });

  it("labels every backend document status explicitly", async () => {
    mockedListDocuments.mockResolvedValue([
      { document_id: 10, file_name: "indexing.docx", status: "indexing", version_id: "v1" },
      { document_id: 11, file_name: "active.docx", status: "active", version_id: "v1" },
      { document_id: 12, file_name: "failed.docx", status: "failed", version_id: "v1" },
      { document_id: 13, file_name: "superseded.docx", status: "superseded", version_id: "v1" },
      { document_id: 14, file_name: "inactive.docx", status: "inactive", version_id: "v1" },
      { document_id: 15, file_name: "needs-ocr.docx", status: "needs_ocr", version_id: "v1" },
    ]);

    render(<DocumentsPage />);

    expect(await screen.findByText("建库中")).toBeInTheDocument();
    expect(screen.getByText("已完成")).toBeInTheDocument();
    expect(screen.getByText("失败")).toBeInTheDocument();
    expect(screen.getByText("已被新版本替代")).toBeInTheDocument();
    expect(screen.getByText("未启用")).toBeInTheDocument();
    expect(screen.getByText("待 OCR")).toBeInTheDocument();
  });

  it("announces document loading and saved scan state", async () => {
    let resolveList!: (value: DocumentRow[]) => void;
    mockedListDocuments.mockReturnValueOnce(new Promise((resolve) => {
      resolveList = resolve;
    }));
    const user = userEvent.setup();
    render(<DocumentsPage />);

    expect(screen.getByRole("status")).toHaveTextContent("正在加载资料列表…");
    resolveList([]);
    await screen.findByText("暂时没有已登记的资料。");

    await user.click(screen.getByRole("button", { name: "扫描资料" }));
    expect(await screen.findByRole("status")).toHaveTextContent("扫描完成，资料列表已刷新");
  });
});

async function waitForListCall(times: number): Promise<void> {
  for (;;) {
    if (mockedListDocuments.mock.calls.length >= times) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}
