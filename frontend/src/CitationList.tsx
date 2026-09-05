import type { Citation } from "./api/types";

type CitationListProps = {
  citations: Citation[];
};

export default function CitationList({ citations }: CitationListProps) {
  const safeCitations = citations.filter(isRenderableCitation);
  if (safeCitations.length === 0) return null;

  return (
    <details className="citation-list">
      <summary>引用依据（{safeCitations.length}）</summary>
      <ol>
        {safeCitations.map((citation) => (
          <li key={citation.reference_id}>
            <div className="citation-meta">
              <strong>{safeFileName(citation.file_name)}</strong>
              <span>{formatLocation(citation)}</span>
            </div>
            {citation.heading_path.length > 0 && (
              <div className="citation-heading">{citation.heading_path.join(" / ")}</div>
            )}
            <p>{citation.excerpt}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}

function safeFileName(fileName: string): string {
  return /^[^<>:"/\\|?*\u0000-\u001f]+\.(?:pdf|docx)$/i.test(fileName)
    ? fileName
    : "未命名资料";
}

function isRenderableCitation(value: Citation): boolean {
  return (
    typeof value.reference_id === "string" &&
    /^S[1-6]$/.test(value.reference_id) &&
    typeof value.file_name === "string" &&
    safeFileName(value.file_name) !== "未命名资料" &&
    Array.isArray(value.heading_path) &&
    value.heading_path.length > 0 &&
    value.heading_path.every((item) => typeof item === "string" && item.trim() !== "") &&
    typeof value.excerpt === "string" &&
    value.excerpt.trim() !== "" &&
    value.excerpt.length <= 300
  );
}

function formatLocation(citation: Citation): string {
  const pages = citation.page
    ? citation.page_end && citation.page_end !== citation.page
      ? `第 ${citation.page}–${citation.page_end} 页`
      : `第 ${citation.page} 页`
    : "";
  const paragraphs = citation.paragraph_start
    ? citation.paragraph_end && citation.paragraph_end !== citation.paragraph_start
      ? `第 ${citation.paragraph_start}–${citation.paragraph_end} 段`
      : `第 ${citation.paragraph_start} 段`
    : "";
  const table = citation.table_id ? `表格 ${citation.table_id.replace("table-", "")}` : "";
  return [pages, paragraphs, table].filter(Boolean).join(" · ") || "位置未提供";
}
