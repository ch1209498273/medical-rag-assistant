import { useCallback, useEffect, useState } from "react";

import { ApiError, listDocuments, reindexDocument, scanDocuments } from "./api/client";
import type { DocumentRow } from "./api/types";

const FAILURE_LABELS: Record<string, string> = {
  PROVIDER_UNAVAILABLE: "服务暂不可用",
  SOURCE_FILE_MISSING: "资料不存在",
  SOURCE_FILE_UNREADABLE: "资料不可读取",
  DOCUMENT_NOT_INDEXABLE: "资料不可索引",
  INDEX_FAILED: "索引失败",
};

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setDocuments(await listDocuments());
    } catch {
      setError("资料列表暂时无法加载，请稍后重试。");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const scan = async () => {
    setBusyAction("scan");
    setError(null);
    setNotice(null);
    try {
      await scanDocuments();
      await refresh();
      setNotice("扫描完成，资料列表已刷新");
    } catch {
      setError("资料扫描失败，请检查配置后重试。");
    } finally {
      setBusyAction(null);
    }
  };

  const reindex = async (document: DocumentRow) => {
    setBusyAction(`reindex-${document.document_id}`);
    setError(null);
    setNotice(null);
    try {
      await reindexDocument(document.document_id);
      await refresh();
      setNotice("资料已更新");
    } catch (errorValue: unknown) {
      const message = errorValue instanceof ApiError && errorValue.status === 404
        ? "资料不存在或已被移除。"
        : "资料重建失败，请稍后重试。";
      setError(message);
    } finally {
      setBusyAction(null);
    }
  };

  return (
    <section className="page documents-page" aria-labelledby="documents-page-title">
      <header className="page-header">
        <div>
          <p className="eyebrow">资料管理</p>
          <h2 id="documents-page-title">制度资料</h2>
          <p className="page-description">查看当前索引状态；页面只展示安全的资料元数据。</p>
        </div>
        <button type="button" className="primary-button" onClick={() => void scan()} disabled={busyAction !== null}>
          {busyAction === "scan" ? "扫描中…" : "扫描资料"}
        </button>
      </header>

      {error && (
        <div className="page-error" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <div className="status-line status-success" role="status" aria-live="polite">
          {notice}
        </div>
      )}

      {loading && (
        <div className="sr-only" role="status" aria-live="polite">
          正在加载资料列表…
        </div>
      )}

      <div className="table-responsive">
        <table className="documents-table">
          <caption className="sr-only">制度资料索引状态</caption>
          <thead>
            <tr>
              <th scope="col">文件名</th>
              <th scope="col">版本</th>
              <th scope="col">状态</th>
              <th scope="col">更新时间</th>
              <th scope="col">失败原因</th>
              <th scope="col">操作</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={6}>正在加载资料列表…</td>
              </tr>
            )}
            {!loading && documents.length === 0 && (
              <tr>
                <td colSpan={6}>暂时没有已登记的资料。</td>
              </tr>
            )}
            {!loading && documents.map((document) => (
              <tr key={document.document_id}>
                <td>{safeFileName(document.file_name)}</td>
                <td>{document.version_id || "—"}</td>
                <td><span className={`document-status status-${statusClass(document.status)}`}>{statusLabel(document.status)}</span></td>
                <td>{formatUpdatedAt(document.updated_at)}</td>
                <td>{failureLabel(document.failure_reason)}</td>
                <td>
                  <button
                    type="button"
                    className="table-action"
                    onClick={() => void reindex(document)}
                    disabled={busyAction !== null}
                  >
                    {busyAction === `reindex-${document.document_id}` ? "重建中…" : `重建 ${safeFileName(document.file_name)}`}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function safeFileName(fileName: string): string {
  const parts = fileName.split(/[\\/]/);
  return parts.at(-1) || "未命名资料";
}

function failureLabel(reason: string | null | undefined): string {
  if (!reason) return "—";
  return FAILURE_LABELS[reason] ?? "索引失败";
}

function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? "状态未知";
}

const STATUS_LABELS: Record<string, string> = {
  indexing: "建库中",
  pending: "处理中",
  active: "已完成",
  failed: "失败",
  superseded: "已被新版本替代",
  inactive: "未启用",
  needs_ocr: "待 OCR",
};

function statusClass(status: string): string {
  return Object.prototype.hasOwnProperty.call(STATUS_LABELS, status) ? status : "unknown";
}

function formatUpdatedAt(updatedAt?: string | null): string {
  if (!updatedAt) return "—";
  const fractionIndex = updatedAt.indexOf(".");
  const withoutFraction = fractionIndex >= 0 ? updatedAt.slice(0, fractionIndex) : updatedAt;
  return (withoutFraction.endsWith("Z") ? withoutFraction.slice(0, -1) : withoutFraction).replace("T", " ");
}
