import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  listDocuments,
  reindexDocument,
  saveDocumentBusinessMetadata,
  scanDocuments,
} from "./api/client";
import type { DocumentBusinessMetadataInput, DocumentRow } from "./api/types";

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
  const [selectedDocument, setSelectedDocument] = useState<DocumentRow | null>(null);
  const [metadataForm, setMetadataForm] = useState<DocumentBusinessMetadataInput>(emptyMetadata());

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

  const selectMetadataDocument = (document: DocumentRow) => {
    setSelectedDocument(document);
    setMetadataForm(document.business_metadata ? {
      content_type: document.business_metadata.content_type,
      applicable_scope: document.business_metadata.applicable_scope,
      effective_from: document.business_metadata.effective_from,
      review_due_at: document.business_metadata.review_due_at,
      business_status: document.business_metadata.business_status,
      owner_role: document.business_metadata.owner_role,
      supersedes_version_id: document.business_metadata.supersedes_version_id,
    } : emptyMetadata());
  };

  const saveMetadata = async () => {
    if (!selectedDocument?.version_id) return;
    setBusyAction(`metadata-${selectedDocument.document_id}`);
    setError(null);
    setNotice(null);
    try {
      const saved = await saveDocumentBusinessMetadata(selectedDocument.document_id, metadataForm);
      setDocuments((current) => current.map((document) => (
        document.document_id === selectedDocument.document_id
          ? { ...document, business_metadata: saved }
          : document
      )));
      setSelectedDocument((current) => current ? { ...current, business_metadata: saved } : current);
      setNotice("资料业务属性已保存");
    } catch {
      setError("资料属性保存失败，请检查受控选项和日期后重试。");
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
              <th scope="col">业务状态</th>
              <th scope="col">适用范围</th>
              <th scope="col">更新时间</th>
              <th scope="col">失败原因</th>
              <th scope="col">操作</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={8}>正在加载资料列表…</td>
              </tr>
            )}
            {!loading && documents.length === 0 && (
              <tr>
                <td colSpan={8}>暂时没有已登记的资料。</td>
              </tr>
            )}
            {!loading && documents.map((document) => (
              <tr key={document.document_id}>
                <td>{safeFileName(document.file_name)}</td>
                <td>{document.version_id || "—"}</td>
                <td><span className={`document-status status-${statusClass(document.status)}`}>{statusLabel(document.status)}</span></td>
                <td>{businessStatusLabel(document.business_metadata?.business_status)}</td>
                <td>{scopeLabel(document.business_metadata?.applicable_scope)}</td>
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
                  <button
                    type="button"
                    className="table-action"
                    onClick={() => selectMetadataDocument(document)}
                    disabled={busyAction !== null || !document.version_id}
                  >
                    {`设置 ${safeFileName(document.file_name)} 的资料属性`}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="document-metadata-panel" aria-label="资料业务属性">
        <h3>资料业务属性</h3>
        <p>属性只决定资料能否作为正式依据；不会展示或下载资料正文，也不构成登录或 ACL。</p>
        <div className="metadata-grid">
          <label>
            资料类型
            <select value={metadataForm.content_type} onChange={(event) => setMetadataForm({ ...metadataForm, content_type: event.target.value as DocumentBusinessMetadataInput["content_type"] })}>
              <option value="policy">制度</option><option value="training">培训</option><option value="procedure">操作规范</option><option value="other">其他</option>
            </select>
          </label>
          <label>
            业务状态
            <select aria-label="业务状态" value={metadataForm.business_status} onChange={(event) => setMetadataForm({ ...metadataForm, business_status: event.target.value as DocumentBusinessMetadataInput["business_status"] })}>
              <option value="unknown">待确认</option><option value="draft">草稿</option><option value="approved">已确认</option><option value="superseded">已被替代</option><option value="retired">已停用</option>
            </select>
          </label>
          <label>
            适用范围
            <select aria-label="适用范围" value={metadataForm.applicable_scope} onChange={(event) => setMetadataForm({ ...metadataForm, applicable_scope: event.target.value as DocumentBusinessMetadataInput["applicable_scope"] })}>
              <option value="unspecified">未指定</option><option value="all_staff">全体医护</option><option value="nurse">护士</option><option value="doctor">医生</option><option value="pharmacist">药师</option><option value="administrator">管理员</option>
            </select>
          </label>
          <label>
            生效日期
            <input aria-label="生效日期" value={metadataForm.effective_from ?? ""} placeholder="YYYY-MM-DD" onChange={(event) => setMetadataForm({ ...metadataForm, effective_from: event.target.value || null })} />
          </label>
          <label>
            复核日期
            <input aria-label="复核日期" value={metadataForm.review_due_at ?? ""} placeholder="YYYY-MM-DD" onChange={(event) => setMetadataForm({ ...metadataForm, review_due_at: event.target.value || null })} />
          </label>
          <label>
            责任角色（可选）
            <input value={metadataForm.owner_role ?? ""} onChange={(event) => setMetadataForm({ ...metadataForm, owner_role: event.target.value || null })} />
          </label>
        </div>
        <button type="button" className="primary-button" onClick={() => void saveMetadata()} disabled={!selectedDocument?.version_id || busyAction !== null}>
          {busyAction?.startsWith("metadata-") ? "保存中…" : "保存资料属性"}
        </button>
      </section>
    </section>
  );
}

function emptyMetadata(): DocumentBusinessMetadataInput {
  return { content_type: "policy", applicable_scope: "unspecified", effective_from: null, review_due_at: null, business_status: "unknown", owner_role: null, supersedes_version_id: null };
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

function businessStatusLabel(status?: string): string {
  return ({ approved: "已确认", draft: "草稿", superseded: "已被替代", retired: "已停用", unknown: "待业务确认" } as Record<string, string>)[status ?? "unknown"] ?? "待业务确认";
}

function scopeLabel(scope?: string): string {
  return ({ all_staff: "全体医护", nurse: "护士", doctor: "医生", pharmacist: "药师", administrator: "管理员", unspecified: "未指定" } as Record<string, string>)[scope ?? "unspecified"] ?? "未指定";
}

function formatUpdatedAt(updatedAt?: string | null): string {
  if (!updatedAt) return "—";
  const fractionIndex = updatedAt.indexOf(".");
  const withoutFraction = fractionIndex >= 0 ? updatedAt.slice(0, fractionIndex) : updatedAt;
  return (withoutFraction.endsWith("Z") ? withoutFraction.slice(0, -1) : withoutFraction).replace("T", " ");
}
