import { useState, type ReactNode } from "react";

import DocumentsPage from "./DocumentsPage";
import FeedbackReviewPage from "./FeedbackReviewPage";

type AdminTab = "documents" | "feedback";

export type AdminWorkspaceProps = {
  feedbackPage?: ReactNode;
};

export default function AdminWorkspace({ feedbackPage }: AdminWorkspaceProps) {
  const [tab, setTab] = useState<AdminTab>("documents");

  return (
    <section className="admin-workspace" aria-labelledby="admin-workspace-title">
      <header className="admin-workspace-header">
        <div>
          <p className="eyebrow">资料管理</p>
          <h2 id="admin-workspace-title">管理员工作台</h2>
          <p className="page-description">本机管理员使用；审核内容来自脱敏反馈投影，不改变问答生产链路。</p>
        </div>
      </header>
      <div className="admin-local-notice" role="note">
        当前版本为本机可信管理员模式，不代表登录、ACL 或多人协作权限。
      </div>
      <div className="admin-tabs" role="tablist" aria-label="管理员工作区">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "documents"}
          aria-controls="admin-tab-documents"
          className={tab === "documents" ? "admin-tab active" : "admin-tab"}
          onClick={() => setTab("documents")}
        >
          制度资料
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "feedback"}
          aria-controls="admin-tab-feedback"
          className={tab === "feedback" ? "admin-tab active" : "admin-tab"}
          onClick={() => setTab("feedback")}
        >
          反馈审核
        </button>
      </div>
      <div id="admin-tab-documents" role="tabpanel" hidden={tab !== "documents"}>
        {tab === "documents" && <DocumentsPage />}
      </div>
      <div id="admin-tab-feedback" role="tabpanel" hidden={tab !== "feedback"}>
        {tab === "feedback" && (feedbackPage ?? <FeedbackReviewPage />)}
      </div>
    </section>
  );
}
