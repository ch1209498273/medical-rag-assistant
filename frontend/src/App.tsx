import { useEffect, useState } from "react";

import ChatPage from "./ChatPage";
import AdminWorkspace from "./AdminWorkspace";
import { getHealth } from "./api/client";
import type { RuntimeMode } from "./api/types";

type Page = "chat" | "documents";

const APP_NAME = "医疗知识问答助手";

function runtimeModeLabel(runtimeMode: RuntimeMode | null): string {
  if (runtimeMode === "demo") return "无 Key 演示模式";
  if (runtimeMode === "cloud") return "自备 Key 云端模式";
  return "运行模式暂不可用";
}

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getHealth()
      .then((health) => {
        if (!cancelled) setRuntimeMode(health.runtime_mode);
      })
      .catch(() => {
        // Keep the explicit unavailable label; never infer a mode from the client.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const modeLabel = runtimeModeLabel(runtimeMode);

  return (
    <div className="app-shell">
      <aside className="side-navigation">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true">透</span>
          <div>
            <h1 className="brand-name">{APP_NAME}</h1>
            <span className="runtime-badge" role="status" aria-live="polite">{modeLabel}</span>
          </div>
        </div>
        <nav aria-label="主导航" className="main-navigation">
          <button
            type="button"
            className={page === "chat" ? "nav-button active" : "nav-button"}
            aria-current={page === "chat" ? "page" : undefined}
            onClick={() => setPage("chat")}
          >
            <span aria-hidden="true">问</span>
            制度问答
          </button>
          <button
            type="button"
            className={page === "documents" ? "nav-button active" : "nav-button"}
            aria-current={page === "documents" ? "page" : undefined}
            onClick={() => setPage("documents")}
          >
            <span aria-hidden="true">册</span>
            资料管理
          </button>
        </nav>
        <p className="navigation-note">仅供受信任的本机演示环境使用。</p>
      </aside>
      <main className="main-content">
        <header className="mobile-header">
          <span className="brand-name">{APP_NAME}</span>
          <span className="runtime-badge" role="status" aria-live="polite">{modeLabel}</span>
        </header>
        {page === "chat" ? <ChatPage /> : <AdminWorkspace />}
      </main>
    </div>
  );
}
