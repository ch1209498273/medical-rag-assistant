import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import WorkflowProgress from "./WorkflowProgress";
import type { WorkflowSummary } from "./api/types";

const summary: WorkflowSummary = {
  workflow_version: "agent_workflow_v2a",
  run_id: "run-12345678",
  route: "verify",
  outcome: "answered",
  verifier_status: "passed",
  http_calls: 4,
  elapsed_ms: 820,
  reason_code: null,
};

describe("WorkflowProgress", () => {
  it("shows fixed, user-facing process facts and does not expose the run id", () => {
    render(<WorkflowProgress summary={summary} />);

    expect(screen.getByText("本次处理过程")).toBeInTheDocument();
    expect(screen.getByText("需核验路径")).toBeInTheDocument();
    expect(screen.getByText("核验通过")).toBeInTheDocument();
    expect(screen.getByText("4 次")).toBeInTheDocument();
    expect(screen.queryByText("run-12345678")).not.toBeInTheDocument();
  });

  it("does not call an unrun verifier passed", () => {
    render(
      <WorkflowProgress
        summary={{ ...summary, verifier_status: "not_run", route: "direct" }}
      />,
    );

    expect(screen.getByText("未执行核验")).toBeInTheDocument();
    expect(screen.queryByText("核验通过")).not.toBeInTheDocument();
  });

  it("explains that old history has no recorded process summary", () => {
    render(<WorkflowProgress summary={null} />);

    expect(screen.getByText("历史执行信息未记录")).toBeInTheDocument();
  });

  it("maps an unknown reason to a fixed label instead of rendering provider text", () => {
    render(
      <WorkflowProgress
        summary={{
          ...summary,
          outcome: "error",
          verifier_status: "unavailable",
          reason_code: "provider secret should never render",
        }}
      />,
    );

    expect(screen.getAllByText("处理未完成").length).toBeGreaterThan(0);
    expect(screen.queryByText("provider secret should never render")).not.toBeInTheDocument();
  });
});
