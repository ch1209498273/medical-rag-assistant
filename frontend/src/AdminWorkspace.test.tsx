import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import AdminWorkspace from "./AdminWorkspace";

vi.mock("./FeedbackReviewPage", () => ({
  default: () => <div>反馈审核内容</div>,
}));

describe("AdminWorkspace", () => {
  it("keeps 资料管理 as the top-level entry and exposes the feedback tab", async () => {
    render(<AdminWorkspace />);
    expect(screen.getByRole("tab", { name: "制度资料" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "反馈审核" })).toBeInTheDocument();

    await userEvent.setup().click(screen.getByRole("tab", { name: "反馈审核" }));
    expect(screen.getByText("反馈审核内容")).toBeInTheDocument();
  });

  it("supports a deep link that opens the feedback tab for a demo or review capture", () => {
    render(<AdminWorkspace initialTab="feedback" />);
    expect(screen.getByRole("tab", { name: "反馈审核" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("反馈审核内容")).toBeInTheDocument();
  });
});
