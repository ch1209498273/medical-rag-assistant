import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { getHealth } from "./api/client";
import type { HealthResponse } from "./api/types";

vi.mock("./api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api/client")>()),
  getHealth: vi.fn(),
}));

const mockedGetHealth = vi.mocked(getHealth);
const demoHealth: HealthResponse = {
  runtime_mode: "demo",
  providers: { deepseek: "not_required", siliconflow: "not_required", minimax: "not_required" },
};

beforeEach(() => {
  localStorage.clear();
  mockedGetHealth.mockResolvedValue(demoHealth);
});

describe("App shell", () => {
  it("presents the medical assistant name and the backend runtime mode", async () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "医疗知识问答助手" })).toBeInTheDocument();
    expect((await screen.findAllByText("无 Key 演示模式")).length).toBeGreaterThan(0);
  });

  it("shows an unavailable label when the health response cannot be loaded", async () => {
    mockedGetHealth.mockRejectedValueOnce(new Error("health unavailable"));
    render(<App />);

    expect((await screen.findAllByText("运行模式暂不可用")).length).toBeGreaterThan(0);
    expect(screen.queryByText("无 Key 演示模式")).not.toBeInTheDocument();
  });

  it("opens the policy chat with the document entry", () => {
    render(<App />);

    expect(screen.getByRole("button", { name: "制度问答" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "资料管理" })).toBeInTheDocument();
  });

  it("switches between the two in-scope administrator pages", async () => {
    localStorage.clear();
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "资料管理" }));
    expect(await screen.findByRole("heading", { name: "制度资料" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "制度问答" }));
    expect(screen.getByRole("heading", { name: "当前会话" })).toBeInTheDocument();
  });
});
