import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatPage from "./ChatPage";
import { getSession, listSessions, saveFeedback } from "./api/client";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return {
    ...actual,
    getSession: vi.fn(),
    listSessions: vi.fn(),
    saveFeedback: vi.fn(),
  };
});

const mockedGetSession = vi.mocked(getSession);
const mockedListSessions = vi.mocked(listSessions);
const mockedSaveFeedback = vi.mocked(saveFeedback);

const citation = {
  reference_id: "S1",
  file_name: "护理制度.docx",
  heading_path: ["低血压"],
  paragraph_start: 3,
  paragraph_end: 4,
  excerpt: "虚构公开样例",
};

function sseResponse(frames: string): Response {
  return new Response(frames, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("ChatPage fetch/SSE boundary", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    mockedGetSession.mockResolvedValue({ session_id: "s1", messages: [] });
    mockedListSessions.mockResolvedValue([]);
    mockedSaveFeedback.mockResolvedValue({
      message_id: "a1",
      helpful: true,
      created_at: "2026-08-23T00:00:00Z",
    });
  });

  it("parses a first-question SSE response through fetch and renders its citation", async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([
      "event: status\ndata: {\"stage\":\"accepted\",\"session_id\":\"s1\",\"message_id\":\"u1\"}",
      "event: status\ndata: {\"stage\":\"rewriting\",\"session_id\":\"s1\"}",
      "event: answer_delta\ndata: {\"text\":\"应先评估原因。\",\"source_ids\":[\"S1\"]}",
      `event: final\ndata: ${JSON.stringify({ session_id: "s1", message_id: "a1", refused: false, citations: [citation] })}`,
    ].join("\n\n") + "\n\n"));

    const user = userEvent.setup();
    render(<ChatPage />);
    await user.type(screen.getByRole("textbox", { name: "问题" }), "透析中低血压怎么处理？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText("应先评估原因。")).toBeInTheDocument();
    expect(localStorage.getItem("hemodialysis.currentSessionId")).toBe("s1");
    expect(fetchMock).toHaveBeenCalledWith("/api/chat/stream", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ question: "透析中低血压怎么处理？" }),
    }));
    const details = screen.getByText("引用依据（1）").closest("details");
    expect(details).not.toHaveAttribute("open");
    await user.click(screen.getByText("引用依据（1）"));
    expect(await screen.findByText("虚构公开样例")).toBeInTheDocument();
  });

  it("renders grounded and unverified sections through the real SSE parser", async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([
      "event: status\ndata: {\"stage\":\"accepted\",\"session_id\":\"s1\",\"message_id\":\"u1\"}",
      "event: answer_delta\ndata: {\"text\":\"资料可确认应先申请。\",\"source_ids\":[\"S1\"]}",
      `event: final\ndata: ${JSON.stringify({
        session_id: "s1",
        message_id: "a1",
        refused: false,
        citations: [citation],
        reference_answer: "审批时长需向负责人确认。",
      })}`,
    ].join("\n\n") + "\n\n"));

    const user = userEvent.setup();
    render(<ChatPage />);
    await user.type(screen.getByRole("textbox", { name: "问题" }), "请假流程和审批时长？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText("资料依据答案（可核验）")).toBeInTheDocument();
    expect(screen.getByText("资料可确认应先申请。")).toBeInTheDocument();
    expect(screen.getByText("模型通用参考（未核验）")).toBeInTheDocument();
    expect(screen.getByText("审批时长需向负责人确认。")).toBeInTheDocument();
  });

  it("sends a follow-up through fetch and renders a safe streamed error", async () => {
    fetchMock
      .mockResolvedValueOnce(sseResponse([
        "event: status\ndata: {\"stage\":\"accepted\",\"session_id\":\"s1\",\"message_id\":\"u1\"}",
        "event: answer_delta\ndata: {\"text\":\"第一条答案\",\"source_ids\":[\"S1\"]}",
        `event: final\ndata: ${JSON.stringify({ session_id: "s1", message_id: "a1", refused: false, citations: [citation] })}`,
      ].join("\n\n") + "\n\n"))
      .mockResolvedValueOnce(sseResponse(
        "event: status\ndata: {\"stage\":\"accepted\",\"session_id\":\"s1\",\"message_id\":\"u2\"}\n\n" +
        "event: error\ndata: {\"session_id\":\"s1\",\"message_id\":\"a2\",\"reason_code\":\"CHAT_UNAVAILABLE\"}\n\n",
      ));

    const user = userEvent.setup();
    render(<ChatPage />);
    const input = screen.getByRole("textbox", { name: "问题" });
    await user.type(input, "第一条问题");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText("第一条答案");
    await user.type(input, "那怎么办？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("暂时无法完成回答");
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chat/stream", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ question: "那怎么办？", session_id: "s1" }),
    }));
    const composer = document.querySelector(".chat-composer-wrap");
    expect(composer).not.toBeNull();
    expect(within(composer as HTMLElement).getByText("状态：失败")).toBeInTheDocument();
    expect(screen.queryByText("provider")).not.toBeInTheDocument();
  });
});
