import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatPage from "./ChatPage";
import { getSession, listSessions, saveFeedback, streamChat } from "./api/client";

vi.mock("./api/client", () => ({
  getSession: vi.fn(),
  listSessions: vi.fn(),
  saveFeedback: vi.fn(),
  streamChat: vi.fn(),
}));

const mockedGetSession = vi.mocked(getSession);
const mockedListSessions = vi.mocked(listSessions);
const mockedSaveFeedback = vi.mocked(saveFeedback);
const mockedStreamChat = vi.mocked(streamChat);

const citation = {
  reference_id: "S1",
  file_name: "护理制度.docx",
  heading_path: ["低血压"],
  paragraph_start: 3,
  paragraph_end: 4,
  excerpt: "虚构公开样例",
};

describe("ChatPage", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mockedGetSession.mockResolvedValue({ session_id: "s1", messages: [] });
    mockedListSessions.mockResolvedValue([]);
    mockedSaveFeedback.mockResolvedValue({
      message_id: "a1",
      helpful: true,
      created_at: "2026-08-23T00:00:00Z",
    });
  });

  it("persists accepted sessions, explains rewriting, and reveals a collapsed citation", async () => {
    const user = userEvent.setup();
    let finish!: () => void;
    mockedStreamChat.mockImplementation(async (_question, _sessionId, onEvent) => {
      onEvent({ type: "status", data: { stage: "accepted", session_id: "s1", message_id: "u1" } });
      onEvent({ type: "status", data: { stage: "rewriting" } });
      await new Promise<void>((resolve) => {
        finish = resolve;
      });
      onEvent({ type: "answer_delta", data: { text: "应先评估原因。", source_ids: ["S1"] } });
      onEvent({
        type: "final",
        data: { session_id: "s1", message_id: "a1", refused: false, citations: [citation] },
      });
    });

    render(<ChatPage />);
    await user.type(screen.getByRole("textbox", { name: "问题" }), "那怎么预防？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText("正在理解追问")).toBeInTheDocument();
    expect(localStorage.getItem("hemodialysis.currentSessionId")).toBe("s1");
    finish();

    expect(await screen.findByText("应先评估原因。")).toBeInTheDocument();
    const details = screen.getByText("引用依据（1）").closest("details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");
    await user.click(screen.getByText("引用依据（1）"));
    expect(await screen.findByText("虚构公开样例")).toBeInTheDocument();
  });

  it("restores the saved session without posting a new question and clears it when starting over", async () => {
    localStorage.setItem("hemodialysis.currentSessionId", "s1");
    mockedGetSession.mockResolvedValue({
      session_id: "s1",
      messages: [
        {
          message_id: "u1",
          role: "user",
          content: "透析中低血压怎么处理？",
          status: "submitted",
          rewritten_question: null,
          citations: [],
          reason_code: null,
          created_at: "2026-08-23T00:00:00Z",
          feedback: null,
        },
        {
          message_id: "a1",
          role: "assistant",
          content: "应先评估原因。",
          status: "answered",
          rewritten_question: null,
          citations: [],
          reason_code: null,
          created_at: "2026-08-23T00:00:01Z",
          feedback: null,
        },
      ],
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    expect(await screen.findByText("透析中低血压怎么处理？")).toBeInTheDocument();
    expect(mockedGetSession).toHaveBeenCalledWith("s1");
    expect(mockedStreamChat).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "新建会话" }));
    expect(localStorage.getItem("hemodialysis.currentSessionId")).toBeNull();
    expect(screen.getByText("还没有问题，先从一条制度问题开始。")).toBeInTheDocument();
  });

  it("shows safe session history and restores a selected conversation", async () => {
    mockedListSessions.mockResolvedValue([
      {
        session_id: "s2",
        title: "低血压处理",
        created_at: "2026-08-23T00:00:00Z",
        last_activity_at: "2026-08-23T00:01:00Z",
        message_count: 2,
      },
      {
        session_id: "s1",
        title: "医保报销范围",
        created_at: "2026-08-22T00:00:00Z",
        last_activity_at: "2026-08-22T00:01:00Z",
        message_count: 1,
      },
    ]);
    mockedGetSession.mockResolvedValue({
      session_id: "s2",
      messages: [
        {
          message_id: "u2",
          role: "user",
          content: "历史问题",
          status: "submitted",
          rewritten_question: null,
          citations: [],
          reason_code: null,
          created_at: "2026-08-23T00:00:00Z",
          feedback: null,
        },
      ],
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    expect(await screen.findByRole("button", { name: "恢复会话：低血压处理" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "恢复会话：医保报销范围" })).toBeInTheDocument();
    expect(screen.getByText("低血压处理")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "恢复会话：低血压处理" }));

    expect(await screen.findByText("历史问题")).toBeInTheDocument();
    expect(mockedGetSession).toHaveBeenCalledWith("s2");
    expect(localStorage.getItem("hemodialysis.currentSessionId")).toBe("s2");
  });

  it("confirms feedback only after success and keeps a retryable error after failure", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _sessionId, onEvent) => {
      onEvent({ type: "status", data: { stage: "accepted", session_id: "s1", message_id: "u1" } });
      onEvent({ type: "answer_delta", data: { text: "答案", source_ids: [] } });
      onEvent({ type: "final", data: { session_id: "s1", message_id: "a1", refused: false, citations: [] } });
    });
    render(<ChatPage />);
    await user.type(screen.getByRole("textbox", { name: "问题" }), "问题");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    expect(await screen.findByText("答案")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "有用" }));
    expect(await screen.findByText("反馈已保存")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("反馈已保存");
    expect(mockedSaveFeedback).toHaveBeenCalledWith("a1", true);

    mockedSaveFeedback.mockRejectedValueOnce(new Error("untrusted backend detail must stay hidden"));
    await user.click(screen.getByRole("button", { name: "无用" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("反馈未保存");
    expect(screen.getByRole("alert")).not.toHaveTextContent("untrusted backend detail");
  });

  it("shows an explicit unverified reference card beneath a grounded refusal", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _sessionId, onEvent) => {
      onEvent({ type: "status", data: { stage: "accepted", session_id: "s1", message_id: "u1" } });
      onEvent({
        type: "final",
        data: {
          session_id: "s1",
          message_id: "a1",
          refused: true,
          reason_code: "INSUFFICIENT_EVIDENCE",
          citations: [],
          reference_answer: "这是一段仅供学习的通用参考。",
        },
      });
    });
    render(<ChatPage />);
    await user.type(screen.getByRole("textbox", { name: "问题" }), "问题");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText("模型通用参考（未核验）")).toBeInTheDocument();
    expect(screen.getByText("这是一段仅供学习的通用参考。")).toBeInTheDocument();
    expect(screen.getByText(/未经过本单位制度或医保资料核验/)).toBeInTheDocument();
    expect(screen.queryByText("引用依据（0）")).not.toBeInTheDocument();
  });

  it("keeps a partial grounded answer and its unverified reference in separate cards", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _sessionId, onEvent) => {
      onEvent({ type: "status", data: { stage: "accepted", session_id: "s1", message_id: "u1" } });
      onEvent({ type: "answer_delta", data: { text: "资料可确认应先申请。", source_ids: ["S1"] } });
      onEvent({
        type: "final",
        data: {
          session_id: "s1",
          message_id: "a1",
          refused: false,
          citations: [citation],
          reference_answer: "审批时长通常需向负责人确认。",
        },
      });
    });
    render(<ChatPage />);
    await user.type(screen.getByRole("textbox", { name: "问题" }), "请假流程和审批时长？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText("资料可确认应先申请。")).toBeInTheDocument();
    expect(screen.getByText("资料依据答案（可核验）")).toBeInTheDocument();
    expect(screen.getByText("审批时长通常需向负责人确认。")).toBeInTheDocument();
    expect(screen.getByText("引用依据（1）")).toBeInTheDocument();
    expect(screen.getByLabelText("模型通用参考（未核验）")).toBeInTheDocument();
  });

  it("maps stream failures to safe text and never renders provider details", async () => {
    const user = userEvent.setup();
    mockedStreamChat.mockImplementation(async (_question, _sessionId, onEvent) => {
      onEvent({ type: "status", data: { stage: "accepted", session_id: "s1", message_id: "u1" } });
      onEvent({
        type: "error",
        data: { session_id: "s1", message_id: "a1", reason_code: "CHAT_UNAVAILABLE" },
      });
    });
    render(<ChatPage />);
    await user.type(screen.getByRole("textbox", { name: "问题" }), "问题");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("暂时无法完成回答");
    expect(alert).not.toHaveTextContent("provider");
    expect(alert).not.toHaveTextContent("secret");
    await waitFor(() => expect(screen.queryByText("引用依据")).not.toBeInTheDocument());
  });

  it("uses the latest assistant terminal state for the composer status", async () => {
    const user = userEvent.setup();
    mockedStreamChat
      .mockImplementationOnce(async (_question, _sessionId, onEvent) => {
        onEvent({ type: "status", data: { stage: "accepted", session_id: "s1", message_id: "u1" } });
        onEvent({ type: "answer_delta", data: { text: "第一条答案", source_ids: ["S1"] } });
        onEvent({ type: "final", data: { session_id: "s1", message_id: "a1", refused: false, citations: [citation] } });
      })
      .mockImplementationOnce(async (_question, _sessionId, onEvent) => {
        onEvent({ type: "status", data: { stage: "accepted", session_id: "s1", message_id: "u2" } });
        onEvent({ type: "error", data: { session_id: "s1", message_id: "a2", reason_code: "CHAT_UNAVAILABLE" } });
      });

    render(<ChatPage />);
    const input = screen.getByRole("textbox", { name: "问题" });
    await user.type(input, "第一条问题");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText("第一条答案");

    await user.type(input, "第二条追问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByRole("alert");

    const composer = document.querySelector(".chat-composer-wrap");
    expect(composer).not.toBeNull();
    expect(composer).toHaveTextContent("状态：失败");
    expect(composer).not.toHaveTextContent("状态：完成");
  });
});
