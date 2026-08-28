import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ClientStreamError,
  getHealth,
  getSession,
  listSessions,
  listDocuments,
  reindexDocument,
  saveFeedback,
  scanDocuments,
  streamChat,
} from "./client";
import type { ChatEvent } from "./types";

const windowsPath = (...parts: string[]) =>
  ["C:", ...parts].join(String.fromCharCode(92));

const sse = (frames: string[]) =>
  new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        const encoder = new TextEncoder();
        frames.forEach((frame) => controller.enqueue(encoder.encode(frame)));
        controller.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );

describe("getHealth", () => {
  afterEach(() => vi.restoreAllMocks());

  it("parses the backend runtime mode and provider statuses", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          runtime_mode: "demo",
          providers: {
            deepseek: "not_required",
            siliconflow: "not_required",
            minimax: "not_required",
          },
          ignored: "field",
        }),
        { status: 200 },
      ),
    );

    await expect(getHealth()).resolves.toEqual({
      runtime_mode: "demo",
      providers: {
        deepseek: "not_required",
        siliconflow: "not_required",
        minimax: "not_required",
      },
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/health", { method: "GET" });
  });

  it("fails closed when a provider status is missing", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          runtime_mode: "demo",
          providers: { deepseek: "not_required", siliconflow: "not_required" },
        }),
        { status: 200 },
      ),
    );

    await expect(getHealth()).rejects.toMatchObject({ code: "HEALTH_UNAVAILABLE" });
  });
});

describe("streamChat", () => {
  afterEach(() => vi.restoreAllMocks());

  it("parses event/data frames split across response chunks", async () => {
    const first = `event: status\ndata: {"stage":"accepted","session_id":"s1","message_id":"u1"}\n\n`;
    const second = `event: final\ndata: {"session_id":"s1","message_id":"a1","refused":false,"citations":[]}\n\n`;
    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          const encoder = new TextEncoder();
          const splitAt = Math.floor(first.length / 2);
          controller.enqueue(encoder.encode(first.slice(0, splitAt)));
          controller.enqueue(encoder.encode(first.slice(splitAt) + second));
          controller.close();
        },
      }),
      { status: 200 },
    );
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response);
    const events: string[] = [];

    await streamChat("请说明制度", null, (event) => events.push(event.type));

    expect(events).toEqual(["status", "final"]);
    expect(fetch).toHaveBeenCalledWith(
      "/api/chat/stream",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ question: "请说明制度" }),
      }),
    );
  });

  it("accepts ordinary slash notation in answer text and citation excerpts", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sse([
        'event: status\ndata: {"stage":"accepted","session_id":"s1","message_id":"u1"}\n\n',
        'event: answer_delta\ndata: {"text":"依据 GB/T 8982，可按制度执行。","source_ids":["S1"]}\n\n',
        'event: final\ndata: {"session_id":"s1","message_id":"a1","refused":false,"citations":[{"reference_id":"S1","file_name":"制度.pdf","heading_path":["血透制度"],"page":1,"page_end":1,"excerpt":"依据 GB/T 8982 的相关要求。"}]}\n\n',
      ]),
    );
    const events: string[] = [];

    await expect(streamChat("请说明 GB/T 8982", null, (event) => events.push(event.type))).resolves.toBeUndefined();

    expect(events).toEqual(["status", "answer_delta", "final"]);
  });

  it("accepts a clearly-labelled reference answer only on a grounding refusal", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sse([
        'event: status\ndata: {"stage":"accepted","session_id":"s1","message_id":"u1"}\n\n',
        'event: final\ndata: {"session_id":"s1","message_id":"a1","refused":true,"reason_code":"INSUFFICIENT_EVIDENCE","citations":[],"reference_answer":"通用学习参考"}\n\n',
      ]),
    );
    const events: ChatEvent[] = [];

    await expect(streamChat("问题", null, (event) => events.push(event))).resolves.toBeUndefined();

    expect(events[1]).toMatchObject({
      type: "final",
      data: { refused: true, reference_answer: "通用学习参考" },
    });
  });

  it("accepts a separately labelled reference answer on an answered SSE result", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sse([
        'event: final\ndata: {"session_id":"s1","message_id":"a1","refused":false,"citations":[{"reference_id":"S1","file_name":"制度.docx","heading_path":["正文"],"page":null,"page_end":null,"paragraph_start":1,"paragraph_end":1,"excerpt":"应先申请。"}],"reference_answer":"单独标注的通用参考"}\n\n',
      ]),
    );
    const events: ChatEvent[] = [];

    await expect(streamChat("问题", null, (event) => events.push(event))).resolves.toBeUndefined();
    expect(events[0]).toMatchObject({
      type: "final",
      data: { refused: false, reference_answer: "单独标注的通用参考" },
    });
  });

  it("fails closed when a frame is not valid JSON", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sse(["event: final\ndata: {not-json}\n\n"]),
    );

    await expect(streamChat("问题", "s1", () => undefined)).rejects.toMatchObject({
      code: "CLIENT_STREAM_INVALID",
    } satisfies Partial<ClientStreamError>);
  });

  it("uses same-origin typed routes for sessions, feedback, and documents", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock
      .mockResolvedValueOnce(new Response(JSON.stringify({ session_id: "s1", messages: [] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ message_id: "a1", helpful: true, created_at: "2026-08-23T00:00:00Z" }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ documents: [] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ results: [], deactivated: [], snapshot_complete: true }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ document_id: 7, file_name: "制度.docx", status: "active" }), { status: 200 }));

    await expect(getSession("s1")).resolves.toMatchObject({ session_id: "s1" });
    await expect(saveFeedback("a1", true)).resolves.toMatchObject({ helpful: true });
    await expect(listDocuments()).resolves.toEqual([]);
    await expect(scanDocuments()).resolves.toMatchObject({ snapshot_complete: true });
    await expect(reindexDocument(7)).resolves.toMatchObject({ document_id: 7 });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/chat/sessions/s1",
      "/api/chat/messages/a1/feedback",
      "/api/documents",
      "/api/documents/scan",
      "/api/documents/7/reindex",
    ]);
  });

  it("lists safe session metadata without exposing message previews", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          sessions: [
            {
              session_id: "s2",
              title: "低血压处理",
              created_at: "2026-08-23T00:00:00Z",
              last_activity_at: "2026-08-23T00:01:00Z",
              message_count: 2,
            },
          ],
        }),
        { status: 200 },
      ),
    );

    await expect(listSessions()).resolves.toEqual([
      {
        session_id: "s2",
        title: "低血压处理",
        created_at: "2026-08-23T00:00:00Z",
        last_activity_at: "2026-08-23T00:01:00Z",
        message_count: 2,
      },
    ]);
    expect(fetch).toHaveBeenCalledWith("/api/chat/sessions?limit=20", { method: "GET" });
  });

  it("rejects malformed session metadata instead of rendering it", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          sessions: [
            {
              session_id: "../../secret",
              created_at: "not-a-time",
              last_activity_at: "not-a-time",
              message_count: -1,
            },
          ],
        }),
        { status: 200 },
      ),
    );

    await expect(listSessions()).rejects.toMatchObject({ code: "SESSION_LIST_UNAVAILABLE" });
  });

  it("converts a tampered restored message into a safe error DTO", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "s1",
          messages: [
            {
              message_id: "a1",
              role: "assistant",
              content: `Traceback: ${windowsPath("private", "provider.log")}`,
              status: "provider-stack",
              rewritten_question: windowsPath("private", "prompt"),
              citations: [
                {
                  reference_id: "S1",
                  file_name: windowsPath("private", "policy.docx"),
                  heading_path: [],
                  excerpt: "raw",
                },
              ],
              reason_code: "raw-provider-detail",
              created_at: "not-a-timestamp",
            },
          ],
        }),
        { status: 200 },
      ),
    );

    const result = await getSession("s1");

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toMatchObject({
      role: "assistant",
      status: "error",
      content: "当前会话记录无法安全恢复。",
      citations: [],
      reason_code: "CHAT_UNAVAILABLE",
    });
    expect(JSON.stringify(result)).not.toContain("provider");
    expect(JSON.stringify(result)).not.toContain("private");
  });

  it("restores an answered message with a separate unverified reference", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "s1",
          messages: [
            {
              message_id: "a1",
              role: "assistant",
              content: "资料可确认应先申请。",
              status: "answered",
              rewritten_question: null,
              citations: [],
              reason_code: null,
              reference_answer: "审批时长需向负责人确认。",
              created_at: "2026-08-26T00:00:00Z",
            },
          ],
        }),
        { status: 200 },
      ),
    );

    const result = await getSession("s1");

    expect(result.messages[0]).toMatchObject({
      status: "answered",
      reference_answer: "审批时长需向负责人确认。",
    });
  });

  it("rejects an incomplete citation in live SSE instead of passing it to the UI", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sse([
        'event: final\ndata: {"session_id":"s1","message_id":"a1","refused":false,"citations":[{"reference_id":"S1","file_name":"制度.docx"}]}\n\n',
      ]),
    );

    await expect(streamChat("问题", "s1", () => undefined)).rejects.toMatchObject({
      code: "CLIENT_STREAM_INVALID",
    });
  });

  it("drops tampered document metadata before it reaches the UI", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          documents: [
            {
              document_id: "bad",
              file_name: windowsPath("private", "provider.docx"),
              version_id: "..\\secret",
              status: "provider-stack",
              updated_at: "Traceback",
              failure_reason: "raw provider error",
            },
          ],
        }),
        { status: 200 },
      ),
    );

    await expect(listDocuments()).resolves.toEqual([]);
  });
});
