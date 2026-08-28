export type ChatStage =
  | "accepted"
  | "rewriting"
  | "retrieving"
  | "reranking"
  | "generating"
  | "validating";

export type RuntimeMode = "demo" | "cloud";

export type ProviderState = "configured" | "missing" | "optional_missing" | "not_required";

export type HealthResponse = {
  runtime_mode: RuntimeMode;
  providers: {
    deepseek: ProviderState;
    siliconflow: ProviderState;
    minimax: ProviderState;
  };
};

export type Citation = {
  reference_id: string;
  file_name: string;
  heading_path: string[];
  page?: number | null;
  page_end?: number | null;
  paragraph_start?: number | null;
  paragraph_end?: number | null;
  excerpt: string;
};

export type ChatEvent =
  | {
      type: "status";
      data: { stage: ChatStage; session_id?: string; message_id?: string };
    }
  | { type: "answer_delta"; data: { text: string; source_ids: string[] } }
  | {
      type: "final";
      data: {
        session_id: string;
        message_id: string;
        refused: boolean;
        citations: Citation[];
        reason_code?: string;
        reference_answer?: string | null;
      };
    }
  | {
      type: "error";
      data: { session_id?: string; message_id?: string; reason_code: string };
    };

export type Feedback = {
  message_id: string;
  helpful: boolean;
  created_at: string;
};

export type SessionMessage = {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  status: string;
  rewritten_question?: string | null;
  citations: Citation[];
  reason_code?: string | null;
  reference_answer?: string | null;
  created_at: string;
  feedback?: Feedback | null;
  helpful?: boolean | null;
};

export type SessionResponse = {
  session_id: string;
  messages: SessionMessage[];
};

export type SessionSummary = {
  session_id: string;
  title: string;
  created_at: string;
  last_activity_at: string;
  message_count: number;
};

export type DocumentRow = {
  document_id: number;
  file_name: string;
  version_id?: string | null;
  status: string;
  updated_at?: string | null;
  failure_reason?: string | null;
};

export type ScanResult = {
  results: DocumentRow[];
  deactivated: unknown[];
  snapshot_complete: boolean;
};
