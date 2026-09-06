export type ChatStage =
  | "accepted"
  | "rewriting"
  | "preflight"
  | "routing"
  | "retrieving"
  | "reranking"
  | "generating"
  | "validating"
  | "verifying";

export type WorkflowRoute = "direct" | "verify" | "clarify" | "out_of_scope";
export type WorkflowOutcome = "answered" | "refused" | "error" | "cancelled";
export type WorkflowVerifierStatus = "not_run" | "passed" | "failed" | "unavailable";
export type WorkflowSummary = {
  workflow_version: "baseline_v1" | "agent_workflow_v2a";
  run_id: string;
  route: WorkflowRoute | null;
  outcome: WorkflowOutcome;
  verifier_status: WorkflowVerifierStatus;
  http_calls: number;
  elapsed_ms: number | null;
  reason_code: string | null;
};

export type AudienceScope =
  | "unspecified"
  | "all_staff"
  | "nurse"
  | "doctor"
  | "pharmacist"
  | "administrator";

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
  table_id?: string | null;
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
        workflow_summary?: WorkflowSummary | null;
      };
    }
  | {
      type: "error";
      data: {
        session_id?: string;
        message_id?: string;
        reason_code: string;
        workflow_summary?: WorkflowSummary | null;
      };
    };

export type Feedback = {
  message_id: string;
  helpful: boolean;
  created_at: string;
  reason?: FeedbackReason | null;
};

export type FeedbackReason =
  | "not_answered"
  | "missing_step"
  | "version_mismatch"
  | "citation_mismatch"
  | "too_slow";

export type SessionMessage = {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  status: string;
  rewritten_question?: string | null;
  citations: Citation[];
  reason_code?: string | null;
  reference_answer?: string | null;
  audience_scope?: AudienceScope;
  workflow_summary?: WorkflowSummary | null;
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
  business_metadata?: DocumentBusinessMetadata | null;
};

export type DocumentBusinessMetadata = {
  version_id: string;
  content_type: "policy" | "training" | "procedure" | "other";
  applicable_scope: "unspecified" | "all_staff" | "nurse" | "doctor" | "pharmacist" | "administrator";
  effective_from: string | null;
  review_due_at: string | null;
  business_status: "draft" | "approved" | "superseded" | "retired" | "unknown";
  owner_role: string | null;
  supersedes_version_id: string | null;
  updated_at: string;
};

export type DocumentBusinessMetadataInput = Omit<DocumentBusinessMetadata, "version_id" | "updated_at">;

export type ScanResult = {
  results: DocumentRow[];
  deactivated: unknown[];
  snapshot_complete: boolean;
};

export type FeedbackTriagePriority = "P0" | "P1" | "P2" | "P3";
export type FeedbackReviewStatus =
  | "unreviewed"
  | "in_review"
  | "approved"
  | "rejected"
  | "adjudication_required";
export type FeedbackPromotionStatus =
  | "not_promoted"
  | "silver"
  | "golden_v2_candidate"
  | "golden_v2";
export type FeedbackAnswerStatus = "answered" | "refused" | "error";
export type FeedbackEvidenceStatus = "available" | "unavailable" | "version_mismatch" | "unsafe";
export type FeedbackReviewerRole = "product" | "engineering" | "clinical_reviewer";
export type FeedbackReviewDecision = "approve" | "reject" | "needs_adjudication";

export type FeedbackReviewSummary = Record<string, number>;

export type FeedbackCaseSummary = {
  case_id: string;
  triage_priority: FeedbackTriagePriority;
  reason_code: string | null;
  question_preview: string;
  source_version: string | null;
  review_status: FeedbackReviewStatus;
  promotion_status: FeedbackPromotionStatus;
  collected_at: string;
};

export type FeedbackEvidenceReference = {
  chunk_id: string;
  source_version: string;
  file_name: string;
  heading_path: string[];
  page: number | null;
  page_end: number | null;
  paragraph_start: number | null;
  paragraph_end: number | null;
  table_id: string | null;
  excerpt: string;
};

export type FeedbackEvidence = {
  status: FeedbackEvidenceStatus;
  reason_code: string | null;
  references: FeedbackEvidenceReference[];
};

export type FeedbackEventView = {
  event_id: string;
  helpful: boolean;
  feedback_reason: string | null;
  event_at: string;
  source: "user_click" | "admin_correction" | "import";
  event_schema_version: number;
};

export type FeedbackReviewView = {
  review_id: string;
  reviewer_role: FeedbackReviewerRole;
  decision: FeedbackReviewDecision;
  evidence_ok: boolean | null;
  points_ok: boolean | null;
  safety_ok: boolean | null;
  note_code: string | null;
  review_version: string;
  reviewed_at: string;
};

export type FeedbackPromotion = {
  promotion_id: string;
  target_set_id: string | null;
  target_version: string | null;
  target_split: "dev" | "holdout";
  promotion_reason: string | null;
  manifest_id: string | null;
  promoted_at: string;
};

export type FeedbackPromotionInput = {
  target_set_id: string;
  target_version: string;
  target_split: "dev" | "holdout";
  manifest_id: string;
};

export type FeedbackCaseDetail = {
  case_id: string;
  question: string | null;
  answer: string | null;
  redaction_status: "passed" | "review_required" | "blocked";
  answer_status: FeedbackAnswerStatus;
  reason_code: string | null;
  source_version: string | null;
  retrieval_profile: string | null;
  citation_chunk_ids: string[];
  citation_count: number;
  model_id: string | null;
  prompt_version: string | null;
  latency_bucket: string | null;
  audience_scope: AudienceScope;
  collected_at: string;
  triage_priority: FeedbackTriagePriority;
  triage_score: number;
  triage_reasons: string[];
  review_status: FeedbackReviewStatus;
  promotion_status: FeedbackPromotionStatus;
  event_count: number;
};

export type FeedbackCaseDetailResponse = {
  case: FeedbackCaseDetail;
  events: FeedbackEventView[];
  reviews: FeedbackReviewView[];
  evidence: FeedbackEvidence;
  promotion: PromotionCheck;
  promotions: FeedbackPromotion[];
};

export type FeedbackReviewInput = {
  reviewer_role: FeedbackReviewerRole;
  decision: FeedbackReviewDecision;
  evidence_ok: boolean | null;
  points_ok: boolean | null;
  safety_ok: boolean | null;
  note_code: string | null;
  review_version: string;
};

export type FeedbackCaseFilters = {
  priority?: FeedbackTriagePriority;
  review_status?: FeedbackReviewStatus;
  reason_code?: string;
  answer_status?: FeedbackAnswerStatus;
  source_version?: string;
  limit?: number;
};

export type PromotionCheck = {
  status: "not_promoted" | "golden_v2_candidate";
  reasons: string[];
};
