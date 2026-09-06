import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import FeedbackReviewPage from "./FeedbackReviewPage";
import {
  getFeedbackCase,
  getFeedbackReviewSummary,
  listFeedbackCases,
  promoteFeedbackCase,
} from "./api/client";

vi.mock("./api/client", () => ({
  getFeedbackCase: vi.fn(),
  getFeedbackReviewSummary: vi.fn(),
  listFeedbackCases: vi.fn(),
  saveFeedbackReview: vi.fn(),
  promoteFeedbackCase: vi.fn(),
}));

const mockedGetFeedbackCase = vi.mocked(getFeedbackCase);
const mockedGetFeedbackReviewSummary = vi.mocked(getFeedbackReviewSummary);
const mockedListFeedbackCases = vi.mocked(listFeedbackCases);
const mockedPromoteFeedbackCase = vi.mocked(promoteFeedbackCase);

describe("FeedbackReviewPage data flywheel promotion", () => {
  beforeEach(() => {
    mockedGetFeedbackReviewSummary.mockResolvedValue({
      review_unreviewed: 0,
      promotion_golden_v2_candidate: 0,
    });
    mockedListFeedbackCases.mockResolvedValue([
      {
        case_id: "fc_safe",
        triage_priority: "P2",
        reason_code: "missing_step",
        question_preview: "透析中低血压怎么处理？",
        source_version: "2026-standard-manual-v1",
        review_status: "approved",
        promotion_status: "not_promoted",
        collected_at: "2026-09-06T01:00:00Z",
      },
    ]);
    mockedGetFeedbackCase.mockResolvedValue({
      case: {
        case_id: "fc_safe",
        question: "透析中低血压怎么处理？",
        answer: "请按本单位流程处理。",
        redaction_status: "passed",
        answer_status: "answered",
        reason_code: "missing_step",
        source_version: "2026-standard-manual-v1",
        retrieval_profile: "baseline_v1+vector+c1",
        citation_chunk_ids: ["chunk-1"],
        citation_count: 1,
        model_id: "deepseek-v4-flash",
        prompt_version: "c1",
        latency_bucket: "lt_5s",
        audience_scope: "nurse",
        collected_at: "2026-09-06T01:00:00Z",
        triage_priority: "P2",
        triage_score: 4,
        triage_reasons: ["user_unhelpful"],
        review_status: "approved",
        promotion_status: "not_promoted",
        event_count: 1,
      },
      events: [],
      reviews: [],
      evidence: {
        status: "available",
        reason_code: null,
        references: [],
      },
      promotion: {
        status: "golden_v2_candidate",
        reasons: ["review_approved"],
      },
      promotions: [],
    });
    mockedPromoteFeedbackCase.mockResolvedValue({
      promotion: {
        status: "golden_v2_candidate",
        reasons: ["review_approved"],
      },
      record: {
        promotion_id: "pr_1",
        target_set_id: "2026-standard-manual-v1-golden-v2",
        target_version: "candidate-001",
        target_split: "dev",
        promotion_reason: "review_approved",
        manifest_id: "manifest-synthetic-001",
        promoted_at: "2026-09-06T01:00:00Z",
      },
    });
  });

  it("appends a reviewed case to the selected golden-v2 split", async () => {
    render(<FeedbackReviewPage />);

    const button = await screen.findByRole("button", { name: "加入 golden-v2 候选集" });
    await userEvent.setup().click(button);

    await waitFor(() => expect(mockedPromoteFeedbackCase).toHaveBeenCalledWith(
      "fc_safe",
      {
        target_set_id: "2026-standard-manual-v1-golden-v2",
        target_version: "candidate-001",
        target_split: "dev",
        manifest_id: "manifest-synthetic-001",
      },
    ));
    expect(await screen.findByText("已加入 golden-v2 候选集")) .toBeInTheDocument();
  });
});
