import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import FeedbackButtons from "./FeedbackButtons";
import { saveFeedback } from "./api/client";

vi.mock("./api/client", () => ({
  saveFeedback: vi.fn(),
}));

const mockedSaveFeedback = vi.mocked(saveFeedback);

describe("FeedbackButtons", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedSaveFeedback.mockResolvedValue({
      message_id: "a1",
      helpful: false,
      reason: "missing_step",
      created_at: "2026-08-23T00:00:00Z",
    });
  });

  it("asks for a controlled reason before saving an unhelpful vote", async () => {
    const user = userEvent.setup();
    render(<FeedbackButtons messageId="a1" />);

    await user.click(screen.getByRole("button", { name: "无用" }));
    expect(screen.getByLabelText("无用原因")).toBeInTheDocument();
    expect(mockedSaveFeedback).not.toHaveBeenCalled();

    await user.selectOptions(screen.getByLabelText("无用原因"), "missing_step");
    await user.click(screen.getByRole("button", { name: "提交无用反馈" }));

    expect(mockedSaveFeedback).toHaveBeenCalledWith("a1", false, "missing_step");
    expect(await screen.findByText("反馈已保存")).toBeInTheDocument();
  });

  it("restores a saved reason without allowing a reason on helpful feedback", () => {
    render(
      <FeedbackButtons
        messageId="a1"
        initialHelpful={false}
        initialReason="too_slow"
      />,
    );

    expect(screen.getByLabelText("无用原因")).toHaveValue("too_slow");
    expect(screen.getByRole("button", { name: "有用" })).toBeInTheDocument();
  });
});
