import { useState } from "react";

import { saveFeedback } from "./api/client";
import type { FeedbackReason } from "./api/types";

type FeedbackButtonsProps = {
  messageId: string;
  initialHelpful?: boolean | null;
  initialReason?: FeedbackReason | null;
};

const REASON_OPTIONS: Array<{ value: FeedbackReason; label: string }> = [
  { value: "not_answered", label: "没有回答问题" },
  { value: "missing_step", label: "缺少关键步骤" },
  { value: "version_mismatch", label: "制度版本可能不一致" },
  { value: "citation_mismatch", label: "引用依据不匹配" },
  { value: "too_slow", label: "回答等待时间过长" },
];

export default function FeedbackButtons({
  messageId,
  initialHelpful = null,
  initialReason = null,
}: FeedbackButtonsProps) {
  const [helpful, setHelpful] = useState<boolean | null>(initialHelpful);
  const [reason, setReason] = useState<FeedbackReason | "">(initialReason ?? "");
  const [showReasons, setShowReasons] = useState(initialHelpful === false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [failed, setFailed] = useState(false);
  const [validationError, setValidationError] = useState(false);

  const submit = async (value: boolean, selectedReason?: FeedbackReason) => {
    setSaving(true);
    setFailed(false);
    setSaved(false);
    setValidationError(false);
    try {
      if (value) {
        await saveFeedback(messageId, true);
      } else {
        await saveFeedback(messageId, false, selectedReason);
      }
      setHelpful(value);
      setSaved(true);
    } catch {
      setFailed(true);
    } finally {
      setSaving(false);
    }
  };

  const chooseHelpful = () => {
    setShowReasons(false);
    setReason("");
    void submit(true);
  };

  const chooseUnhelpful = () => {
    setShowReasons(true);
    setSaved(false);
    setFailed(false);
    setValidationError(false);
  };

  const submitUnhelpful = () => {
    if (!reason) {
      setValidationError(true);
      return;
    }
    void submit(false, reason);
  };

  return (
    <div className="feedback-control" aria-label="回答反馈">
      <span>这条回答有帮助吗？</span>
      <div className="feedback-actions">
        <button
          type="button"
          className={helpful === true ? "selected" : ""}
          aria-pressed={helpful === true}
          disabled={saving}
          onClick={chooseHelpful}
        >
          有用
        </button>
        <button
          type="button"
          className={helpful === false ? "selected" : ""}
          aria-pressed={helpful === false}
          disabled={saving}
          onClick={chooseUnhelpful}
        >
          无用
        </button>
      </div>
      {showReasons && (
        <div className="feedback-reason">
          <label htmlFor={`feedback-reason-${messageId}`}>无用原因</label>
          <select
            id={`feedback-reason-${messageId}`}
            aria-label="无用原因"
            value={reason}
            disabled={saving}
            onChange={(event) => {
              setReason(event.target.value as FeedbackReason | "");
              setValidationError(false);
            }}
          >
            <option value="">请选择一个原因</option>
            {REASON_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <button type="button" disabled={saving} onClick={submitUnhelpful}>
            提交无用反馈
          </button>
          {validationError && (
            <span className="feedback-error" role="alert">
              请选择一个原因后提交。
            </span>
          )}
        </div>
      )}
      {saving && (
        <span className="sr-only" role="status" aria-live="polite">
          正在保存反馈…
        </span>
      )}
      {saved && (
        <span className="feedback-saved" role="status" aria-live="polite">
          反馈已保存
        </span>
      )}
      {failed && (
        <span className="feedback-error" role="alert">
          反馈未保存，请重试。
        </span>
      )}
    </div>
  );
}
