import { useState } from "react";

import { saveFeedback } from "./api/client";

type FeedbackButtonsProps = {
  messageId: string;
  initialHelpful?: boolean | null;
};

export default function FeedbackButtons({ messageId, initialHelpful = null }: FeedbackButtonsProps) {
  const [helpful, setHelpful] = useState<boolean | null>(initialHelpful);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [failed, setFailed] = useState(false);

  const submit = async (value: boolean) => {
    setSaving(true);
    setFailed(false);
    setSaved(false);
    try {
      await saveFeedback(messageId, value);
      setHelpful(value);
      setSaved(true);
    } catch {
      setFailed(true);
    } finally {
      setSaving(false);
    }
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
          onClick={() => void submit(true)}
        >
          有用
        </button>
        <button
          type="button"
          className={helpful === false ? "selected" : ""}
          aria-pressed={helpful === false}
          disabled={saving}
          onClick={() => void submit(false)}
        >
          无用
        </button>
      </div>
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
