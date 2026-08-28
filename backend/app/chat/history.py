"""Bounded reconstruction of complete conversation turns."""

from __future__ import annotations

from collections.abc import Sequence

from app.chat.models import ChatMessage, ChatTurn, HistoryContext, StoredMessage

MAX_HISTORY_CHARS = 6000


class HistoryContextBuilder:
    """Select recent complete turns without splitting a question/answer pair."""

    def __init__(self, max_turns: int = 5, max_chars: int = 6000) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if not 1 <= max_chars <= MAX_HISTORY_CHARS:
            raise ValueError("max_chars must be between 1 and 6000")
        self.max_turns = max_turns
        self.max_chars = max_chars

    def build(self, messages: Sequence[ChatMessage]) -> HistoryContext:
        """Return newest-first complete turns from a chronological message list."""

        turns: list[ChatTurn] = []
        total_chars = 0
        # Rows are ordered by created_at/rowid by the repository.  Walking
        # backwards makes the newest complete turn win when a partial turn is
        # present at either end of a session.
        for index in range(len(messages) - 1, 0, -1):
            assistant = messages[index]
            user = messages[index - 1]
            if assistant.role != "assistant" or assistant.status not in {
                "answered",
                "refused",
            }:
                continue
            if user.role != "user":
                continue
            if (
                assistant.reply_to_message_id is not None
                and assistant.reply_to_message_id != user.message_id
            ):
                continue
            # A submitted user row followed by an error response is not a
            # complete turn; terminal statuses above already exclude it.
            pair_chars = len(user.content) + len(assistant.content)
            if total_chars + pair_chars > self.max_chars:
                break
            turns.append(
                ChatTurn(
                    user_message_id=user.message_id,
                    user_question=user.content,
                    rewritten_question=user.rewritten_question,
                    assistant_message_id=assistant.message_id,
                    answer=assistant.content,
                    status=assistant.status,
                    citations=assistant.citations,
                )
            )
            total_chars += pair_chars
            if len(turns) >= self.max_turns:
                break
        return HistoryContext(turns=tuple(turns))


__all__ = ["MAX_HISTORY_CHARS", "HistoryContextBuilder", "StoredMessage"]
