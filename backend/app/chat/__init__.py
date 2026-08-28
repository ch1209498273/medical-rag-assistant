"""Stateful conversational backend components."""

__all__ = [
    "AssistantMessageStatus",
    "ChatMessage",
    "ChatOrchestrator",
    "ChatTurn",
    "Feedback",
    "FollowUpRewriter",
    "HistoryContext",
    "HistoryContextBuilder",
    "RewriteError",
    "StoredMessage",
]


def __getattr__(name: str):
    if name in {"AssistantMessageStatus", "ChatMessage", "ChatTurn", "Feedback", "HistoryContext", "StoredMessage"}:
        from app.chat import models

        return getattr(models, name)
    if name == "HistoryContextBuilder":
        from app.chat.history import HistoryContextBuilder

        return HistoryContextBuilder
    if name in {"FollowUpRewriter", "RewriteError"}:
        from app.chat.rewriter import FollowUpRewriter, RewriteError

        return {"FollowUpRewriter": FollowUpRewriter, "RewriteError": RewriteError}[name]
    if name == "ChatOrchestrator":
        from app.chat.orchestrator import ChatOrchestrator

        return ChatOrchestrator
    raise AttributeError(name)
