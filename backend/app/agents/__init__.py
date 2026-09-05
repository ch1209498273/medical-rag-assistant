"""Small, bounded role-workflow primitives for the v2.2a experiment.

The package deliberately contains orchestration contracts only.  It does not
create provider clients, read credentials, or expose a general-purpose tool
executor.
"""

from app.agents.budget import (
    BudgetConfigurationError,
    BudgetTransport,
    RequestBudget,
    UsageRecord,
    WorkflowBudgetExceeded,
    current_budget,
    current_stage,
    request_scope,
    stage_scope,
)
from app.agents.contracts import (
    BufferedAnswer,
    RouteDecision,
    WorkflowSummary,
    collect_answer,
)
from app.agents.router import (
    ROUTER_SYSTEM_PROMPT,
    DeepSeekRouter,
    DeterministicRouter,
    RouterUnavailable,
)
from app.agents.workflow import (
    AgentWorkflow,
    DeterministicVerifier,
    RetrievalUnavailableError,
)

__all__ = [
    "ROUTER_SYSTEM_PROMPT",
    "AgentWorkflow",
    "BudgetConfigurationError",
    "BudgetTransport",
    "BufferedAnswer",
    "DeepSeekRouter",
    "DeterministicRouter",
    "DeterministicVerifier",
    "RequestBudget",
    "RetrievalUnavailableError",
    "RouteDecision",
    "RouterUnavailable",
    "UsageRecord",
    "WorkflowBudgetExceeded",
    "WorkflowSummary",
    "collect_answer",
    "current_budget",
    "current_stage",
    "request_scope",
    "stage_scope",
]
