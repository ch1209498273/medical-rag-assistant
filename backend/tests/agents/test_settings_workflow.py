from __future__ import annotations

import pytest
from app.settings import Settings


def test_answer_workflow_defaults_to_baseline_v1():
    assert Settings(_env_file=None).answer_workflow == "baseline_v1"


def test_answer_workflow_is_an_explicit_literal_switch():
    assert Settings(_env_file=None, answer_workflow="agent_workflow_v2a").answer_workflow == "agent_workflow_v2a"
    with pytest.raises(ValueError):
        Settings(_env_file=None, answer_workflow="anything_else")
