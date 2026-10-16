"""Explicit unlimited delegation budgets reach the child constructor as integers."""
import sys
from unittest.mock import patch

import pytest

from tests.tools.test_delegate import _make_mock_parent
from tools import delegate_tool as dt


@pytest.mark.parametrize("limit", [None, 0, "unlimited", 401])
def test_configured_budget_is_normalized_at_child_boundary(limit):
    parent = _make_mock_parent()
    with (
        patch.object(dt, "_load_config", return_value={"max_iterations": limit}),
        patch.object(dt, "_resolve_delegation_credentials", return_value={"model": "fixture"}),
        patch.object(dt, "_build_children", return_value=([], "fixture: stop before spawn")) as build,
        patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])),
    ):
        dt.delegate_task(goal="isolated budget fixture", parent_agent=parent)
    assert build.call_count == 1
    assert build.call_args.kwargs["max_iterations"] == (401 if limit == 401 else sys.maxsize)
