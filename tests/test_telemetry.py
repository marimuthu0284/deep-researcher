import pytest

from deep_researcher.telemetry import summarize_metrics, timed


@pytest.mark.asyncio
async def test_timed_appends_metrics_entry():
    @timed("node_x")
    async def node(state):
        return {"foo": 1}

    out = await node({})
    assert out["foo"] == 1
    assert out["metrics"][0]["node"] == "node_x"
    assert "seconds" in out["metrics"][0]


@pytest.mark.asyncio
async def test_timed_preserves_existing_metrics():
    @timed("node_y")
    async def node(state):
        return {"metrics": [{"node": "prior", "seconds": 0.1}]}

    out = await node({})
    nodes = [m["node"] for m in out["metrics"]]
    assert nodes == ["prior", "node_y"]


def test_summarize_metrics_aggregates():
    metrics = [
        {"node": "a", "seconds": 1.0},
        {"node": "a", "seconds": 2.0},
        {"node": "b", "seconds": 0.5},
    ]
    s = summarize_metrics(metrics)
    assert s["total_seconds"] == 3.5
    assert s["per_node"]["a"]["calls"] == 2
    assert s["per_node"]["a"]["total_seconds"] == 3.0
    assert s["per_node"]["b"]["calls"] == 1
