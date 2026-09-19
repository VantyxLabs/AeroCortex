from demo import run_gps_reuse_demo


def test_gps_reuse_demo_writes_and_reinforces():
    demo = run_gps_reuse_demo()
    m1 = demo["mission1"]
    m2 = demo["mission2"]

    assert m1["episode_id"]
    assert m1["action"]
    assert m1["chroma_count"] >= 1
    assert isinstance(m1["latency_ms"], dict)

    assert m2["top"] is not None
    top = m2["top"]
    assert 0.0 <= top.vector_similarity <= 1.0
    assert 0.0 <= top.graph_relevance <= 1.0
    assert 0.0 <= top.final_score <= 1.0

    h1 = m1["edge"].get("hits") or 0
    h2 = m2["edge"].get("hits") or 0
    w1 = m1["edge"].get("weight") or 0.0
    w2 = m2["edge"].get("weight") or 0.0
    assert h2 > h1
    assert w2 >= w1
    assert m2["latency_ms"] is not None
