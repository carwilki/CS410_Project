import pytest

from cfpb_hyde.admin import AdminStore
from cfpb_hyde.search import Hit


def test_store_roundtrip(tmp_path):
    s = AdminStore(tmp_path / "a.sqlite")
    hits = [Hit(1, "100", 0.9), Hit(2, "200", 0.5)]
    s.log_search("q1", "overdraft fees", "hybrid", 2, {"fusion_alpha": 0.1}, hits)
    s.rate("q1", "100", "human", 3)
    s.rate("q1", "100", "llm:judge", 2, {"result_status": 1})
    s.rate("q1", "100", "human", 1)  # re-rating replaces
    assert s.ratings("q1") == {"100": {"human": 1, "llm:judge": 2}}
    rev = s.review()
    assert rev[0]["query"] == "overdraft fees" and rev[0]["hits"] == [(1, "100", 0.9), (2, "200", 0.5)]
    assert rev[0]["settings"] == {"fusion_alpha": 0.1}
    with pytest.raises(ValueError):
        s.rate("q1", "100", "human", 4)
