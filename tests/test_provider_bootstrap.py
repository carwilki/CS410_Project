"""Provider client (config-driven, OpenAI-compatible) and bootstrap marker chaining."""
import numpy as np

from cfpb_hyde import bootstrap, provider


class Resp:
    def __init__(self, j):
        self._j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


def test_chat_merges_chat_extra_and_auth(monkeypatch):
    sent = {}
    monkeypatch.setitem(provider.P, "chat_extra", {"reasoning_effort": "none"})
    monkeypatch.setitem(provider.P, "api_key_env", "TEST_PROVIDER_KEY")
    monkeypatch.setitem(provider.P, "base_url", "http://example.invalid/v1")
    monkeypatch.setenv("TEST_PROVIDER_KEY", "abc")

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, body=json, headers=headers)
        return Resp({"choices": []})

    monkeypatch.setattr(provider.requests, "post", fake_post)
    provider.chat([{"role": "user", "content": "hi"}], model="m", temperature=0.5)
    assert sent["url"] == "http://example.invalid/v1/chat/completions"
    assert sent["body"]["reasoning_effort"] == "none" and sent["body"]["temperature"] == 0.5
    assert sent["headers"] == {"Authorization": "Bearer abc"}


def test_no_auth_header_without_key(monkeypatch):
    monkeypatch.setitem(provider.P, "api_key_env", None)
    assert provider.headers() == {}


def test_embed_normalizes_only_non_unit(monkeypatch):
    out = {"data": [{"index": 1, "embedding": [0.0, 3.0]}, {"index": 0, "embedding": [3.0, 4.0]}]}
    monkeypatch.setattr(provider, "post", lambda ep, body: Resp(out))
    v = provider.embed(["a", "b"])
    np.testing.assert_allclose(v, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)
    unit = {"data": [{"index": 0, "embedding": [0.6, 0.8]}]}
    monkeypatch.setattr(provider, "post", lambda ep, body: Resp(unit))
    assert provider.embed(["a"]).tolist() == [[np.float32(0.6), np.float32(0.8)]]  # untouched


def test_step_markers_chain_on_data_and_corpus(monkeypatch):
    fp = [["a.zip", 10]]
    monkeypatch.setattr(bootstrap, "data_fingerprint", lambda: fp)
    s = {"corpus": {"data": fp, "corpus_sha": "x"}, "bm25": {"corpus_sha": "x"}, "embed": {"corpus_sha": "old"}}
    assert bootstrap.step_status(s) == {"corpus": True, "bm25": True, "embed": False, "faiss": False}
    monkeypatch.setattr(bootstrap, "data_fingerprint", lambda: [["a.zip", 11]])  # data changed
    assert not any(bootstrap.step_status(s).values())
    assert not any(bootstrap.step_status({}).values())


def test_adoption_only_for_legacy_or_same_corpus():
    s = {"corpus": {"corpus_sha": "new"}, "bm25": {"corpus_sha": "old"}}
    assert not bootstrap.adoptable(s, "bm25")      # built from another corpus: rebuild
    assert bootstrap.adoptable(s, "faiss")         # no marker: legacy build, verify and adopt
    s["bm25"]["corpus_sha"] = "new"
    assert bootstrap.adoptable(s, "bm25")
