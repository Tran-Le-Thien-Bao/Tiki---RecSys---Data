import pytest

from crawler.tiki_client import MIN_DELAY, CrawlBlocked, TikiClient


class FakeResp:
    def __init__(self, status=200, body=None, ctype="application/json", headers=None):
        self.status_code = status
        self._body = body if body is not None else {"data": []}
        self.headers = {"Content-Type": ctype, **(headers or {})}

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def make(responses):
    sleeps = []
    sess = FakeSession(responses)
    client = TikiClient("sv@gm.uit.edu.vn", session=sess, sleep=sleeps.append, clock=lambda: 0.0)
    return client, sess, sleeps


def test_user_agent_has_contact_and_delay_floor():
    client, sess, _ = make([])
    assert "sv@gm.uit.edu.vn" in sess.headers["User-Agent"]
    assert "research" in sess.headers["User-Agent"]
    assert TikiClient("a@b.c", delay=0.1, session=FakeSession([])).delay == MIN_DELAY
    with pytest.raises(ValueError):
        TikiClient("", session=FakeSession([]))


def test_throttle_between_requests():
    client, _, sleeps = make([FakeResp(), FakeResp()])
    client.get_json("/api/v2/products", {})
    client.get_json("/api/v2/products", {})
    assert len(sleeps) == 1 and MIN_DELAY <= sleeps[0] <= MIN_DELAY + 1.0


def test_disallowed_path():
    client, sess, _ = make([])
    with pytest.raises(ValueError):
        client.get_json("/api/v2/me/", {})
    with pytest.raises(ValueError):
        client.get_json("/api/v2/reviews/writable", {})
    assert sess.calls == 0


@pytest.mark.parametrize("resp", [FakeResp(403), FakeResp(401), FakeResp(200, "<html>", "text/html")])
def test_blocked_stops_immediately(resp):
    client, sess, _ = make([resp, FakeResp()])
    with pytest.raises(CrawlBlocked):
        client.get_json("/api/v2/products", {})
    assert sess.calls == 1


def test_429_retry_after_then_backoff():
    client, sess, sleeps = make([
        FakeResp(429, headers={"Retry-After": "7"}), FakeResp(503), FakeResp(200, {"ok": 1}),
    ])
    assert client.get_json("/api/v2/reviews", {}) == {"ok": 1}
    backoffs = [s for s in sleeps if s > MIN_DELAY + 1.0]
    assert backoffs == [7.0, 10.0]  # Retry-After, rồi 5 * 2**1


def test_gives_up_after_max_retries():
    client, sess, _ = make([FakeResp(500)] * 5)
    with pytest.raises(RuntimeError):
        client.get_json("/api/v2/reviews", {})
    assert sess.calls == 5
