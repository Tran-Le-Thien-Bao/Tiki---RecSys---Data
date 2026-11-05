import json
from pathlib import Path

from crawler.privacy import pseudo
from crawler.reviews import REVIEW_FIELDS, clean, crawl_reviews

KEY = b"k" * 32


def raw_review(**over):
    r = {
        "id": 1, "product_id": 10, "customer_id": 999, "rating": 5, "created_at": 1700000000,
        "title": "Tốt", "content": "Liên hệ 0912345678 hoặc x@y.com",
        "thank_count": 3,
        "created_by": {
            "id": 999, "name": "Nguyễn Văn A", "full_name": "Nguyễn Văn A", "avatar_url": "http://a",
            "region": "Hà Nội", "purchased": True, "purchased_at": 1690000000,
            "contribute_info": {"summary": {"total_review": 42}},
        },
        "images": [{"full_path": "http://img"}],
    }
    r.update(over)
    return r


def test_clean_schema_and_privacy():
    rec = clean(raw_review(), KEY)
    assert list(rec) == REVIEW_FIELDS
    assert rec["user"] == pseudo(999, KEY)
    assert rec["purchased"] is True
    assert rec["user_total_review"] == 42
    assert rec["content"] == "Liên hệ [sdt] hoặc [email]"
    dumped = json.dumps(rec, ensure_ascii=False)
    for leaked in ("999", "Nguyễn", "Hà Nội", "http", "1690000000"):
        assert leaked not in dumped


def test_clean_rejects_invalid():
    assert clean(raw_review(rating=0), KEY) is None
    assert clean(raw_review(rating=6), KEY) is None
    assert clean(raw_review(customer_id=None), KEY) is None
    r = raw_review()
    del r["id"]
    assert clean(r, KEY) is None


def test_clean_missing_optional_fields():
    rec = clean(raw_review(created_by=None, title=None, thank_count=None), KEY)
    assert rec["purchased"] is False and rec["user_total_review"] == 0
    assert rec["title"] == "" and rec["thank_count"] == 0


class FakeClient:
    """Trả trang review giả, không gọi mạng."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get_json(self, path, params):
        self.calls.append(params["page"])
        return {"data": self.pages[params["page"] - 1], "paging": {"last_page": len(self.pages)}}


def test_crawl_reviews_appends_and_dedups(tmp_path: Path):
    out = tmp_path / "reviews.jsonl"
    client = FakeClient([[raw_review(id=1), raw_review(id=2)], [raw_review(id=2), raw_review(id=3)]])
    seen: set = set()
    assert crawl_reviews(client, 10, KEY, out, seen) == 3
    assert client.calls == [1, 2]
    ids = [json.loads(l)["review_id"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert ids == [1, 2, 3]
    # Chạy lại không ghi trùng.
    assert crawl_reviews(FakeClient([[raw_review(id=1)]]), 10, KEY, out, seen) == 0
