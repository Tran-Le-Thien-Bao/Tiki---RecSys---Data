"""Cào và làm sạch review của một sản phẩm."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from crawler.privacy import pseudo, scrub
from crawler.products import last_page
from crawler.tiki_client import TikiClient

log = logging.getLogger(__name__)

REVIEW_LIMIT = 20
REVIEW_FIELDS = [
    "review_id", "product_id", "user", "rating", "created_at",
    "title", "content", "thank_count", "purchased", "user_total_review",
]


def clean(raw: dict[str, Any], key: bytes) -> dict[str, Any] | None:
    """Chuyển một review thô thành bản ghi theo schema; bỏ mọi thông tin cá nhân. Trả None nếu không hợp lệ."""
    try:
        review_id = int(raw["id"])
        product_id = int(raw["product_id"])
        rating = int(raw["rating"])
    except (KeyError, TypeError, ValueError):
        return None
    customer_id = raw.get("customer_id")
    if customer_id in (None, "", 0) or not 1 <= rating <= 5:
        return None

    created_by = raw.get("created_by") or {}
    summary = ((created_by.get("contribute_info") or {}).get("summary") or {})
    return {
        "review_id": review_id,
        "product_id": product_id,
        "user": pseudo(customer_id, key),
        "rating": rating,
        "created_at": int(raw.get("created_at") or 0),
        "title": scrub(raw.get("title")),
        "content": scrub(raw.get("content")),
        "thank_count": int(raw.get("thank_count") or 0),
        "purchased": bool(created_by.get("purchased")),
        "user_total_review": int(summary.get("total_review") or 0),
    }


def fetch_review_page(client: TikiClient, product_id: int, page: int) -> dict[str, Any]:
    """Gọi API một trang review."""
    return client.get_json("/api/v2/reviews", {
        "product_id": product_id, "limit": REVIEW_LIMIT, "page": page, "include": "contribute_info",
    })


def load_seen_ids(path: Path) -> set[int]:
    """Đọc review_id đã có trong file JSONL để lọc trùng khi chạy lại."""
    seen: set[int] = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(int(json.loads(line)["review_id"]))
    return seen


def crawl_reviews(
    client: TikiClient,
    product_id: int,
    key: bytes,
    out_path: Path,
    seen_ids: set[int],
    max_pages: int | None = None,
) -> int:
    """Cào toàn bộ review của sản phẩm, ghi nối từng trang vào JSONL. Trả số review mới."""
    written = 0
    page = 1
    while True:
        payload = fetch_review_page(client, product_id, page)
        items = payload.get("data") or []
        records = []
        for raw in items:
            rec = clean(raw, key)
            if rec and rec["review_id"] not in seen_ids:
                seen_ids.add(rec["review_id"])
                records.append(rec)
        if records:
            with out_path.open("a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += len(records)

        stop = last_page(payload)
        if max_pages is not None:
            stop = min(stop, max_pages)
        if page >= stop or not items:
            return written
        page += 1
