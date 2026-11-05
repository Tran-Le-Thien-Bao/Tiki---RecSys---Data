"""Lấy danh sách sản phẩm theo danh mục."""
from __future__ import annotations

import logging
import re
from typing import Any, Iterator

from crawler.tiki_client import TikiClient

log = logging.getLogger(__name__)

PRODUCT_LIMIT = 40
PRODUCT_FIELDS = ["id", "name", "brand", "price", "rating_average", "review_count", "category"]

_CAT_RE = re.compile(r"/c(\d+)(?:[/?#]|$)")


def category_id(value: str | int) -> int:
    """Lấy ID danh mục từ URL dạng tiki.vn/<slug>/c1789 hoặc từ số."""
    if isinstance(value, int):
        return value
    value = value.strip()
    if value.isdigit():
        return int(value)
    m = _CAT_RE.search(value)
    if not m:
        raise ValueError(f"Không tìm thấy ID danh mục trong: {value}")
    return int(m.group(1))


def parse_products(payload: dict[str, Any], category: int) -> list[dict[str, Any]]:
    """Chuyển một trang JSON sản phẩm thành các dòng theo schema products.csv."""
    rows = []
    for p in payload.get("data") or []:
        if p.get("id") is None:
            continue
        rows.append({
            "id": int(p["id"]),
            "name": p.get("name") or "",
            "brand": p.get("brand_name") or "",
            "price": p.get("price"),
            "rating_average": p.get("rating_average"),
            "review_count": int(p.get("review_count") or 0),
            "category": category,
        })
    return rows


def last_page(payload: dict[str, Any]) -> int:
    """Số trang cuối trong trường paging (mặc định 1)."""
    return int((payload.get("paging") or {}).get("last_page") or 1)


def fetch_product_page(client: TikiClient, category: int, page: int) -> dict[str, Any]:
    """Gọi API một trang sản phẩm."""
    return client.get_json("/api/v2/products", {"category": category, "limit": PRODUCT_LIMIT, "page": page})


def list_products(
    client: TikiClient,
    category: int,
    max_pages: int | None = None,
    min_reviews: int = 0,
) -> Iterator[dict[str, Any]]:
    """Duyệt từng trang sản phẩm của danh mục, chỉ trả sản phẩm có ≥ min_reviews review."""
    page = 1
    while True:
        payload = fetch_product_page(client, category, page)
        rows = parse_products(payload, category)
        log.info("Danh mục %s trang %d: %d sản phẩm", category, page, len(rows))
        for row in rows:
            if row["review_count"] >= min_reviews:
                yield row
        stop = last_page(payload)
        if max_pages is not None:
            stop = min(stop, max_pages)
        if page >= stop or not rows:
            return
        page += 1
