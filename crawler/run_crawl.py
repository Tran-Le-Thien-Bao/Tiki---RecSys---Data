"""CLI cào review Tiki.

Ví dụ:
    python -m crawler.run_crawl --smoke
    python -m crawler.run_crawl --categories https://tiki.vn/dien-thoai-may-tinh-bang/c1789 \
        --max-product-pages 3 --min-reviews 5 --limit-products 100
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from crawler.privacy import load_or_create_key
from crawler.products import PRODUCT_FIELDS, category_id, fetch_product_page, list_products, parse_products
from crawler.reviews import clean, crawl_reviews, fetch_review_page, load_seen_ids
from crawler.tiki_client import MIN_DELAY, CrawlBlocked, TikiClient

log = logging.getLogger("crawler")

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_CATEGORY = 1789

# Tên trường mong đợi — dùng cho --smoke.
EXPECTED_PRODUCT_FIELDS = ["id", "name", "brand_name", "price", "rating_average", "review_count", "seller_id"]
EXPECTED_REVIEW_FIELDS = ["id", "product_id", "customer_id", "rating", "created_at", "title", "content",
                          "thank_count", "created_by"]


def setup_logging(raw_dir: Path) -> None:
    """Ghi log ra console và data/raw/crawl.log."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(raw_dir / "crawl.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    p = argparse.ArgumentParser(description="Cào review Tiki cho nghiên cứu hệ khuyến nghị")
    p.add_argument("--categories", nargs="+", default=[], help="URL hoặc ID danh mục (vd .../c1789)")
    p.add_argument("--max-product-pages", type=int, default=None, help="Số trang sản phẩm tối đa mỗi danh mục")
    p.add_argument("--max-review-pages", type=int, default=None, help="Số trang review tối đa mỗi sản phẩm")
    p.add_argument("--min-reviews", type=int, default=1, help="Bỏ qua sản phẩm có ít review hơn")
    p.add_argument("--limit-products", type=int, default=None, help="Số sản phẩm tối đa mỗi danh mục")
    p.add_argument("--data-dir", default=None, help="Mặc định lấy từ DATA_DIR")
    p.add_argument("--delay", type=float, default=None, help=f"Độ trễ (giây), tối thiểu {MIN_DELAY}")
    p.add_argument("--smoke", action="store_true", help="Kiểm tra nhanh: 1 danh mục, 1 sản phẩm, 1 trang review")
    return p.parse_args(argv)


def read_lines(path: Path) -> set[str]:
    """Đọc các dòng không rỗng của file (nếu có)."""
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def read_product_ids(path: Path) -> set[int]:
    """ID sản phẩm đã có trong products.csv."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {int(r["id"]) for r in csv.DictReader(f)}


def append_product(path: Path, row: dict) -> None:
    """Ghi nối một sản phẩm vào products.csv."""
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRODUCT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def check_fields(kind: str, obj: dict, expected: list[str]) -> None:
    """Cảnh báo nếu API thiếu trường mong đợi."""
    missing = [k for k in expected if k not in obj]
    if missing:
        log.warning("%s thiếu trường: %s", kind, missing)
    else:
        log.info("%s đủ trường mong đợi", kind)


def run_smoke(client: TikiClient, key: bytes, categories: list[int]) -> None:
    """Gọi 1 danh mục, 1 sản phẩm, 1 trang review và in kết quả đã làm sạch."""
    cat = categories[0] if categories else SMOKE_CATEGORY
    payload = fetch_product_page(client, cat, 1)
    raw_products = payload.get("data") or []
    if not raw_products:
        log.error("Danh mục %s không trả sản phẩm nào", cat)
        sys.exit(1)
    raw_p = next((p for p in raw_products if (p.get("review_count") or 0) > 0), raw_products[0])
    check_fields("Sản phẩm", raw_p, EXPECTED_PRODUCT_FIELDS)
    product = parse_products({"data": [raw_p]}, cat)[0]
    log.info("Sản phẩm: %s", json.dumps(product, ensure_ascii=False))

    rpayload = fetch_review_page(client, product["id"], 1)
    items = rpayload.get("data") or []
    log.info("Trang review 1/%s: %d review", (rpayload.get("paging") or {}).get("last_page"), len(items))
    if items:
        check_fields("Review", items[0], EXPECTED_REVIEW_FIELDS)
    cleaned = [r for r in (clean(it, key) for it in items) if r]
    log.info("Làm sạch được %d/%d review", len(cleaned), len(items))
    for rec in cleaned[:3]:
        rec = {**rec, "content": rec["content"][:120]}
        log.info("Review: %s", json.dumps(rec, ensure_ascii=False))
    log.info("SMOKE OK")


def run_crawl(client: TikiClient, key: bytes, args: argparse.Namespace, categories: list[int], raw_dir: Path) -> None:
    """Cào toàn bộ danh mục; bỏ qua sản phẩm đã có trong products.done."""
    reviews_path = raw_dir / "reviews.jsonl"
    products_path = raw_dir / "products.csv"
    done_path = raw_dir / "products.done"

    done = read_lines(done_path)
    seen_ids = load_seen_ids(reviews_path)
    known_products = read_product_ids(products_path)
    log.info("Đã có %d sản phẩm xong, %d review", len(done), len(seen_ids))

    for cat in categories:
        count = 0
        for product in list_products(client, cat, args.max_product_pages, args.min_reviews):
            if args.limit_products is not None and count >= args.limit_products:
                break
            count += 1
            pid = product["id"]
            if pid not in known_products:
                append_product(products_path, product)
                known_products.add(pid)
            if str(pid) in done:
                continue
            n = crawl_reviews(client, pid, key, reviews_path, seen_ids, args.max_review_pages)
            with done_path.open("a", encoding="utf-8") as f:
                f.write(f"{pid}\n")
            done.add(str(pid))
            log.info("Danh mục %s, sản phẩm %d (%d/%s): +%d review, tổng %d",
                     cat, pid, count, args.limit_products or "∞", n, len(seen_ids))
        log.info("Xong danh mục %s: %d sản phẩm", cat, count)


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI."""
    load_dotenv()
    args = parse_args(argv)

    data_dir = Path(args.data_dir or os.getenv("DATA_DIR", "./data")).expanduser().resolve()
    raw_dir = data_dir / "raw"
    setup_logging(raw_dir)

    contact = os.getenv("CRAWLER_CONTACT", "")
    key_path = os.getenv("HMAC_KEY_PATH")
    if not key_path:
        log.error("Chưa đặt HMAC_KEY_PATH trong .env")
        return 1
    delay = args.delay if args.delay is not None else float(os.getenv("CRAWL_DELAY", MIN_DELAY))

    try:
        key = load_or_create_key(key_path, forbidden_dirs=[REPO_ROOT, data_dir])
        client = TikiClient(contact, delay=delay)
        categories = [category_id(c) for c in args.categories]
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    try:
        if args.smoke:
            run_smoke(client, key, categories)
        elif not categories:
            log.error("Cần --categories hoặc --smoke")
            return 1
        else:
            run_crawl(client, key, args, categories, raw_dir)
    except CrawlBlocked as exc:
        log.error("BỊ CHẶN: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("Đã dừng bởi người dùng; chạy lại để tiếp tục")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
