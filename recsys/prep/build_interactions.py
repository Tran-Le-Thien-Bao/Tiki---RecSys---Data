"""Prep: đọc dữ liệu thô → lọc trùng, scrub → tương tác nhị phân → các bản chia (docs/DAC_TA.md mục 5.1).

Chạy:  python -m recsys.prep.build_interactions --star all --k 0 2
Chỉ log số liệu tổng hợp; không in nội dung review hay user_id.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from crawler.privacy import EMAIL_RE, PHONE_RE, scrub
from recsys import config
from recsys.prep.kcore import kcore
from recsys.prep.split import make_split, time_cutoffs, write_split
from recsys.schema import validate_interactions

log = logging.getLogger(__name__)

REVIEW_COLUMNS = ["review_id", "user_id", "product_id", "rating", "title", "content",
                  "thank_count", "created_at", "is_verified_purchase"]


class PrepCheckError(RuntimeError):
    """Số liệu sau xử lý lệch với số đã kiểm (config.EXPECTED*)."""


# ---------------------------------------------------------------- đọc
def read_jsonl(path: Path) -> pd.DataFrame:
    """Đọc JSON Lines, không tự đổi kiểu (created_at giữ nguyên unix giây)."""
    return pd.read_json(path, lines=True, convert_dates=False, dtype=False)


def clean_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    """Lọc trùng theo review_id; ghi cờ had_phone/had_email TRƯỚC khi scrub (cho flags.promo); scrub title/content."""
    missing = [c for c in REVIEW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"reviews thiếu cột {missing}")
    df = raw[REVIEW_COLUMNS].drop_duplicates("review_id", keep="first").reset_index(drop=True)
    if df[["user_id", "product_id", "rating", "created_at"]].isna().any().any():
        raise ValueError("reviews có giá trị thiếu ở user_id/product_id/rating/created_at")
    title = df["title"].fillna("").astype(str)
    content = df["content"].fillna("").astype(str)
    text = title + "\n" + content
    df["had_phone"] = text.str.contains(PHONE_RE)
    df["had_email"] = text.str.contains(EMAIL_RE)
    df["title"] = title.map(scrub)
    df["content"] = content.map(scrub)
    return df.astype({"review_id": "int64", "user_id": "int64", "product_id": "int64",
                      "rating": "int8", "created_at": "int64", "thank_count": "int64"})


def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
    """Lọc trùng theo product_id; specifications (dict) → chuỗi JSON để ghi parquet."""
    df = raw.drop_duplicates("product_id", keep="first").reset_index(drop=True)
    if "specifications" in df.columns:
        df["specifications"] = df["specifications"].map(
            lambda v: json.dumps(v, ensure_ascii=False) if v is not None and not isinstance(v, str) else v)
    return df.astype({"product_id": "int64"})


def to_interactions(reviews: pd.DataFrame) -> pd.DataFrame:
    """Mỗi cặp (user, product) → 1 dòng: ts/rating của review sớm nhất (hòa: review_id nhỏ hơn), n_reviews."""
    key = ["user_id", "product_id"]
    ordered = reviews.sort_values([*key, "created_at", "review_id"], kind="mergesort")
    first = ordered.drop_duplicates(key, keep="first")[[*key, "created_at", "rating"]]
    n = ordered.groupby(key).size().rename("n_reviews").reset_index()
    out = first.merge(n, on=key, how="left", validate="one_to_one").rename(columns={"created_at": "ts"})
    out = out.astype({"user_id": "int64", "product_id": "int64", "ts": "int64", "rating": "int8", "n_reviews": "int16"})
    out["is_fake"] = False
    out = out.sort_values(["ts", "user_id", "product_id"], kind="mergesort").reset_index(drop=True)
    validate_interactions(out)
    return out


# ---------------------------------------------------------------- đối chiếu
def compare_expected(actual: dict, expected: dict) -> list[str]:
    """Danh sách mô tả các khóa lệch (rỗng = khớp hết). Khóa không có trong actual cũng tính là lệch."""
    return [f"{k}: thực tế {actual.get(k)!r} ≠ kỳ vọng {v!r}" for k, v in expected.items() if actual.get(k) != v]


def raw_counts(raw_reviews: pd.DataFrame, reviews: pd.DataFrame, raw_products: pd.DataFrame,
               products: pd.DataFrame, inter: pd.DataFrame) -> dict:
    """Các con số để đối chiếu với config.EXPECTED."""
    return {
        "reviews_rows": int(len(raw_reviews)),
        "reviews_unique": int(len(reviews)),
        "users": int(reviews["user_id"].nunique()),
        "pairs": int(len(inter)),
        "products_with_reviews": int(reviews["product_id"].nunique()),
        "products_rows": int(len(raw_products)),
        "products_unique": int(len(products)),
    }


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------- chạy
def run(stars: list[str], ks: list[int], write: bool = True) -> dict:
    """Chạy toàn bộ prep, trả summary (dict). Ném PrepCheckError nếu số liệu lệch (sau khi đã ghi summary)."""
    log.info("Đọc %s", config.RAW_REVIEWS.name)
    raw_reviews = read_jsonl(config.RAW_REVIEWS)
    raw_products = read_jsonl(config.RAW_PRODUCTS)
    reviews = clean_reviews(raw_reviews)
    products = clean_products(raw_products)
    inter = to_interactions(reviews)

    counts = raw_counts(raw_reviews, reviews, raw_products, products, inter)
    rating_counts = {int(k): int(v) for k, v in reviews["rating"].value_counts().sort_index().items()}
    kcore_all = {}
    for k in config.EXPECTED_KCORE_ALL_PAIRS:
        c = kcore(inter, k)
        kcore_all[k] = (int(c["user_id"].nunique()), int(c["product_id"].nunique()), int(len(c)))
    mismatches = (compare_expected(counts, config.EXPECTED)
                  + compare_expected(rating_counts, config.EXPECTED_RATING_COUNTS)
                  + compare_expected(kcore_all, config.EXPECTED_KCORE_ALL_PAIRS))

    t_val, t_test = time_cutoffs(inter)
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "counts": counts,
        "rating_counts": rating_counts,
        "kcore_all_pairs": {str(k): dict(zip(("users", "items", "pairs"), v)) for k, v in kcore_all.items()},
        "n_reviews_with_phone_before_scrub": int(reviews["had_phone"].sum()),
        "n_reviews_with_email_before_scrub": int(reviews["had_email"].sum()),
        "pairs_with_multiple_reviews": int((inter["n_reviews"] > 1).sum()),
        "cutoffs": {
            "q_val": config.VAL_QUANTILE, "q_test": config.TEST_QUANTILE,
            "t_val": t_val, "t_val_iso": _iso(t_val),
            "t_test": t_test, "t_test_iso": _iso(t_test),
            "ts_min_iso": _iso(int(inter["ts"].min())), "ts_max_iso": _iso(int(inter["ts"].max())),
        },
        "checks_ok": not mismatches,
        "check_mismatches": mismatches,
        "splits": {},
    }
    if config.PREP_SUMMARY_PATH.exists():   # giữ kết quả các biến thể đã chạy trước
        try:
            old = json.loads(config.PREP_SUMMARY_PATH.read_text(encoding="utf-8"))
            if old.get("cutoffs", {}).get("t_val") == t_val:
                summary["splits"].update(old.get("splits", {}))
        except (json.JSONDecodeError, OSError):
            pass

    if write:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        inter.to_parquet(config.INTERACTIONS_PATH, index=False)
        reviews.to_parquet(config.REVIEWS_CLEAN_PATH, index=False)
        products.to_parquet(config.PRODUCTS_CLEAN_PATH, index=False)

    for star in stars:
        for k in ks:
            sp = make_split(inter, star, k, t_val, t_test)
            summary["splits"][f"{star}_{k}core"] = sp.stats
            if write:
                write_split(sp)
            s = sp.stats
            log.info("%s_%dcore: train %d tt / %d user / %d sp | val %d user (warm %.1f%%) | test %d user (warm %.1f%%)",
                     star, k, s["train"]["n_interactions"], s["train"]["n_users"], s["train"]["n_items"],
                     s["val"]["n_users"], 100 * (s["val"]["warm_user_ratio"] or 0),
                     s["test"]["n_users"], 100 * (s["test"]["warm_user_ratio"] or 0))

    if write:
        config.PREP_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.PREP_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if mismatches:
        raise PrepCheckError("Số liệu lệch so với docs/DAC_TA.md mục 2:\n  " + "\n  ".join(mismatches))
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--star", nargs="+", default=["all"], choices=list(config.STAR_VARIANTS))
    ap.add_argument("--k", nargs="+", type=int, default=[0], choices=list(config.KCORE_LEVELS))
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):   # console Windows mặc định cp1252 → lỗi chữ Việt
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(args.star, args.k)
    except PrepCheckError as e:
        log.error("%s", e)
        return 1
    log.info("Xong. Tóm tắt: %s", config.PREP_SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
