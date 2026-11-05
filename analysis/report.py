"""Thống kê độ thưa và k-core cho reviews.jsonl.

    python -m analysis.report data/raw/reviews.jsonl
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def k_core(df: pd.DataFrame, k: int, user_col: str = "user", item_col: str = "product_id") -> pd.DataFrame:
    """Lọc lặp đến khi mọi user và item đều có ≥ k tương tác (mỗi cặp user–item tính một lần)."""
    df = df.drop_duplicates([user_col, item_col])
    while True:
        u = df[user_col].map(df[user_col].value_counts())
        i = df[item_col].map(df[item_col].value_counts())
        mask = (u >= k) & (i >= k)
        if mask.all():
            return df
        df = df[mask]


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Số liệu tổng hợp: số review, user, item, độ thưa, phân bố số sản phẩm/user."""
    df = df.drop_duplicates("review_id")
    pairs = df.drop_duplicates(["user", "product_id"])
    n_users, n_items = pairs["user"].nunique(), pairs["product_id"].nunique()
    per_user = pairs.groupby("user").size()
    out: dict[str, float] = {
        "reviews": len(df),
        "users": n_users,
        "items": n_items,
        "user_item_pairs": len(pairs),
        "density": len(pairs) / (n_users * n_items) if n_users and n_items else 0.0,
    }
    for t in (2, 3, 5, 10):
        out[f"users_>={t}_items"] = int((per_user >= t).sum())
    for k in (5,):
        core = k_core(pairs, k)
        out[f"{k}core_pairs"] = len(core)
        out[f"{k}core_users"] = core["user"].nunique()
        out[f"{k}core_items"] = core["product_id"].nunique()
    return out


def main(argv: list[str] | None = None) -> int:
    """In báo cáo ra màn hình."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Đường dẫn reviews.jsonl")
    args = p.parse_args(argv)
    df = pd.read_json(args.path, lines=True)
    if df.empty:
        print("File rỗng")
        return 1
    for name, value in summarize(df).items():
        print(f"{name:<20} {value:.6f}" if isinstance(value, float) else f"{name:<20} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
