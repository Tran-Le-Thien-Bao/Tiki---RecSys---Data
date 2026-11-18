"""Bộ gắn cờ: 3 dấu hiệu dup_content, promo, burst (docs/DO_AN.md mục 4.5.4).

Đầu vào: data/processed/reviews_clean.parquet (đã scrub, có cột had_phone tính trước khi scrub).
Đầu ra: cờ theo review, rồi gộp theo cặp (user, product): cặp bị gắn cờ nếu có ít nhất một review bị gắn cờ.
Chỉ log số đếm tổng hợp, không in nội dung.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from recsys import config

log = logging.getLogger(__name__)
_WS = re.compile(r"\s+")


def normalize_text(s: pd.Series) -> pd.Series:
    """lowercase, strip, gộp khoảng trắng."""
    return s.fillna("").astype(str).str.lower().str.strip().str.replace(_WS, " ", regex=True)


def flag_dup_content(reviews: pd.DataFrame, min_len: int = config.DUP_MIN_LEN,
                     min_count: int = config.DUP_MIN_COUNT) -> pd.Series:
    """Nội dung chuẩn hóa dài > min_len và xuất hiện ≥ min_count lần trên toàn bộ dữ liệu."""
    norm = normalize_text(reviews["content"])
    long_ = norm.str.len() > min_len
    counts = norm[long_].map(norm[long_].value_counts())
    out = pd.Series(False, index=reviews.index)
    out.loc[counts.index] = counts >= min_count
    return out


def flag_promo(reviews: pd.DataFrame, keywords=config.PROMO_KEYWORDS,
               use_phone: bool = config.PROMO_USE_PHONE) -> pd.Series:
    """Chứa từ khóa quảng cáo (trong tiêu đề hoặc nội dung) hoặc có SĐT trước khi scrub."""
    text = normalize_text(reviews["title"].fillna("") + " " + reviews["content"].fillna(""))
    pat = "|".join(re.escape(k.lower()) for k in keywords)
    out = text.str.contains(pat, regex=True)
    if use_phone and "had_phone" in reviews:
        out = out | reviews["had_phone"].astype(bool)
    return out


def flag_burst(reviews: pd.DataFrame, same_product_min: int = config.BURST_SAME_PRODUCT_MIN,
               n_products: int = config.BURST_N_PRODUCTS, window_min: int = config.BURST_WINDOW_MIN) -> pd.Series:
    """Cùng user ≥ same_product_min review cho cùng sản phẩm, HOẶC review ≥ n_products sản phẩm khác nhau
    trong một cửa sổ window_min phút (mọi review trong cửa sổ đó bị gắn cờ)."""
    pair_n = reviews.groupby(["user_id", "product_id"])["review_id"].transform("size")
    same = pair_n >= same_product_min

    burst = np.zeros(len(reviews), dtype=bool)
    win = window_min * 60
    df = reviews[["user_id", "product_id", "created_at"]].reset_index()
    df = df.sort_values(["user_id", "created_at"], kind="mergesort")
    big = df.groupby("user_id")["product_id"].transform("size") >= n_products
    for _, g in df[big].groupby("user_id", sort=False):
        t = g["created_at"].to_numpy()
        p = g["product_id"].to_numpy()
        idx = g["index"].to_numpy()
        j = 0
        for i in range(len(t)):
            while t[i] - t[j] > win:
                j += 1
            if len(set(p[j:i + 1])) >= n_products:
                burst[idx[j:i + 1]] = True
    return same | pd.Series(burst, index=reviews.index)


def flag_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Bảng cờ theo review: review_id, user_id, product_id + 3 cột bool."""
    reviews = reviews.reset_index(drop=True)
    out = reviews[["review_id", "user_id", "product_id"]].copy()
    out["dup_content"] = flag_dup_content(reviews).to_numpy()
    out["promo"] = flag_promo(reviews).to_numpy()
    out["burst"] = flag_burst(reviews).to_numpy()
    return out


def flags_by_pair(review_flags: pd.DataFrame) -> pd.DataFrame:
    """Gộp theo cặp (user_id, product_id): any() cho từng dấu hiệu."""
    cols = list(config.FLAG_NAMES)
    return review_flags.groupby(["user_id", "product_id"], as_index=False)[cols].any()


def summarize(review_flags: pd.DataFrame, pair_flags: pd.DataFrame) -> dict:
    """Số đếm tổng hợp (không chứa nội dung)."""
    cols = list(config.FLAG_NAMES)
    s = {"n_reviews": int(len(review_flags)), "n_pairs": int(len(pair_flags))}
    for c in cols:
        s[f"reviews_{c}"] = int(review_flags[c].sum())
        s[f"pairs_{c}"] = int(pair_flags[c].sum())
    s["pairs_dup_or_promo"] = int((pair_flags["dup_content"] | pair_flags["promo"]).sum())
    s["pairs_any"] = int(pair_flags[cols].any(axis=1).sum())
    return s
