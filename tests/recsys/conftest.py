"""Dữ liệu giả nhỏ dùng chung cho test recsys (không đọc data/raw)."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def toy_interactions() -> pd.DataFrame:
    """4 user thật × 5 sản phẩm, 1 hồ sơ giả (user_id âm)."""
    rows = [
        # user_id, product_id, ts, rating
        (1, 10, 100, 5), (1, 11, 110, 4), (1, 12, 120, 5),
        (2, 10, 105, 5), (2, 13, 130, 3),
        (3, 11, 115, 5), (3, 12, 125, 5), (3, 14, 140, 1),
        (4, 10, 101, 5),
        (-1, 14, 150, 5), (-1, 10, 151, 5),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "product_id", "ts", "rating"])
    df = df.astype({"user_id": "int64", "product_id": "int64", "ts": "int64", "rating": "int8"})
    df["n_reviews"] = pd.Series([1] * len(df), dtype="int16")
    df["is_fake"] = df["user_id"] < 0
    return df
