"""Lọc k-core lặp: bỏ user và sản phẩm có < k tương tác cho đến khi ổn định."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def kcore(df: pd.DataFrame, k: int, user_col: str = "user_id", item_col: str = "product_id") -> pd.DataFrame:
    """Trả bản sao chỉ gồm các dòng thuộc k-core (mỗi user và mỗi sản phẩm có >= k dòng).
    k <= 1 → không lọc. Mỗi dòng phải là một cặp (user, item) duy nhất."""
    out = df.copy()
    if k <= 1:
        return out
    n_iter = 0
    while True:
        n_iter += 1
        u_cnt = out[user_col].map(out[user_col].value_counts())
        i_cnt = out[item_col].map(out[item_col].value_counts())
        keep = (u_cnt >= k) & (i_cnt >= k)
        if keep.all():
            break
        out = out[keep]
    log.debug("k-core k=%d hội tụ sau %d vòng: %d dòng", k, n_iter, len(out))
    return out.reset_index(drop=True)
