"""Metric gợi ý tính theo từng user (docs/DO_AN.md mục 4.5.3).

Quy ước:
- `recs`: dict user_id -> list product_id đã xếp hạng (đã loại sản phẩm có trong train).
- `truth`: dict user_id -> set product_id user tương tác trong val/test.
- Relevance nhị phân. Recall@K chia cho |G_u| (không phải min(K, |G_u|)).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from recsys.evaluation.results import hash_user


def recall_at_k(rec: Sequence[int], truth: set[int], k: int) -> float:
    """|top-K ∩ G| / |G|; G rỗng → nan."""
    if not truth:
        return float("nan")
    return len(set(rec[:k]) & truth) / len(truth)


def ndcg_at_k(rec: Sequence[int], truth: set[int], k: int) -> float:
    """NDCG@K nhị phân: DCG = Σ 1/log2(vị trí+1) trên các hit; IDCG với min(|G|, K) hit ở đầu."""
    if not truth:
        return float("nan")
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(rec[:k]) if p in truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg


def hit_at_k(rec: Sequence[int], targets: set[int], k: int) -> float:
    """1 nếu có ít nhất một sản phẩm mục tiêu trong top-K, ngược lại 0."""
    return float(bool(set(rec[:k]) & targets))


def jaccard_at_k(a: Sequence[int], b: Sequence[int], k: int) -> float:
    """|A ∩ B| / |A ∪ B| của hai top-K; cả hai rỗng → 1."""
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def truth_from_df(df: pd.DataFrame) -> dict[int, set[int]]:
    """Ground truth từ bảng tương tác val/test."""
    return {int(u): set(map(int, g)) for u, g in df.groupby("user_id")["product_id"]}


def evaluate(recs: Mapping[int, Sequence[int]], truth: Mapping[int, set[int]], warm_users: set[int],
             ks: Iterable[int] = (10, 20)) -> pd.DataFrame:
    """Bảng metric theo user (đúng recsys.schema.PER_USER_REQUIRED + cột <metric>@<K>).
    User có trong truth nhưng không có trong recs được tính với danh sách rỗng (metric = 0)."""
    rows = []
    for u, g in truth.items():
        rec = list(recs.get(u, []))
        row = {"user_hash": hash_user(u), "is_warm": u in warm_users, "n_truth": len(g)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(rec, g, k)
            row[f"ndcg@{k}"] = ndcg_at_k(rec, g, k)
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["n_truth"] = df["n_truth"].astype("int32")
        df["user_hash"] = df["user_hash"].astype("string")
    return df


def coverage(recs: Mapping[int, Sequence[int]], n_items: int, k: int) -> float:
    """Tỉ lệ sản phẩm (trong n_items) xuất hiện ít nhất một lần trong top-K của mọi user."""
    seen = set()
    for r in recs.values():
        seen.update(r[:k])
    return len(seen) / n_items if n_items else 0.0


def prediction_shift(score_clean: np.ndarray, score_attack: np.ndarray) -> float:
    """Trung bình (điểm sau tấn công − điểm sạch) trên mọi cặp (user, sản phẩm mục tiêu)."""
    a, b = np.asarray(score_clean, dtype=float), np.asarray(score_attack, dtype=float)
    if a.shape != b.shape:
        raise ValueError("hai ma trận điểm phải cùng kích thước")
    return float(np.nanmean(b - a))


def kendall_tau(scores_a: Mapping[str, float], scores_b: Mapping[str, float]) -> float:
    """Kendall tau giữa hai bảng xếp hạng mô hình (cùng tập tên mô hình)."""
    names = sorted(scores_a)
    if set(names) != set(scores_b):
        raise ValueError("hai bảng phải cùng tập mô hình")
    return float(kendalltau([scores_a[n] for n in names], [scores_b[n] for n in names]).statistic)
