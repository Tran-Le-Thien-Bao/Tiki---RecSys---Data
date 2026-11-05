"""Chia train/val/test theo mốc thời gian toàn cục, k-core chỉ trên train, tách warm/cold (docs/DAC_TA.md mục 5.1, bước 4–7)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys import config
from recsys.prep.kcore import kcore
from recsys.schema import validate_interactions

log = logging.getLogger(__name__)


def time_cutoffs(interactions: pd.DataFrame, q_val: float = config.VAL_QUANTILE,
                 q_test: float = config.TEST_QUANTILE) -> tuple[int, int]:
    """(t_val, t_test) = phân vị q_val, q_test của ts trên TOÀN BỘ tương tác (biến thể all),
    dùng chung cho mọi biến thể sao để các bản chia so sánh được. Làm tròn lên giây nguyên."""
    t_val, t_test = np.quantile(interactions["ts"].to_numpy(), [q_val, q_test])
    return int(np.ceil(t_val)), int(np.ceil(t_test))


def apply_star(interactions: pd.DataFrame, star: str) -> pd.DataFrame:
    """all: mọi cặp là dương. pos45: chỉ giữ cặp có rating >= POS45_MIN_RATING (áp cho cả train/val/test)."""
    if star == "all":
        return interactions.copy()
    if star == "pos45":
        return interactions[interactions["rating"] >= config.POS45_MIN_RATING].copy()
    raise ValueError(f"star không hợp lệ: {star!r}")


def temporal_split(df: pd.DataFrame, t_val: int, t_test: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train: ts < t_val; val: t_val <= ts < t_test; test: ts >= t_test."""
    if not t_val <= t_test:
        raise ValueError("cần t_val <= t_test")
    ts = df["ts"]
    return (df[ts < t_val].reset_index(drop=True),
            df[(ts >= t_val) & (ts < t_test)].reset_index(drop=True),
            df[ts >= t_test].reset_index(drop=True))


@dataclass
class Split:
    """Một bản chia hoàn chỉnh của (star, k)."""

    star: str
    k: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stats: dict


def _eval_stats(df: pd.DataFrame, train_users: set[int], n_before_item_filter: int) -> dict:
    """Thống kê warm/cold của val hoặc test."""
    users = df["user_id"].unique()
    warm_mask_rows = df["user_id"].isin(train_users)
    n_warm = int(np.isin(users, list(train_users)).sum()) if len(users) else 0
    n_users = int(len(users))
    return {
        "n_interactions": int(len(df)),
        "n_interactions_dropped_item_not_in_train": int(n_before_item_filter - len(df)),
        "n_users": n_users,
        "n_items": int(df["product_id"].nunique()),
        "n_warm_users": n_warm,
        "n_cold_users": n_users - n_warm,
        "warm_user_ratio": round(n_warm / n_users, 4) if n_users else None,
        "n_warm_interactions": int(warm_mask_rows.sum()),
        "n_cold_interactions": int((~warm_mask_rows).sum()),
    }


def make_split(interactions: pd.DataFrame, star: str, k: int, t_val: int, t_test: int) -> Split:
    """interactions (biến thể all) → áp sao → chia thời gian → k-core trên train → bỏ khỏi val/test
    sản phẩm không có trong train → thống kê warm/cold."""
    pos = apply_star(interactions, star)
    train_raw, val_raw, test_raw = temporal_split(pos, t_val, t_test)
    train = kcore(train_raw, k)
    train_items = set(train["product_id"].unique())
    train_users = set(train["user_id"].unique())
    val = val_raw[val_raw["product_id"].isin(train_items)].reset_index(drop=True)
    test = test_raw[test_raw["product_id"].isin(train_items)].reset_index(drop=True)
    for name, d in (("train", train), ("val", val), ("test", test)):
        validate_interactions(d, f"{star}_{k}core/{name}")
    if len(train) and len(val) and train["ts"].max() >= val["ts"].min():
        raise AssertionError("rò rỉ thời gian: train chạm vào val")
    stats = {
        "train": {
            "n_interactions_before_kcore": int(len(train_raw)),
            "n_interactions": int(len(train)),
            "n_users": int(len(train_users)),
            "n_items": int(len(train_items)),
            "density": round(len(train) / (len(train_users) * len(train_items)), 8) if len(train) else 0.0,
            "median_interactions_per_user": float(train.groupby("user_id").size().median()) if len(train) else 0.0,
        },
        "val": _eval_stats(val, train_users, len(val_raw)),
        "test": _eval_stats(test, train_users, len(test_raw)),
    }
    return Split(star, k, train, val, test, stats)


def write_split(split: Split, root: Path | None = None) -> Path:
    """Ghi data/splits/<star>_<k>core/{train,val,test}.parquet."""
    out = (root / f"{split.star}_{split.k}core") if root is not None else config.split_dir(split.star, split.k)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        getattr(split, name).to_parquet(out / f"{name}.parquet", index=False)
    return out
