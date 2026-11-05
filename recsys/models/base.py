"""Giao diện chung cho mọi mô hình gợi ý (docs/DAC_TA.md mục 5.2).

Mô hình con chỉ cần cài `_fit(R, train, item_features, seed)` và `_score(user_idx, item_idx)`.
Phần ánh xạ id ↔ chỉ số, dựng ma trận R (có trọng số), loại sản phẩm đã xem và lấy top-K nằm ở đây.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Recommender(ABC):
    """Lớp cơ sở. Điểm càng cao càng nên gợi ý.

    Quy ước:
    - `train` theo `recsys.schema.INTERACTION_COLUMNS` (cần ít nhất user_id, product_id).
    - `sample_weight`: mảng cùng độ dài `train` (w_ui), None = toàn 1. Mô hình không dùng w thì bỏ qua.
    - User không có trong train (cold): `score` trả hàng NaN, trừ khi `supports_cold = True`.
    """

    name: str = "base"
    supports_cold: bool = False   # Popular, LightFM đặt True
    uses_weight: bool = True      # LightFM đặt False

    def __init__(self) -> None:
        self.user_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.R: sp.csr_matrix | None = None   # user × item, ô (u,i) = w_ui (hoặc 1)
        self.fitted: bool = False

    # ------------------------------------------------------------ fit
    def fit(
        self,
        train: pd.DataFrame,
        item_features: pd.DataFrame | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
        seed: int = 0,
    ) -> "Recommender":
        """Dựng chỉ số + ma trận R rồi gọi `_fit` của mô hình con."""
        if sample_weight is not None and len(sample_weight) != len(train):
            raise ValueError("sample_weight phải cùng độ dài với train")
        self.user_ids = np.sort(train["user_id"].unique()).astype(np.int64)
        self.item_ids = np.sort(train["product_id"].unique()).astype(np.int64)
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self.item_index = {int(p): j for j, p in enumerate(self.item_ids)}
        rows = train["user_id"].map(self.user_index).to_numpy()
        cols = train["product_id"].map(self.item_index).to_numpy()
        w = np.ones(len(train), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        self.R = sp.csr_matrix((w, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)), dtype=np.float32)
        self.R.sum_duplicates()
        self._fit(self.R, train, item_features, seed)
        self.fitted = True
        return self

    @abstractmethod
    def _fit(self, R: sp.csr_matrix, train: pd.DataFrame, item_features: pd.DataFrame | None, seed: int) -> None:
        """Huấn luyện trên ma trận R (user × item, giá trị = w_ui)."""

    # ------------------------------------------------------------ score
    @abstractmethod
    def _score(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Điểm cho user ĐÃ BIẾT (chỉ số nội bộ) × item_idx, shape (len(user_idx), len(item_idx))."""

    def _score_cold(self, n_users: int, item_idx: np.ndarray) -> np.ndarray:
        """Điểm cho user lạ; mô hình có supports_cold=True phải ghi đè."""
        return np.full((n_users, len(item_idx)), np.nan, dtype=np.float32)

    def score(self, user_ids: Sequence[int], item_ids: Sequence[int] | None = None) -> np.ndarray:
        """Điểm (len(user_ids), số item) theo thứ tự `item_ids` (mặc định: mọi item trong train, theo self.item_ids).
        Item không có trong train nhận -inf."""
        self._check_fitted()
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if item_ids is None:
            item_idx = np.arange(len(self.item_ids))
            known_item = np.ones(len(item_idx), dtype=bool)
        else:
            mapped = np.array([self.item_index.get(int(p), -1) for p in item_ids], dtype=np.int64)
            known_item = mapped >= 0
            item_idx = mapped[known_item]
        out = np.full((len(user_ids), len(known_item)), -np.inf, dtype=np.float32)
        uidx = np.array([self.user_index.get(int(u), -1) for u in user_ids], dtype=np.int64)
        warm = uidx >= 0
        if warm.any():
            out[np.ix_(warm, known_item)] = self._score(uidx[warm], item_idx)
        if (~warm).any():
            out[np.ix_(~warm, known_item)] = self._score_cold(int((~warm).sum()), item_idx)
        return out

    # ------------------------------------------------------------ recommend
    def recommend(self, user_ids: Sequence[int], k: int, exclude_seen: bool = True) -> dict[int, list[int]]:
        """Top-k product_id cho mỗi user. Loại sản phẩm đã có trong train nếu exclude_seen.
        User cold với mô hình không hỗ trợ cold → danh sách rỗng. Hòa điểm: ưu tiên product_id nhỏ hơn."""
        self._check_fitted()
        user_ids = [int(u) for u in user_ids]
        scores = self.score(user_ids)
        result: dict[int, list[int]] = {}
        for row, u in enumerate(user_ids):
            s = scores[row].astype(np.float64, copy=True)
            if np.isnan(s).all():
                result[u] = []
                continue
            s[np.isnan(s)] = -np.inf
            uidx = self.user_index.get(u)
            if exclude_seen and uidx is not None:
                seen = self.R.indices[self.R.indptr[uidx]: self.R.indptr[uidx + 1]]
                s[seen] = -np.inf
            valid = np.flatnonzero(np.isfinite(s))
            if len(valid) == 0:
                result[u] = []
                continue
            kk = min(k, len(valid))
            # sắp theo (-điểm, chỉ số) → ổn định; item_ids đã sort nên chỉ số nhỏ = product_id nhỏ
            order = valid[np.lexsort((valid, -s[valid]))][:kk]
            result[u] = [int(p) for p in self.item_ids[order]]
        return result

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name}: chưa fit")
