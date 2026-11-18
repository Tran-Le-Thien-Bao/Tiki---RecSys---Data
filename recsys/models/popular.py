"""Popular: điểm của sản phẩm = tổng trọng số tương tác trong train (docs/DO_AN.md mục 4.5.2)."""
from __future__ import annotations

import numpy as np

from recsys.models.base import Recommender


class Popular(Recommender):
    """Mọi user nhận cùng một danh sách; gợi ý được cho user cold."""

    name = "popular"
    supports_cold = True

    def _fit(self, R, train, item_features, seed):
        self.pop = np.asarray(R.sum(axis=0), dtype=np.float32).ravel()

    def _score(self, user_idx, item_idx):
        return np.tile(self.pop[item_idx], (len(user_idx), 1))

    def _score_cold(self, n_users, item_idx):
        return np.tile(self.pop[item_idx], (n_users, 1))
