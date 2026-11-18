"""ALS (Hu, Koren, Volinsky 2008) qua thư viện implicit; độ tin cậy c = 1 + α·w (docs/DO_AN.md mục 4.5.2)."""
from __future__ import annotations

import numpy as np

from recsys.models.base import Recommender


class ALS(Recommender):
    """Bọc implicit.cpu.als.AlternatingLeastSquares; một luồng để cùng seed ra cùng kết quả."""

    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.01, alpha: float = 1.0,
                 iterations: int = 15, num_threads: int = 1) -> None:
        super().__init__()
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha,
                           iterations=iterations, num_threads=num_threads)

    def _fit(self, R, train, item_features, seed):
        from implicit.cpu.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        model = AlternatingLeastSquares(random_state=seed, **self.params)
        with threadpool_limits(1, "blas"):   # implicit khuyến nghị: BLAS đa luồng làm ALS chậm hẳn
            model.fit(R.tocsr().astype(np.float32), show_progress=False)
        self.user_factors = np.asarray(model.user_factors)
        self.item_factors = np.asarray(model.item_factors)

    def _score(self, user_idx, item_idx):
        return (self.user_factors[user_idx] @ self.item_factors[item_idx].T).astype(np.float32)
