"""ItemKNN: cosine giữa các cột của R (đã nhân w), giữ k láng giềng (docs/DO_AN.md mục 4.5.2)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.models.base import Recommender


class ItemKNN(Recommender):
    """điểm(u, i) = Σ_j R[u, j] · sim_k(j, i), sim_k giữ k láng giềng lớn nhất của mỗi j."""

    name = "itemknn"

    def __init__(self, k: int = 100, shrink: float = 0.0) -> None:
        super().__init__()
        self.k = k
        self.shrink = shrink

    def _fit(self, R, train, item_features, seed):
        X = R.tocsc().astype(np.float32)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        sim = (X.T @ X).toarray()
        denom = np.outer(norms, norms) + self.shrink
        denom[denom == 0] = 1.0
        sim /= denom
        np.fill_diagonal(sim, 0.0)
        n = sim.shape[0]
        if self.k < n:
            # giữ k giá trị lớn nhất mỗi hàng
            drop = np.argpartition(-sim, self.k, axis=1)[:, self.k:]
            np.put_along_axis(sim, drop, 0.0, axis=1)
        self.sim = sp.csr_matrix(sim)

    def _score(self, user_idx, item_idx):
        scores = (self.R[user_idx] @ self.sim).toarray()
        return scores[:, item_idx].astype(np.float32)
