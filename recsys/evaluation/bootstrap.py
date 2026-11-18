"""Bootstrap trên vectơ metric theo user (docs/DO_AN.md mục 4.5.3). Không huấn luyện lại."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from recsys import config


@dataclass(frozen=True)
class CI:
    """Trung bình và khoảng tin cậy percentile."""

    mean: float
    lo: float
    hi: float
    n: int

    @property
    def distinguishable(self) -> bool:
        """Khoảng không chứa 0 → phân biệt được với 0."""
        return not (self.lo <= 0.0 <= self.hi)

    def to_dict(self) -> dict:
        return {**asdict(self), "distinguishable": self.distinguishable}


def _ci(values: np.ndarray, n_boot: int, level: float, seed: int) -> CI:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    step = max(1, 2_000_000 // n)            # chia lô để không tốn bộ nhớ
    for s in range(0, n_boot, step):
        e = min(n_boot, s + step)
        idx = rng.integers(0, n, size=(e - s, n))
        means[s:e] = values[idx].mean(axis=1)
    a = (1 - level) / 2
    return CI(float(values.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), n)


def bootstrap_ci(values, n_boot: int = config.N_BOOTSTRAP, level: float = config.CI_LEVEL,
                 seed: int = config.BOOTSTRAP_SEED) -> CI:
    """Khoảng tin cậy cho trung bình một vectơ metric theo user (bỏ nan)."""
    return _ci(np.asarray(values, dtype=float), n_boot, level, seed)


def paired_bootstrap(a, b, n_boot: int = config.N_BOOTSTRAP, level: float = config.CI_LEVEL,
                     seed: int = config.BOOTSTRAP_SEED) -> CI:
    """Khoảng tin cậy cho trung bình hiệu (b − a) trên CÙNG tập user (hai vectơ cùng thứ tự)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("bootstrap cặp cần hai vectơ cùng độ dài, cùng thứ tự user")
    return _ci(b - a, n_boot, level, seed)
