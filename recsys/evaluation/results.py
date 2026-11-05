"""Ghi kết quả một lần chạy: results/<stage>/<run_id>/{per_user.parquet, summary.json}.

Chỉ ghi dạng tổng hợp + metric theo user đã băm (docs/DAC_TA.md mục 3, quy tắc 1). Mọi experiment dùng save_run.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys import config
from recsys.schema import validate_per_user

log = logging.getLogger(__name__)

_HASH_SALT = "tiki-recsys-per-user-v1"   # đổi salt = mọi user_hash đổi; giữ nguyên trong suốt đồ án


def hash_user(user_id: int) -> str:
    """Băm user_id (đã giả danh) thêm một lần nữa để file kết quả không chứa id gốc."""
    return hashlib.sha256(f"{_HASH_SALT}:{int(user_id)}".encode()).hexdigest()[:16]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def save_run(stage: str, run_id: str, per_user: pd.DataFrame, summary: dict, root: Path | None = None) -> Path:
    """Kiểm tra per_user theo schema rồi ghi per_user.parquet + summary.json. Trả về thư mục kết quả."""
    validate_per_user(per_user)
    out = (root / stage / run_id) if root is not None else config.run_dir(stage, run_id)
    out.mkdir(parents=True, exist_ok=True)
    per_user.to_parquet(out / "per_user.parquet", index=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, ensure_ascii=False, indent=2)
    log.info("Đã ghi kết quả %s/%s (%d user)", stage, run_id, len(per_user))
    return out
