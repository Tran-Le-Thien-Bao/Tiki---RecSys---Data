"""Quy ước dữ liệu dùng chung giữa các module (prep → models → evaluation → experiments).

Đây là "hợp đồng" để các thành viên làm song song: ai sinh bảng thì gọi validate_* trước khi ghi,
ai dùng bảng thì có thể dựa vào đúng tên cột/kiểu ở đây. Đổi quy ước = sửa file này + báo cả nhóm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

# ---------------------------------------------------------------- bảng tương tác
# data/processed/interactions.parquet và data/splits/<star>_<k>core/{train,val,test}.parquet
# Mỗi dòng = một cặp (user, product) duy nhất.
INTERACTION_COLUMNS: dict[str, str] = {
    "user_id": "int64",      # user thật >= 0; hồ sơ giả < 0 (config.FAKE_USER_ID_START)
    "product_id": "int64",
    "ts": "int64",           # unix giây của review SỚM NHẤT của cặp
    "rating": "int8",        # rating của review sớm nhất đó (1–5)
    "n_reviews": "int16",    # số review của cặp
    "is_fake": "bool",       # True chỉ cho hồ sơ tấn công (chỉ xuất hiện trong train)
}
# Cột tùy chọn: cờ theo cặp (mitigation/flags.py) và trọng số tin cậy (mitigation/trust.py)
OPTIONAL_INTERACTION_COLUMNS: dict[str, str] = {
    "dup_content": "bool",
    "promo": "bool",
    "burst": "bool",
    "w": "float32",          # trọng số tin cậy trong [ε, 1]
}
PAIR_KEY: tuple[str, str] = ("user_id", "product_id")

# ---------------------------------------------------------------- kết quả theo user
# results/<stage>/<run_id>/per_user.parquet — KHÔNG có user_id gốc, KHÔNG có nội dung review.
PER_USER_REQUIRED: dict[str, str] = {
    "user_hash": "string",   # evaluation.results.hash_user(user_id)
    "is_warm": "bool",
    "n_truth": "int32",      # số sản phẩm ground truth của user
}
# Cột metric đặt tên "<metric>@<K>", vd recall@10, ndcg@20, hit@10 (hit ratio sp mục tiêu).
METRIC_NAMES: tuple[str, ...] = ("recall", "ndcg", "hit", "target_rank", "target_score")
FORBIDDEN_RESULT_COLUMNS: frozenset[str] = frozenset(
    {"user_id", "user", "customer_id", "title", "content", "review_id"}
)


@dataclass(frozen=True)
class RunConfig:
    """Một cấu hình thí nghiệm. run_config(cfg) -> kết quả phải là hàm thuần của đối tượng này
    (để IE212 phân phối bằng PySpark: mỗi RunConfig một task)."""

    stage: str                      # pilot | stage1 | stage2
    model: str                      # config.MODELS
    star: str = "all"               # config.STAR_VARIANTS
    k: int = 0                      # config.KCORE_LEVELS
    seed: int = 0
    attack: str | None = None       # None = sạch; hoặc config.ATTACK_TYPES
    attack_level: float = 0.0       # config.ATTACK_LEVELS
    mitigation: str | None = None   # None | filter | trust | tce | rce
    eval_split: str = "val"         # val | test (test chỉ dùng khi đã chốt mọi lựa chọn)
    params: dict = field(default_factory=dict)   # siêu tham số mô hình

    @property
    def run_id(self) -> str:
        """Mã chạy ổn định, dùng làm tên thư mục kết quả."""
        parts = [self.model, f"{self.star}_{self.k}core", f"s{self.seed}", self.eval_split]
        if self.attack:
            parts.append(f"{self.attack}{self.attack_level:g}")
        if self.mitigation:
            parts.append(self.mitigation)
        if self.params:
            parts.append("-".join(f"{k}{v}" for k, v in sorted(self.params.items())))
        return "__".join(parts)

    def to_dict(self) -> dict:
        """Dạng dict (ghi vào summary.json)."""
        return asdict(self)


class SchemaError(ValueError):
    """Bảng không đúng quy ước."""


def _check_columns(df: pd.DataFrame, required: dict[str, str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: thiếu cột {missing}")
    for col, dtype in required.items():
        actual = str(df[col].dtype)
        if dtype == "string":
            ok = actual in ("string", "object")
        elif dtype.startswith("int"):
            ok = pd.api.types.is_integer_dtype(df[col])
        elif dtype.startswith("float"):
            ok = pd.api.types.is_float_dtype(df[col])
        elif dtype == "bool":
            ok = pd.api.types.is_bool_dtype(df[col])
        else:
            ok = actual == dtype
        if not ok:
            raise SchemaError(f"{name}: cột {col} kiểu {actual}, cần {dtype}")


def validate_interactions(df: pd.DataFrame, name: str = "interactions") -> None:
    """Kiểm tra bảng tương tác: đủ cột, đúng kiểu, mỗi cặp (user, product) duy nhất, rating 1–5."""
    _check_columns(df, INTERACTION_COLUMNS, name)
    opt = {c: t for c, t in OPTIONAL_INTERACTION_COLUMNS.items() if c in df.columns}
    _check_columns(df, opt, name)
    if df.duplicated(list(PAIR_KEY)).any():
        raise SchemaError(f"{name}: có cặp (user_id, product_id) trùng")
    if len(df) and not df["rating"].between(1, 5).all():
        raise SchemaError(f"{name}: rating ngoài 1–5")
    if len(df) and (df["is_fake"] != (df["user_id"] < 0)).any():
        raise SchemaError(f"{name}: is_fake phải trùng với user_id < 0")
    if "w" in df.columns and len(df) and not df["w"].between(0, 1).all():
        raise SchemaError(f"{name}: w ngoài [0, 1]")


def validate_per_user(df: pd.DataFrame, name: str = "per_user") -> None:
    """Kiểm tra bảng metric theo user: đủ cột bắt buộc, không chứa cột định danh/nội dung."""
    bad = FORBIDDEN_RESULT_COLUMNS.intersection(df.columns)
    if bad:
        raise SchemaError(f"{name}: không được chứa cột {sorted(bad)} (quyền riêng tư)")
    _check_columns(df, PER_USER_REQUIRED, name)
    for col in df.columns:
        if col in PER_USER_REQUIRED:
            continue
        metric, _, k = col.partition("@")
        if metric not in METRIC_NAMES or not k.isdigit():
            raise SchemaError(f"{name}: cột {col!r} không theo dạng <metric>@<K> với metric thuộc {METRIC_NAMES}")
