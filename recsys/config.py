"""Cấu hình chung của phần recsys — NƠI DUY NHẤT chứa đường dẫn, mốc cắt, K, seed, ngưỡng gắn cờ, λ, ε.

Quy tắc (docs/DAC_TA.md mục 3):
- Ngưỡng gắn cờ và λ phải cố định ở đây TRƯỚC khi sinh dữ liệu tấn công; không chỉnh theo metric gợi ý.
- Mọi lựa chọn siêu tham số dựa trên validation, không dựa trên test.
Giá trị nào đánh dấu "TẠM" phải được chốt (và ghi ngày chốt) trước giai đoạn 2.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- đường dẫn
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RAW_REVIEWS: Path = RAW_DIR / "reviews_anonymized.jsonl"
RAW_PRODUCTS: Path = RAW_DIR / "products.jsonl"
PROCESSED_DIR: Path = DATA_DIR / "processed"
INTERACTIONS_PATH: Path = PROCESSED_DIR / "interactions.parquet"
REVIEWS_CLEAN_PATH: Path = PROCESSED_DIR / "reviews_clean.parquet"   # đã lọc trùng + scrub
PRODUCTS_CLEAN_PATH: Path = PROCESSED_DIR / "products.parquet"
SPLITS_DIR: Path = DATA_DIR / "splits"
LABELS_DIR: Path = DATA_DIR / "labels"
RESULTS_DIR: Path = ROOT / "results"
PREP_SUMMARY_PATH: Path = RESULTS_DIR / "prep_summary.json"


def split_dir(star: str, k: int) -> Path:
    """Thư mục chứa train/val/test của một biến thể sao + mức k-core, vd data/splits/all_2core."""
    if star not in STAR_VARIANTS:
        raise ValueError(f"star phải thuộc {STAR_VARIANTS}, nhận {star!r}")
    if k not in KCORE_LEVELS:
        raise ValueError(f"k phải thuộc {KCORE_LEVELS}, nhận {k!r}")
    return SPLITS_DIR / f"{star}_{k}core"


def run_dir(stage: str, run_id: str) -> Path:
    """Thư mục kết quả của một lần chạy: results/<stage>/<run_id>."""
    if stage not in STAGES:
        raise ValueError(f"stage phải thuộc {STAGES}, nhận {stage!r}")
    return RESULTS_DIR / stage / run_id


STAGES: tuple[str, ...] = ("pilot", "stage1", "stage2")

# ---------------------------------------------------------------- prep
VAL_QUANTILE: float = 0.80    # t_val  = phân vị 80% của created_at trên toàn bộ tương tác
TEST_QUANTILE: float = 0.90   # t_test = phân vị 90%
STAR_VARIANTS: tuple[str, ...] = ("all", "pos45")
POS45_MIN_RATING: int = 4     # pos45: chỉ rating >= 4 là dương
KCORE_LEVELS: tuple[int, ...] = (0, 2, 3, 5)   # 0 = không lọc; k-core chỉ áp trên train

# Số liệu đã kiểm (docs/DAC_TA.md mục 2) — prep phải khớp, lệch thì dừng lại báo.
EXPECTED: dict[str, int] = {
    "reviews_rows": 408_160,
    "reviews_unique": 353_035,
    "users": 227_557,
    "pairs": 347_901,
    "products_with_reviews": 7_360,
    "products_rows": 15_338,
    "products_unique": 15_336,
}
EXPECTED_RATING_COUNTS: dict[int, int] = {1: 5_016, 2: 2_376, 3: 5_366, 4: 29_656, 5: 310_621}
# k-core trên TOÀN BỘ cặp (chỉ để đối chiếu, không phải k-core trên train): (user, sp, cặp)
EXPECTED_KCORE_ALL_PAIRS: dict[int, tuple[int, int, int]] = {
    2: (54_656, 4_500, 173_263),
    3: (21_943, 2_931, 105_637),
    5: (6_235, 1_537, 50_140),
}

# ---------------------------------------------------------------- đánh giá
TOP_KS: tuple[int, ...] = (10, 20)
JACCARD_K: int = 10
N_BOOTSTRAP: int = 1_000
CI_LEVEL: float = 0.95
BOOTSTRAP_SEED: int = 12345

# ---------------------------------------------------------------- seed
SEEDS_STAGE1: tuple[int, ...] = (0, 1, 2)
SEEDS_STAGE2: tuple[int, ...] = (0, 1, 2, 3, 4)   # thiếu thời gian: giảm còn (0, 1, 2)
PILOT_SEEDS: tuple[int, ...] = (0,)

# ---------------------------------------------------------------- pilot
PILOT_JACCARD_THRESHOLD: float = 0.9   # giữ RQ3 nếu CI(ΔNDCG@10) không chứa 0 HOẶC Jaccard@10 TB < 0,9
PILOT_FLAGS: tuple[str, ...] = ("dup_content", "promo")

# ---------------------------------------------------------------- gắn cờ (mục 6.4)
DUP_MIN_LEN: int = 20          # nội dung chuẩn hóa dài > 20 ký tự
DUP_MIN_COUNT: int = 5         # và xuất hiện >= 5 lần trên toàn bộ dữ liệu
PROMO_KEYWORDS: tuple[str, ...] = (
    # TẠM — nhóm rà soát/bổ sung rồi chốt trước giai đoạn 2. So khớp trên văn bản đã lowercase.
    "săn sale", "trả công", "inbox", "zalo", "ib mình", "liên hệ", "kết bạn", "nhận quà", "mã giảm giá",
)
PROMO_USE_PHONE: bool = True   # có SĐT trong title/content TRƯỚC khi scrub cũng tính là promo
BURST_SAME_PRODUCT_MIN: int = 2   # cùng user >= 2 review cho cùng sản phẩm
BURST_N_PRODUCTS: int = 5         # TẠM: hoặc >= N sản phẩm ...
BURST_WINDOW_MIN: int = 60        # TẠM: ... trong cửa sổ T phút
FLAG_NAMES: tuple[str, ...] = ("dup_content", "promo", "burst")

# ---------------------------------------------------------------- trọng số tin cậy (mục 6.5)
# w_ui = max(ε, 1 − min(1, Σ_k λ_k · s_k(u,i)))
TRUST_EPS: float = 0.05
TRUST_LAMBDA: dict[str, float] = {"dup_content": 0.5, "promo": 0.5, "burst": 0.5}
TRUST_LAMBDA_IS_PROVISIONAL: bool = True   # TẠM: thay bằng precision trên data/labels/ rồi đặt False

# ---------------------------------------------------------------- tấn công (mục 6.7)
ATTACK_TYPES: tuple[str, ...] = ("random", "average", "bandwagon", "observed")
ATTACK_LEVELS: tuple[float, ...] = (0.01, 0.03, 0.05)   # số hồ sơ giả / số user train
N_TARGETS_UNPOPULAR: int = 10
N_TARGETS_MID: int = 10
FAKE_USER_ID_START: int = -1   # hồ sơ giả dùng user_id âm: -1, -2, ... (user thật luôn >= 0)

# ---------------------------------------------------------------- mô hình (giá trị mặc định; tinh chỉnh trên val)
MODELS: tuple[str, ...] = ("popular", "itemknn", "als", "bpr", "mf_bce", "lightfm")
MF_BCE_NEG_PER_POS: int = 4
