"""Cấu hình đường dẫn Medallion (Bronze -> Silver -> Gold) và SparkSession cho phần Big Data (IE212)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from recsys import config as recsys_config

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

log = logging.getLogger(__name__)

BIGDATA_RESULTS_DIR: Path = recsys_config.RESULTS_DIR / "bigdata"
BIGDATA_FIGURES_DIR: Path = BIGDATA_RESULTS_DIR / "figures"


def resolve_paths(
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Chuẩn hóa bộ đường dẫn cho local, Google Colab hoặc Kaggle.

    Thứ tự ưu tiên: tham số truyền vào -> biến môi trường (TIKI_DATA_DIR / TIKI_RESULTS_DIR)
    -> mặc định trong recsys.config.
    """
    raw_data_env = os.environ.get("TIKI_DATA_DIR")
    base_data = (
        Path(data_dir).expanduser()
        if data_dir is not None
        else (Path(raw_data_env).expanduser() if raw_data_env else recsys_config.DATA_DIR)
    )

    raw_res_env = os.environ.get("TIKI_RESULTS_DIR")
    base_results = (
        Path(results_dir).expanduser()
        if results_dir is not None
        else (Path(raw_res_env).expanduser() if raw_res_env else BIGDATA_RESULTS_DIR)
    )

    bronze_dir = base_data / "raw"
    bigdata_root = base_data / "bigdata"
    silver_dir = bigdata_root / "silver"
    gold_dir = bigdata_root / "gold"

    return {
        "data_dir": base_data,
        "bronze_dir": bronze_dir,
        "raw_reviews": bronze_dir / "reviews_anonymized.jsonl",
        "raw_products": bronze_dir / "products.jsonl",
        "silver_dir": silver_dir,
        "silver_reviews": silver_dir / "reviews_clean.parquet",
        "silver_interactions": silver_dir / "interactions.parquet",
        "silver_products": silver_dir / "products.parquet",
        "gold_dir": gold_dir,
        "splits_dir": gold_dir / "splits",
        "gold_review_flags": gold_dir / "review_flags.parquet",
        "gold_pair_flags": gold_dir / "pair_flags.parquet",
        "results_dir": base_results,
        "figures_dir": base_results / "figures",
    }


def build_spark(
    app_name: str = "Tiki-IE212-BigData",
    master: str = "local[*]",
    shuffle_partitions: int | None = None,
    driver_memory: str = "4g",
    adaptive: bool = True,
) -> "SparkSession":
    """Khởi tạo SparkSession (lazy import pyspark) với cấu hình tối ưu cho Colab/Kaggle."""
    from pyspark.sql import SparkSession

    if shuffle_partitions is None:
        shuffle_partitions = max(4, (os.cpu_count() or 2) * 2)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(int(shuffle_partitions)))
        .config("spark.sql.adaptive.enabled", "true" if adaptive else "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.log.level", "WARN")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info(
        "SparkSession %s | master=%s | shuffle_partitions=%d | adaptive=%s",
        app_name,
        master,
        shuffle_partitions,
        adaptive,
    )
    return spark
