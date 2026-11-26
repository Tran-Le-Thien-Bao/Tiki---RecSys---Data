"""PySpark ETL (Bronze -> Silver -> Gold), bộ gắn cờ Window Functions và Parity Check (IE212).

Tái hiện hoàn toàn pipeline trong recsys/prep/ và recsys/mitigation/flags.py bằng PySpark thuần,
đối chiếu 100% số liệu với results/prep_summary.json và results/pilot/pilot_summary.json.

Chạy:
    python -m bigdata.etl --star all pos45 --k 0 2 3 5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bigdata.config import build_spark, resolve_paths
from recsys import config as recsys_config
from recsys.prep.build_interactions import REVIEW_COLUMNS

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession

log = logging.getLogger(__name__)

# (?u) bật UNICODE_CHARACTER_CLASS trong Java Regex để \w và \s khớp tiếng Việt giống Python 3
EMAIL_REGEX = r"(?u)[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
PHONE_REGEX = r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)"


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def spark_clean_reviews(raw_df: "DataFrame") -> "DataFrame":
    """Lọc trùng review_id (giữ bản ghi đầu tiên), gắn cờ had_phone/had_email trước scrub, scrub PII."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    missing = [c for c in REVIEW_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"reviews thiếu cột {missing}")

    df = raw_df.select(*REVIEW_COLUMNS).withColumn("_row_id", F.monotonically_increasing_id())
    w_dedup = Window.partitionBy("review_id").orderBy(F.col("_row_id").asc())
    df = df.withColumn("_rn", F.row_number().over(w_dedup)).filter(F.col("_rn") == 1).drop("_row_id", "_rn")

    has_null = (
        df.filter(
            F.col("user_id").isNull()
            | F.col("product_id").isNull()
            | F.col("rating").isNull()
            | F.col("created_at").isNull()
        )
        .limit(1)
        .count()
        > 0
    )
    if has_null:
        raise ValueError("reviews có giá trị thiếu ở user_id/product_id/rating/created_at")

    title_raw = F.coalesce(F.col("title").cast("string"), F.lit(""))
    content_raw = F.coalesce(F.col("content").cast("string"), F.lit(""))
    combined_raw = F.concat(title_raw, F.lit("\n"), content_raw)

    title_clean = F.regexp_replace(F.regexp_replace(title_raw, EMAIL_REGEX, "[email]"), PHONE_REGEX, "[sdt]")
    content_clean = F.regexp_replace(F.regexp_replace(content_raw, EMAIL_REGEX, "[email]"), PHONE_REGEX, "[sdt]")

    return df.select(
        F.col("review_id").cast("long").alias("review_id"),
        F.col("user_id").cast("long").alias("user_id"),
        F.col("product_id").cast("long").alias("product_id"),
        F.col("rating").cast("byte").alias("rating"),
        title_clean.alias("title"),
        content_clean.alias("content"),
        F.coalesce(F.col("thank_count"), F.lit(0)).cast("long").alias("thank_count"),
        F.col("created_at").cast("long").alias("created_at"),
        F.coalesce(F.col("is_verified_purchase"), F.lit(False)).cast("boolean").alias("is_verified_purchase"),
        combined_raw.rlike(PHONE_REGEX).cast("boolean").alias("had_phone"),
        combined_raw.rlike(EMAIL_REGEX).cast("boolean").alias("had_email"),
    )


def spark_clean_products(raw_df: "DataFrame") -> "DataFrame":
    """Lọc trùng theo product_id; chuyển specifications thành chuỗi JSON nếu là struct/map."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    df = raw_df.withColumn("_row_id", F.monotonically_increasing_id())
    w_dedup = Window.partitionBy("product_id").orderBy(F.col("_row_id").asc())
    df = df.withColumn("_rn", F.row_number().over(w_dedup)).filter(F.col("_rn") == 1).drop("_row_id", "_rn")

    if "specifications" in df.columns and not isinstance(df.schema["specifications"].dataType, StringType):
        df = df.withColumn(
            "specifications",
            F.when(F.col("specifications").isNotNull(), F.to_json(F.col("specifications"))).otherwise(
                F.lit(None).cast("string")
            ),
        )
    return df.withColumn("product_id", F.col("product_id").cast("long"))


def spark_to_interactions(reviews_df: "DataFrame") -> "DataFrame":
    """Mỗi cặp (user_id, product_id) giữ review sớm nhất (hòa: review_id nhỏ hơn) + đếm n_reviews."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w_pair = Window.partitionBy("user_id", "product_id")
    w_first = Window.partitionBy("user_id", "product_id").orderBy(
        F.col("created_at").asc(), F.col("review_id").asc()
    )
    return (
        reviews_df.withColumn("n_reviews", F.count(F.lit(1)).over(w_pair).cast("short"))
        .withColumn("_rn", F.row_number().over(w_first))
        .filter(F.col("_rn") == 1)
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("product_id").cast("long").alias("product_id"),
            F.col("created_at").cast("long").alias("ts"),
            F.col("rating").cast("byte").alias("rating"),
            F.col("n_reviews").cast("short").alias("n_reviews"),
            F.lit(False).cast("boolean").alias("is_fake"),
        )
    )


def spark_time_cutoffs(
    inter_df: "DataFrame",
    q_val: float = recsys_config.VAL_QUANTILE,
    q_test: float = recsys_config.TEST_QUANTILE,
    exact: bool = True,
) -> tuple[int, int]:
    """Tính (t_val, t_test) trên biến thể all. exact=True khớp np.quantile; exact=False dùng approxQuantile."""
    from pyspark.sql import functions as F

    if exact:
        row = inter_df.agg(F.expr(f"percentile(ts, array({float(q_val)}, {float(q_test)}))").alias("q")).first()
        vals = row["q"]
    else:
        vals = inter_df.approxQuantile("ts", [float(q_val), float(q_test)], 0.0001)
    return int(math.ceil(vals[0])), int(math.ceil(vals[1]))


def spark_kcore(df: "DataFrame", k: int, use_checkpoint: bool = True) -> "DataFrame":
    """Lọc k-core lặp trên Spark; localCheckpoint định kỳ để cắt DAG lineage."""
    from pyspark.sql import functions as F

    if k <= 1:
        return df

    cur = df
    prev_count = cur.count()
    n_iter = 0
    while True:
        n_iter += 1
        u_keep = cur.groupBy("user_id").count().filter(F.col("count") >= k).select("user_id")
        i_keep = cur.groupBy("product_id").count().filter(F.col("count") >= k).select("product_id")
        nxt = cur.join(u_keep, on="user_id", how="left_semi").join(
            F.broadcast(i_keep), on="product_id", how="left_semi"
        )
        if use_checkpoint and n_iter % 2 == 0:
            nxt = nxt.localCheckpoint(eager=True)
        else:
            nxt = nxt.cache()
        new_count = nxt.count()
        if new_count == prev_count:
            return nxt
        cur = nxt
        prev_count = new_count


def _spark_eval_stats(df: "DataFrame", train_users_df: "DataFrame", n_before_item_filter: int) -> dict[str, Any]:
    from pyspark.sql import functions as F

    tagged = df.join(
        F.broadcast(train_users_df.withColumn("_is_warm", F.lit(1))),
        on="user_id",
        how="left",
    ).withColumn("_is_warm", F.coalesce(F.col("_is_warm"), F.lit(0)))

    agg_row = tagged.agg(
        F.count(F.lit(1)).alias("n_interactions"),
        F.countDistinct("product_id").alias("n_items"),
        F.coalesce(F.sum("_is_warm"), F.lit(0)).alias("n_warm_interactions"),
    ).first()

    u_row = (
        tagged.select("user_id", "_is_warm")
        .distinct()
        .agg(
            F.count(F.lit(1)).alias("n_users"),
            F.coalesce(F.sum("_is_warm"), F.lit(0)).alias("n_warm_users"),
        )
        .first()
    )

    n_inter = int(agg_row["n_interactions"])
    n_warm_inter = int(agg_row["n_warm_interactions"])
    n_users = int(u_row["n_users"])
    n_warm_users = int(u_row["n_warm_users"])
    return {
        "n_interactions": n_inter,
        "n_interactions_dropped_item_not_in_train": int(n_before_item_filter - n_inter),
        "n_users": n_users,
        "n_items": int(agg_row["n_items"]),
        "n_warm_users": n_warm_users,
        "n_cold_users": n_users - n_warm_users,
        "warm_user_ratio": round(n_warm_users / n_users, 4) if n_users else None,
        "n_warm_interactions": n_warm_inter,
        "n_cold_interactions": n_inter - n_warm_inter,
    }


def spark_make_split(
    inter_df: "DataFrame",
    star: str,
    k: int,
    t_val: int,
    t_test: int,
    use_checkpoint: bool = True,
) -> tuple["DataFrame", "DataFrame", "DataFrame", dict[str, Any]]:
    """Chia thời gian + k-core chỉ trên train + bỏ sản phẩm ngoài train ở val/test."""
    from pyspark.sql import functions as F

    if star == "all":
        pos = inter_df
    elif star == "pos45":
        pos = inter_df.filter(F.col("rating") >= recsys_config.POS45_MIN_RATING)
    else:
        raise ValueError(f"star không hợp lệ: {star!r}")

    train_raw = pos.filter(F.col("ts") < t_val)
    val_raw = pos.filter((F.col("ts") >= t_val) & (F.col("ts") < t_test))
    test_raw = pos.filter(F.col("ts") >= t_test)

    n_train_before = int(train_raw.count())
    n_val_before = int(val_raw.count())
    n_test_before = int(test_raw.count())

    train = spark_kcore(train_raw, k, use_checkpoint=use_checkpoint).cache()
    train_items = train.select("product_id").distinct().cache()
    train_users = train.select("user_id").distinct().cache()

    val = val_raw.join(F.broadcast(train_items), on="product_id", how="left_semi").cache()
    test = test_raw.join(F.broadcast(train_items), on="product_id", how="left_semi").cache()

    n_train = int(train.count())
    n_train_users = int(train_users.count())
    n_train_items = int(train_items.count())
    median_upu = 0.0
    if n_train > 0:
        med_row = (
            train.groupBy("user_id")
            .count()
            .agg(F.expr("percentile(count, 0.5)").alias("med"))
            .first()
        )
        median_upu = float(med_row["med"]) if med_row["med"] is not None else 0.0

    stats = {
        "train": {
            "n_interactions_before_kcore": n_train_before,
            "n_interactions": n_train,
            "n_users": n_train_users,
            "n_items": n_train_items,
            "density": round(n_train / (n_train_users * n_train_items), 8) if n_train else 0.0,
            "median_interactions_per_user": median_upu,
        },
        "val": _spark_eval_stats(val, train_users, n_val_before),
        "test": _spark_eval_stats(test, train_users, n_test_before),
    }
    return train, val, test, stats


def spark_normalize_text(col: "Column") -> "Column":
    """lowercase, gộp mọi whitespace thành 1 dấu cách rồi trim (khớp Python strip + \\s+ -> ' ')."""
    from pyspark.sql import functions as F

    return F.trim(F.regexp_replace(F.lower(F.coalesce(col.cast("string"), F.lit(""))), r"(?u)\s+", " "))


def spark_flag_reviews(
    reviews_df: "DataFrame",
    min_len: int = recsys_config.DUP_MIN_LEN,
    min_count: int = recsys_config.DUP_MIN_COUNT,
    keywords: tuple[str, ...] = recsys_config.PROMO_KEYWORDS,
    use_phone: bool = recsys_config.PROMO_USE_PHONE,
    same_product_min: int = recsys_config.BURST_SAME_PRODUCT_MIN,
    n_products: int = recsys_config.BURST_N_PRODUCTS,
    window_min: int = recsys_config.BURST_WINDOW_MIN,
) -> "DataFrame":
    """Gắn 3 cờ dup_content, promo, burst bằng Spark SQL + Window Functions thuần (không UDF)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    base = reviews_df.withColumn("_norm_content", spark_normalize_text(F.col("content")))

    # 1. dup_content: tách riêng nhóm nội dung dài để tránh skew ở chuỗi rỗng
    dup_keys = (
        base.filter(F.length("_norm_content") > min_len)
        .groupBy("_norm_content")
        .count()
        .filter(F.col("count") >= min_count)
        .select(F.col("_norm_content").alias("_dup_key"), F.lit(True).alias("dup_content"))
    )
    flagged = base.join(
        F.broadcast(dup_keys),
        on=F.col("_norm_content") == F.col("_dup_key"),
        how="left",
    ).withColumn("dup_content", F.coalesce(F.col("dup_content"), F.lit(False)))

    # 2. promo: khớp title.fillna('') + ' ' + content.fillna('') đã normalize, hoặc had_phone
    promo_text = spark_normalize_text(
        F.concat(
            F.coalesce(F.col("title").cast("string"), F.lit("")),
            F.lit(" "),
            F.coalesce(F.col("content").cast("string"), F.lit("")),
        )
    )
    if keywords:
        pat = "|".join(re.escape(k.lower()) for k in keywords)
        promo_col = promo_text.rlike(pat)
    else:
        promo_col = F.lit(False)
    if use_phone and "had_phone" in flagged.columns:
        promo_col = promo_col | F.coalesce(F.col("had_phone").cast("boolean"), F.lit(False))
    flagged = flagged.withColumn("promo", promo_col.cast("boolean"))

    # 3. burst: cùng cặp >= same_product_min HOẶC >= n_products sản phẩm trong cửa sổ window_min phút
    win_sec = int(window_min) * 60
    same_prod = F.count(F.lit(1)).over(Window.partitionBy("user_id", "product_id")) >= same_product_min
    w_back = Window.partitionBy("user_id").orderBy(F.col("created_at").asc()).rangeBetween(-win_sec, 0)
    w_fwd = Window.partitionBy("user_id").orderBy(F.col("created_at").asc()).rangeBetween(0, win_sec)
    flagged = flagged.withColumn(
        "_burst_end",
        (F.size(F.collect_set("product_id").over(w_back)) >= n_products).cast("int"),
    ).withColumn(
        "burst",
        (same_prod | (F.max("_burst_end").over(w_fwd) > 0)).cast("boolean"),
    )

    return flagged.select("review_id", "user_id", "product_id", "dup_content", "promo", "burst")


def spark_flags_by_pair(review_flags_df: "DataFrame") -> "DataFrame":
    """Gộp cờ theo cặp (user_id, product_id): True nếu có ít nhất 1 review bị gắn cờ."""
    from pyspark.sql import functions as F

    return review_flags_df.groupBy("user_id", "product_id").agg(
        F.max("dup_content").cast("boolean").alias("dup_content"),
        F.max("promo").cast("boolean").alias("promo"),
        F.max("burst").cast("boolean").alias("burst"),
    )


def spark_summarize_flags(review_flags_df: "DataFrame", pair_flags_df: "DataFrame") -> dict[str, int]:
    """Số đếm tổng hợp khớp với recsys/mitigation/flags.py:summarize."""
    from pyspark.sql import functions as F

    rf_row = review_flags_df.agg(
        F.count(F.lit(1)).alias("n_reviews"),
        *[F.coalesce(F.sum(F.col(c).cast("long")), F.lit(0)).alias(f"reviews_{c}") for c in recsys_config.FLAG_NAMES],
    ).first()

    pf_row = pair_flags_df.agg(
        F.count(F.lit(1)).alias("n_pairs"),
        *[F.coalesce(F.sum(F.col(c).cast("long")), F.lit(0)).alias(f"pairs_{c}") for c in recsys_config.FLAG_NAMES],
        F.coalesce(F.sum((F.col("dup_content") | F.col("promo")).cast("long")), F.lit(0)).alias("pairs_dup_or_promo"),
        F.coalesce(
            F.sum((F.col("dup_content") | F.col("promo") | F.col("burst")).cast("long")),
            F.lit(0),
        ).alias("pairs_any"),
    ).first()

    out = {"n_reviews": int(rf_row["n_reviews"]), "n_pairs": int(pf_row["n_pairs"])}
    for c in recsys_config.FLAG_NAMES:
        out[f"reviews_{c}"] = int(rf_row[f"reviews_{c}"])
        out[f"pairs_{c}"] = int(pf_row[f"pairs_{c}"])
    out["pairs_dup_or_promo"] = int(pf_row["pairs_dup_or_promo"])
    out["pairs_any"] = int(pf_row["pairs_any"])
    return out


def compare_parity(
    spark_summary: dict[str, Any],
    pandas_prep_summary: dict[str, Any] | None = None,
    spark_flag_summary: dict[str, Any] | None = None,
    pandas_pilot_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Đối chiếu 100% số liệu PySpark với kết quả Pandas đã lưu (hàm thuần không cần Spark)."""
    mismatches: list[str] = []
    checked = 0

    def _check(path: str, actual: Any, expected: Any) -> None:
        nonlocal checked
        checked += 1
        if actual != expected:
            mismatches.append(f"{path}: PySpark={actual!r} != Pandas={expected!r}")

    if pandas_prep_summary:
        for k, exp in pandas_prep_summary.get("counts", {}).items():
            _check(f"counts.{k}", spark_summary.get("counts", {}).get(k), exp)
        for k, exp in pandas_prep_summary.get("rating_counts", {}).items():
            _check(f"rating_counts.{k}", spark_summary.get("rating_counts", {}).get(str(k)), exp)
        for k in ("t_val", "t_test"):
            _check(f"cutoffs.{k}", spark_summary.get("cutoffs", {}).get(k), pandas_prep_summary.get("cutoffs", {}).get(k))
        for k in ("n_reviews_with_phone_before_scrub", "n_reviews_with_email_before_scrub", "pairs_with_multiple_reviews"):
            if k in pandas_prep_summary:
                _check(k, spark_summary.get(k), pandas_prep_summary.get(k))
        for sp_name, sp_exp in pandas_prep_summary.get("splits", {}).items():
            sp_act = spark_summary.get("splits", {}).get(sp_name)
            if sp_act is None:
                continue
            for part in ("train", "val", "test"):
                for metric in ("n_interactions", "n_users", "n_items", "n_warm_users"):
                    if metric in sp_exp.get(part, {}):
                        _check(f"splits.{sp_name}.{part}.{metric}", sp_act.get(part, {}).get(metric), sp_exp[part][metric])

    if spark_flag_summary is not None and pandas_pilot_summary is not None:
        exp_flags = pandas_pilot_summary.get("flags", {})
        for k, exp in exp_flags.items():
            _check(f"flags.{k}", spark_flag_summary.get(k), exp)

    return {"all_match": not mismatches, "checked_fields": checked, "mismatches": mismatches}


def run_etl(
    spark: "SparkSession",
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    stars: list[str] | None = None,
    ks: list[int] | None = None,
    write: bool = True,
    use_checkpoint: bool = True,
) -> dict[str, Any]:
    """Chạy toàn bộ ETL Medallion Bronze -> Silver -> Gold và đối chiếu Parity."""
    from pyspark.sql import functions as F

    t0 = time.perf_counter()
    paths = resolve_paths(data_dir=data_dir, results_dir=results_dir)
    stars = stars or ["all", "pos45"]
    ks = ks if ks is not None else list(recsys_config.KCORE_LEVELS)

    raw_reviews = spark.read.json(str(paths["raw_reviews"])).cache()
    raw_products = spark.read.json(str(paths["raw_products"])).cache()
    n_raw_reviews = int(raw_reviews.count())
    n_raw_products = int(raw_products.count())

    reviews_clean = spark_clean_reviews(raw_reviews).cache()
    products_clean = spark_clean_products(raw_products).cache()
    interactions = spark_to_interactions(reviews_clean).cache()

    n_reviews_unique = int(reviews_clean.count())
    n_products_unique = int(products_clean.count())
    n_pairs = int(interactions.count())
    n_users = int(reviews_clean.select("user_id").distinct().count())
    n_prod_rev = int(reviews_clean.select("product_id").distinct().count())

    rating_rows = reviews_clean.groupBy("rating").count().collect()
    rating_counts = {str(int(r["rating"])): int(r["count"]) for r in sorted(rating_rows, key=lambda x: int(x["rating"]))}

    agg_extra = reviews_clean.agg(
        F.coalesce(F.sum(F.col("had_phone").cast("long")), F.lit(0)).alias("phone"),
        F.coalesce(F.sum(F.col("had_email").cast("long")), F.lit(0)).alias("email"),
    ).first()
    n_multi_pairs = int(interactions.filter(F.col("n_reviews") > 1).count())

    ts_bounds = interactions.agg(F.min("ts").alias("ts_min"), F.max("ts").alias("ts_max")).first()
    t_val, t_test = spark_time_cutoffs(interactions, exact=True)

    counts = {
        "reviews_rows": n_raw_reviews,
        "reviews_unique": n_reviews_unique,
        "users": n_users,
        "pairs": n_pairs,
        "products_with_reviews": n_prod_rev,
        "products_rows": n_raw_products,
        "products_unique": n_products_unique,
    }

    if write:
        paths["silver_dir"].mkdir(parents=True, exist_ok=True)
        paths["gold_dir"].mkdir(parents=True, exist_ok=True)
        reviews_clean.write.mode("overwrite").parquet(str(paths["silver_reviews"]))
        products_clean.write.mode("overwrite").parquet(str(paths["silver_products"]))
        interactions.write.mode("overwrite").parquet(str(paths["silver_interactions"]))

    splits_stats: dict[str, Any] = {}
    for star in stars:
        for k in ks:
            tr, va, te, st = spark_make_split(interactions, star, k, t_val, t_test, use_checkpoint=use_checkpoint)
            splits_stats[f"{star}_{k}core"] = st
            if write:
                sp_dir = paths["splits_dir"] / f"{star}_{k}core"
                sp_dir.mkdir(parents=True, exist_ok=True)
                tr.write.mode("overwrite").parquet(str(sp_dir / "train.parquet"))
                va.write.mode("overwrite").parquet(str(sp_dir / "val.parquet"))
                te.write.mode("overwrite").parquet(str(sp_dir / "test.parquet"))
            tr.unpersist()
            va.unpersist()
            te.unpersist()

    # Bộ gắn cờ tầng Gold
    review_flags = spark_flag_reviews(reviews_clean).cache()
    pair_flags = spark_flags_by_pair(review_flags).cache()
    flag_summary = spark_summarize_flags(review_flags, pair_flags)
    if write:
        review_flags.write.mode("overwrite").parquet(str(paths["gold_review_flags"]))
        pair_flags.write.mode("overwrite").parquet(str(paths["gold_pair_flags"]))

    etl_summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "engine": "pyspark",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "counts": counts,
        "rating_counts": rating_counts,
        "n_reviews_with_phone_before_scrub": int(agg_extra["phone"]),
        "n_reviews_with_email_before_scrub": int(agg_extra["email"]),
        "pairs_with_multiple_reviews": n_multi_pairs,
        "cutoffs": {
            "q_val": recsys_config.VAL_QUANTILE,
            "q_test": recsys_config.TEST_QUANTILE,
            "t_val": t_val,
            "t_val_iso": _iso(t_val),
            "t_test": t_test,
            "t_test_iso": _iso(t_test),
            "ts_min_iso": _iso(int(ts_bounds["ts_min"])),
            "ts_max_iso": _iso(int(ts_bounds["ts_max"])),
        },
        "splits": splits_stats,
        "flags": flag_summary,
    }

    pd_prep = (
        json.loads(recsys_config.PREP_SUMMARY_PATH.read_text(encoding="utf-8"))
        if recsys_config.PREP_SUMMARY_PATH.exists()
        else None
    )
    pilot_path = recsys_config.RESULTS_DIR / "pilot" / "pilot_summary.json"
    pd_pilot = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else None
    parity = compare_parity(etl_summary, pd_prep, flag_summary, pd_pilot)
    etl_summary["parity"] = parity

    if write:
        paths["results_dir"].mkdir(parents=True, exist_ok=True)
        (paths["results_dir"] / "etl_summary.json").write_text(
            json.dumps(etl_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (paths["results_dir"] / "parity_check.json").write_text(
            json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    log.info("ETL PySpark xong trong %.2fs | parity all_match=%s (%d fields)",
             etl_summary["elapsed_s"], parity["all_match"], parity["checked_fields"])
    return etl_summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PySpark Medallion ETL + Parity Check (IE212)")
    ap.add_argument("--data-dir", default=None, help="Thư mục dữ liệu gốc (chứa raw/)")
    ap.add_argument("--results-dir", default=None, help="Thư mục lưu kết quả bigdata")
    ap.add_argument("--star", nargs="+", default=["all", "pos45"], choices=list(recsys_config.STAR_VARIANTS))
    ap.add_argument("--k", nargs="+", type=int, default=list(recsys_config.KCORE_LEVELS), choices=list(recsys_config.KCORE_LEVELS))
    ap.add_argument("--master", default="local[*]")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = build_spark(app_name="Tiki-IE212-ETL", master=args.master)
    try:
        summary = run_etl(spark, data_dir=args.data_dir, results_dir=args.results_dir, stars=args.star, ks=args.k)
        if not summary["parity"]["all_match"]:
            log.warning("Có sai lệch parity: %s", summary["parity"]["mismatches"])
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
