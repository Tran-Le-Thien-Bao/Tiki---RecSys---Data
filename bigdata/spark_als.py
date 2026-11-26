"""Spark MLlib ALS (implicitPrefs=True) và so sánh với implicit ALS của DS300 (IE212).

Chạy:
    python -m bigdata.spark_als --k 0 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from bigdata.config import build_spark, resolve_paths
from recsys import config as recsys_config
from recsys.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from recsys.evaluation.metrics import evaluate, jaccard_at_k, truth_from_df
from recsys.evaluation.results import save_run
from recsys.schema import RunConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

log = logging.getLogger(__name__)


def fit_recommend_spark_als(
    spark: "SparkSession",
    train_df: "DataFrame",
    eval_users: list[int],
    k: int = 20,
    rank: int = 64,
    reg_param: float = 0.01,
    alpha: float = 1.0,
    max_iter: int = 15,
    seed: int = 0,
    weight_col: str | None = None,
) -> tuple[dict[int, list[int]], float]:
    """Huấn luyện Spark MLlib ALS (implicitPrefs=True), loại sản phẩm đã xem và trả top-k."""
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    t0 = time.perf_counter()

    # Ánh xạ user_id / product_id sang int32 liên tiếp 0..N-1 (an toàn với int64 hoặc user_id âm)
    user_map = (
        train_df.select("user_id")
        .distinct()
        .orderBy("user_id")
        .withColumn("user_idx", (F.row_number().over(Window.orderBy("user_id")) - 1).cast("int"))
        .cache()
    )
    item_map = (
        train_df.select("product_id")
        .distinct()
        .orderBy("product_id")
        .withColumn("item_idx", (F.row_number().over(Window.orderBy("product_id")) - 1).cast("int"))
        .cache()
    )

    w_col = F.col(weight_col).cast("float") if weight_col and weight_col in train_df.columns else F.lit(1.0).cast("float")
    train_idx = (
        train_df.join(user_map, on="user_id", how="inner")
        .join(F.broadcast(item_map), on="product_id", how="inner")
        .select("user_idx", "item_idx", w_col.alias("rating"))
        .cache()
    )

    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        alpha=alpha,
        implicitPrefs=True,
        coldStartStrategy="drop",
        seed=seed,
    )
    model = als.fit(train_idx)

    if not eval_users:
        return {}, round(time.perf_counter() - t0, 3)

    eval_df = spark.createDataFrame([(int(u),) for u in eval_users], ["user_id"]).join(
        user_map, on="user_id", how="inner"
    ).cache()

    max_seen_row = (
        train_idx.join(F.broadcast(eval_df.select("user_idx")), on="user_idx", how="inner")
        .groupBy("user_idx")
        .count()
        .agg(F.coalesce(F.max("count"), F.lit(0)).alias("m"))
        .first()
    )
    max_seen = int(max_seen_row["m"])
    n_items = int(item_map.count())
    n_cand = min(n_items, k + max_seen)

    raw_recs = model.recommendForUserSubset(eval_df.select("user_idx"), n_cand)
    exploded = (
        raw_recs.select("user_idx", F.explode("recommendations").alias("rec"))
        .select("user_idx", F.col("rec.item_idx").alias("item_idx"), F.col("rec.rating").alias("score"))
    )

    unseen = exploded.join(train_idx, on=["user_idx", "item_idx"], how="left_anti")
    ranked = (
        unseen.join(eval_df, on="user_idx", how="inner")
        .join(F.broadcast(item_map), on="item_idx", how="inner")
        .withColumn(
            "_rk",
            F.row_number().over(
                Window.partitionBy("user_id").orderBy(F.col("score").desc(), F.col("product_id").asc())
            ),
        )
        .filter(F.col("_rk") <= k)
        .orderBy("user_id", "_rk")
    )

    recs: dict[int, list[int]] = {int(u): [] for u in eval_users}
    for row in ranked.select("user_id", "product_id").collect():
        recs[int(row["user_id"])].append(int(row["product_id"]))

    elapsed = round(time.perf_counter() - t0, 3)
    user_map.unpersist()
    item_map.unpersist()
    train_idx.unpersist()
    eval_df.unpersist()
    return recs, elapsed


def compare_with_pilot(
    spark: "SparkSession",
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    ks: tuple[int, ...] = (0, 2),
    star: str = "all",
    seeds: tuple[int, ...] = (0, 1),
) -> dict[str, Any]:
    """Chạy Spark MLlib ALS trên các bản chia Pilot (gốc + lọc) và đối chiếu với implicit ALS."""
    from pyspark.sql import functions as F

    paths = resolve_paths(data_dir=data_dir, results_dir=results_dir)
    splits_root = paths["splits_dir"] if paths["splits_dir"].exists() else recsys_config.SPLITS_DIR
    pair_flags_path = paths["gold_pair_flags"]

    flagged_df = None
    if pair_flags_path.exists():
        pf = spark.read.parquet(str(pair_flags_path))
        flagged_df = pf.filter(F.col("dup_content") | F.col("promo")).select("user_id", "product_id").cache()

    pilot_path = recsys_config.RESULTS_DIR / "pilot" / "pilot_summary.json"
    pilot_summary = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else {}

    report: dict[str, Any] = {"star": star, "seeds": list(seeds), "configs": {}}
    top_k = max(recsys_config.TOP_KS)

    for k in ks:
        key = f"{star}_{k}core"
        sp_dir = splits_root / key
        train_sp = spark.read.parquet(str(sp_dir / "train.parquet")).cache()
        val_pd = pd.read_parquet(sp_dir / "val.parquet")

        warm_users = {int(r["user_id"]) for r in train_sp.select("user_id").distinct().collect()}
        truth = truth_from_df(val_pd[val_pd["user_id"].isin(warm_users)])
        eval_users = sorted(truth)

        if flagged_df is not None:
            train_filt = train_sp.join(flagged_df, on=["user_id", "product_id"], how="left_anti").cache()
        else:
            train_filt = train_sp

        per_seed: dict[int, tuple[pd.DataFrame, dict[int, list[int]], float]] = {}
        for s in seeds:
            recs, dt = fit_recommend_spark_als(spark, train_sp, eval_users, k=top_k, seed=s)
            pu = evaluate(recs, truth, warm_users=warm_users, ks=recsys_config.TOP_KS)
            cfg = RunConfig(stage="pilot", model="spark_als", star=star, k=k, seed=s)
            save_run(
                "pilot",
                cfg.run_id,
                pu,
                {"config": cfg.to_dict(), "n_users": len(pu), "elapsed_s": dt,
                 "mean": {c: float(pu[c].mean()) for c in pu.columns if "@" in c}},
                root=paths["results_dir"],
            )
            per_seed[s] = (pu, recs, dt)

        s0 = seeds[0]
        pu_goc, recs_goc, dt_goc = per_seed[s0]
        recs_loc, dt_loc = fit_recommend_spark_als(spark, train_filt, eval_users, k=top_k, seed=s0)
        pu_loc = evaluate(recs_loc, truth, warm_users=warm_users, ks=recsys_config.TOP_KS)
        cfg_loc = RunConfig(stage="pilot", model="spark_als", star=star, k=k, seed=s0, mitigation="filter")
        save_run(
            "pilot",
            cfg_loc.run_id,
            pu_loc,
            {"config": cfg_loc.to_dict(), "n_users": len(pu_loc), "elapsed_s": dt_loc,
             "mean": {c: float(pu_loc[c].mean()) for c in pu_loc.columns if "@" in c}},
            root=paths["results_dir"],
        )

        jac_filter = np.array([jaccard_at_k(recs_goc[u], recs_loc[u], recsys_config.JACCARD_K) for u in eval_users])
        d_filter = paired_bootstrap(pu_goc["ndcg@10"].to_numpy(), pu_loc["ndcg@10"].to_numpy(), seed=recsys_config.BOOTSTRAP_SEED)

        entry: dict[str, Any] = {
            "n_users": len(eval_users),
            "elapsed_goc_s": dt_goc,
            "elapsed_loc_s": dt_loc,
            "spark_als_ndcg@10_goc": bootstrap_ci(pu_goc["ndcg@10"]).to_dict(),
            "spark_als_ndcg@10_loc": bootstrap_ci(pu_loc["ndcg@10"]).to_dict(),
            "delta_ndcg@10": d_filter.to_dict(),
            "jaccard@10_goc_vs_loc": bootstrap_ci(jac_filter, seed=recsys_config.BOOTSTRAP_SEED).to_dict(),
        }

        if len(seeds) >= 2:
            pu_s1, recs_s1, dt_s1 = per_seed[seeds[1]]
            jac_seed = np.array([jaccard_at_k(recs_goc[u], recs_s1[u], recsys_config.JACCARD_K) for u in eval_users])
            d_seed = paired_bootstrap(pu_goc["ndcg@10"].to_numpy(), pu_s1["ndcg@10"].to_numpy(), seed=recsys_config.BOOTSTRAP_SEED)
            entry["elapsed_seed1_s"] = dt_s1
            entry["seed0_vs_seed1_jaccard@10"] = bootstrap_ci(jac_seed, seed=recsys_config.BOOTSTRAP_SEED).to_dict()
            entry["seed0_vs_seed1_delta_ndcg@10"] = d_seed.to_dict()

        imp_als = pilot_summary.get("configs", {}).get(key, {}).get("models", {}).get("als")
        if imp_als:
            entry["implicit_als_ref_ndcg@10_goc"] = imp_als.get("ndcg@10_goc")
            entry["implicit_als_ref_ndcg@10_loc"] = imp_als.get("ndcg@10_loc")

        report["configs"][key] = entry
        train_sp.unpersist()
        if flagged_df is not None:
            train_filt.unpersist()

    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    out_file = paths["results_dir"] / "spark_als_summary.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Đã ghi %s", out_file)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spark MLlib ALS vs implicit ALS (IE212)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--star", default="all", choices=list(recsys_config.STAR_VARIANTS))
    ap.add_argument("--k", nargs="+", type=int, default=[0, 2], choices=list(recsys_config.KCORE_LEVELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--master", default="local[*]")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = build_spark(app_name="Tiki-IE212-SparkALS", master=args.master)
    try:
        compare_with_pilot(
            spark,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            ks=tuple(args.k),
            star=args.star,
            seeds=tuple(args.seeds),
        )
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
