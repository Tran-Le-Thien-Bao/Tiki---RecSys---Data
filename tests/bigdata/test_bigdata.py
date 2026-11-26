"""Test cho phần Big Data (IE212): unit test chạy mọi máy + integration test PySpark (skip nếu chưa cài)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bigdata.benchmark import plot_benchmarks
from bigdata.config import resolve_paths
from bigdata.etl import compare_parity
from bigdata.runner import pilot_configs, run_single_config, stage1_configs
from recsys import config as recsys_config
from recsys.mitigation.flags import flag_reviews, flags_by_pair, summarize
from recsys.prep.build_interactions import clean_reviews, to_interactions
from recsys.prep.kcore import kcore
from recsys.schema import RunConfig


# ---------------------------------------------------------------- Tests không cần PySpark
def test_resolve_paths_default_and_custom(tmp_path):
    p_def = resolve_paths()
    assert p_def["data_dir"] == recsys_config.DATA_DIR
    assert p_def["raw_reviews"] == recsys_config.RAW_REVIEWS

    p_custom = resolve_paths(data_dir=tmp_path / "d", results_dir=tmp_path / "r")
    assert p_custom["silver_reviews"] == tmp_path / "d" / "bigdata" / "silver" / "reviews_clean.parquet"
    assert p_custom["gold_pair_flags"] == tmp_path / "d" / "bigdata" / "gold" / "pair_flags.parquet"
    assert p_custom["figures_dir"] == tmp_path / "r" / "figures"


def test_compare_parity_match_and_mismatch():
    prep = {
        "counts": {"reviews_rows": 100, "pairs": 90},
        "rating_counts": {"5": 80, "4": 10},
        "cutoffs": {"t_val": 1000, "t_test": 2000},
        "n_reviews_with_phone_before_scrub": 2,
        "splits": {"all_0core": {"train": {"n_interactions": 70, "n_users": 50, "n_items": 10}}},
    }
    spark_ok = {
        "counts": {"reviews_rows": 100, "pairs": 90},
        "rating_counts": {"5": 80, "4": 10},
        "cutoffs": {"t_val": 1000, "t_test": 2000},
        "n_reviews_with_phone_before_scrub": 2,
        "splits": {"all_0core": {"train": {"n_interactions": 70, "n_users": 50, "n_items": 10}}},
    }
    flags = {"n_reviews": 100, "n_pairs": 90, "pairs_dup_or_promo": 3}
    pilot = {"flags": {"n_reviews": 100, "n_pairs": 90, "pairs_dup_or_promo": 3}}

    ok = compare_parity(spark_ok, prep, flags, pilot)
    assert ok["all_match"] is True
    assert ok["mismatches"] == []

    bad = json.loads(json.dumps(spark_ok))
    bad["counts"]["pairs"] = 89
    res_bad = compare_parity(bad, prep, flags, pilot)
    assert res_bad["all_match"] is False
    assert any("counts.pairs" in m for m in res_bad["mismatches"])


def test_runner_configs_and_single_run_resume(toy_interactions, tmp_path):
    assert len(pilot_configs()) == 6 * 2  # 3 models * 2 ks * 2 mitigations
    assert len(stage1_configs(models=("popular",), ks=(0, 2))) == 2 * 2 * 3

    sp_dir = tmp_path / "splits" / "all_0core"
    sp_dir.mkdir(parents=True)
    # 7 dòng đầu làm train (users 1,2,3,4; items 10..14), tạo val giả cho user 1 và 4
    train = toy_interactions[toy_interactions["user_id"] > 0].copy()
    val = pd.DataFrame({
        "user_id": pd.array([1, 4], dtype="int64"),
        "product_id": pd.array([14, 11], dtype="int64"),
        "ts": pd.array([200, 201], dtype="int64"),
        "rating": pd.array([5, 5], dtype="int8"),
        "n_reviews": pd.array([1, 1], dtype="int16"),
        "is_fake": pd.array([False, False], dtype="bool"),
    })
    train.to_parquet(sp_dir / "train.parquet", index=False)
    val.to_parquet(sp_dir / "val.parquet", index=False)

    cfg = RunConfig(stage="pilot", model="popular", star="all", k=0, seed=0)
    res_root = tmp_path / "res"
    out1 = run_single_config(cfg, tmp_path / "splits", res_root, skip_existing=True)
    assert out1["skipped"] is False
    assert (res_root / "pilot" / cfg.run_id / "per_user.parquet").exists()
    assert (res_root / "pilot" / cfg.run_id / "summary.json").exists()

    out2 = run_single_config(cfg, tmp_path / "splits", res_root, skip_existing=True)
    assert out2["skipped"] is True
    assert out2["mean"] == out1["mean"]


def test_plot_benchmarks_creates_figures(tmp_path):
    summary = {
        "pandas_vs_spark": [
            {"scale": 1, "pandas": {"elapsed_s": 1.2, "peak_py_mb": 120.0}, "pyspark": {"elapsed_s": 3.5, "rss_mb": 400.0}},
            {"scale": 5, "pandas": {"elapsed_s": 9.8, "peak_py_mb": 650.0}, "pyspark": {"elapsed_s": 7.1, "rss_mb": 520.0}},
        ],
        "spark_tuning": [
            {"config": "naive_jsonl_200p", "elapsed_s": 12.4},
            {"config": "parquet_200p", "elapsed_s": 7.8},
            {"config": "parquet_tuned", "elapsed_s": 4.1},
        ],
        "core_scaling": [
            {"cores": 1, "elapsed_s": 10.0, "speedup": 1.0, "efficiency": 1.0},
            {"cores": 2, "elapsed_s": 5.5, "speedup": 1.82, "efficiency": 0.91},
        ],
    }
    figs = plot_benchmarks(summary, tmp_path / "figs")
    names = {p.name for p in figs}
    assert names == {"bd1_pandas_vs_spark.png", "bd2_spark_tuning.png", "bd3_core_scaling.png"}
    assert all(p.stat().st_size > 0 for p in figs)


def test_colab_notebook_valid_and_clean():
    nb_path = Path("bigdata/notebooks/ie212_colab.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 8
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


# ---------------------------------------------------------------- Tests có PySpark (skip nếu chưa cài)
def test_spark_etl_and_flags_parity_with_pandas(tmp_path):
    pyspark = pytest.importorskip("pyspark")
    assert pyspark is not None
    from bigdata.config import build_spark
    from bigdata.etl import (
        spark_clean_reviews,
        spark_flag_reviews,
        spark_flags_by_pair,
        spark_kcore,
        spark_summarize_flags,
        spark_to_interactions,
    )

    dup_text = "  Sản   phẩm rất tốt,\n\tđóng gói cẩn thận lắm luôn  "
    rows = [
        # Trùng review_id=1 + chứa SĐT trước scrub
        {"review_id": 1, "user_id": 10, "product_id": 100, "rating": 5, "title": "Tốt",
         "content": "Liên hệ 0912345678 nhé", "thank_count": 1, "created_at": 1000, "is_verified_purchase": True},
        {"review_id": 1, "user_id": 10, "product_id": 100, "rating": 1, "title": "Trùng",
         "content": "Bản ghi trùng sẽ bị loại", "thank_count": 0, "created_at": 900, "is_verified_purchase": True},
        # 5 bản ghi nội dung dài lặp >= 5 lần (dup_content)
        {"review_id": 2, "user_id": 11, "product_id": 100, "rating": 5, "title": "", "content": dup_text,
         "thank_count": 0, "created_at": 1100, "is_verified_purchase": True},
        {"review_id": 3, "user_id": 12, "product_id": 101, "rating": 5, "title": "", "content": dup_text.strip(),
         "thank_count": 0, "created_at": 1200, "is_verified_purchase": True},
        {"review_id": 4, "user_id": 13, "product_id": 102, "rating": 5, "title": "", "content": dup_text.lower(),
         "thank_count": 0, "created_at": 1300, "is_verified_purchase": True},
        {"review_id": 5, "user_id": 14, "product_id": 103, "rating": 5, "title": "", "content": dup_text,
         "thank_count": 0, "created_at": 1400, "is_verified_purchase": True},
        {"review_id": 6, "user_id": 15, "product_id": 104, "rating": 5, "title": "", "content": dup_text,
         "thank_count": 0, "created_at": 1500, "is_verified_purchase": True},
        # User 20 review 5 sản phẩm khác nhau trong vòng 50 phút (burst) + từ khóa promo ở 1 review
        {"review_id": 7, "user_id": 20, "product_id": 100, "rating": 5, "title": "Săn sale", "content": "rẻ",
         "thank_count": 0, "created_at": 2000, "is_verified_purchase": True},
        {"review_id": 8, "user_id": 20, "product_id": 101, "rating": 5, "title": "", "content": "ok",
         "thank_count": 0, "created_at": 2500, "is_verified_purchase": True},
        {"review_id": 9, "user_id": 20, "product_id": 102, "rating": 5, "title": "", "content": "ok",
         "thank_count": 0, "created_at": 3000, "is_verified_purchase": True},
        {"review_id": 10, "user_id": 20, "product_id": 103, "rating": 5, "title": "", "content": "ok",
         "thank_count": 0, "created_at": 3500, "is_verified_purchase": True},
        {"review_id": 11, "user_id": 20, "product_id": 104, "rating": 5, "title": "", "content": "ok",
         "thank_count": 0, "created_at": 4000, "is_verified_purchase": True},
        # Cùng user=10, cùng product=101 -> 2 review (burst same product)
        {"review_id": 12, "user_id": 10, "product_id": 101, "rating": 5, "title": "", "content": "mua lần 1",
         "thank_count": 0, "created_at": 5000, "is_verified_purchase": True},
        {"review_id": 13, "user_id": 10, "product_id": 101, "rating": 4, "title": "", "content": "mua lần 2",
         "thank_count": 0, "created_at": 9000, "is_verified_purchase": True},
    ]
    jsonl_path = tmp_path / "reviews.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    # Pandas baseline
    raw_pd = pd.read_json(jsonl_path, lines=True, convert_dates=False, dtype=False)
    rev_pd = clean_reviews(raw_pd)
    inter_pd = to_interactions(rev_pd)
    rf_pd = flag_reviews(rev_pd)
    pf_pd = flags_by_pair(rf_pd)
    sum_pd = summarize(rf_pd, pf_pd)
    k2_pd = kcore(inter_pd, 2)

    spark = build_spark("Test-IE212-Parity", master="local[2]", shuffle_partitions=4)
    try:
        raw_sp = spark.read.json(str(jsonl_path))
        rev_sp = spark_clean_reviews(raw_sp)
        inter_sp = spark_to_interactions(rev_sp)
        rf_sp = spark_flag_reviews(rev_sp)
        pf_sp = spark_flags_by_pair(rf_sp)
        sum_sp = spark_summarize_flags(rf_sp, pf_sp)
        k2_sp = spark_kcore(inter_sp, 2, use_checkpoint=False)

        assert sum_sp == sum_pd
        assert len(k2_sp.toPandas()) == len(k2_pd)
        sp_rf = rf_sp.toPandas().sort_values("review_id").reset_index(drop=True)
        pd_rf = rf_pd.sort_values("review_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(sp_rf, pd_rf)
    finally:
        spark.stop()


def test_spark_als_on_toy_data(toy_interactions):
    pytest.importorskip("pyspark")
    from bigdata.config import build_spark
    from bigdata.spark_als import fit_recommend_spark_als

    spark = build_spark("Test-IE212-SparkALS", master="local[2]", shuffle_partitions=4)
    try:
        train_df = spark.createDataFrame(toy_interactions[toy_interactions["user_id"] > 0])
        recs1, _ = fit_recommend_spark_als(spark, train_df, eval_users=[1, 4], k=2, rank=4, max_iter=5, seed=0)
        recs2, _ = fit_recommend_spark_als(spark, train_df, eval_users=[1, 4], k=2, rank=4, max_iter=5, seed=0)
        assert recs1 == recs2
        # user 1 đã xem {10, 11, 12} -> không gợi ý lại
        assert set(recs1[1]).isdisjoint({10, 11, 12})
        assert len(recs1[4]) == 2
    finally:
        spark.stop()
