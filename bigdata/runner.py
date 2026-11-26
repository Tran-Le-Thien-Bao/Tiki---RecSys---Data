"""Chạy phân tán danh sách RunConfig trên PySpark (IE212) với cơ chế resume/skip (docs/DO_AN.md mục 4.1).

Chạy:
    python -m bigdata.runner --suite pilot --mode spark
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from bigdata.config import build_spark, resolve_paths
from recsys import config as recsys_config
from recsys.evaluation.metrics import evaluate, truth_from_df
from recsys.evaluation.results import save_run
from recsys.mitigation.flags import flag_reviews, flags_by_pair
from recsys.models.base import Recommender
from recsys.schema import RunConfig

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

log = logging.getLogger(__name__)

_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "popular": ("recsys.models.popular", "Popular"),
    "itemknn": ("recsys.models.itemknn", "ItemKNN"),
    "als": ("recsys.models.als", "ALS"),
    "bpr": ("recsys.models.bpr", "BPR"),
    "mf_bce": ("recsys.models.mf_bce", "MFBCE"),
    "lightfm": ("recsys.models.lightfm_model", "LightFMModel"),
}


def available_models() -> list[str]:
    """Danh sách tên mô hình hiện đã cài trong recsys/models/."""
    out: list[str] = []
    for name, (mod_path, cls_name) in _MODEL_SPECS.items():
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, cls_name):
                out.append(name)
        except ImportError:
            continue
    return out


def get_model(name: str, params: dict[str, Any] | None = None) -> Recommender:
    """Khởi tạo mô hình theo tên và siêu tham số."""
    if name not in _MODEL_SPECS:
        raise ValueError(f"Mô hình không hợp lệ: {name!r}")
    mod_path, cls_name = _MODEL_SPECS[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(**(params or {}))


def pilot_configs(
    ks: tuple[int, ...] = (0, 2),
    star: str = "all",
    seeds: tuple[int, ...] = (0,),
    models: tuple[str, ...] = ("popular", "itemknn", "als"),
) -> list[RunConfig]:
    """Tạo danh sách RunConfig cho bài kiểm tra Pilot (gốc + lọc)."""
    cfgs: list[RunConfig] = []
    for k in ks:
        for m in models:
            for s in seeds:
                for mit in (None, "filter"):
                    cfgs.append(RunConfig(stage="pilot", model=m, star=star, k=k, seed=s, mitigation=mit))
    return cfgs


def stage1_configs(
    models: tuple[str, ...] | None = None,
    stars: tuple[str, ...] = recsys_config.STAR_VARIANTS,
    ks: tuple[int, ...] = recsys_config.KCORE_LEVELS,
    seeds: tuple[int, ...] = recsys_config.SEEDS_STAGE1,
) -> list[RunConfig]:
    """Tạo danh sách RunConfig cho Giai đoạn 1 (lọc theo mô hình đã cài nếu models=None)."""
    chosen = list(models) if models is not None else available_models()
    return [
        RunConfig(stage="stage1", model=m, star=star, k=k, seed=s)
        for m in chosen
        for star in stars
        for k in ks
        for s in seeds
    ]


def load_flagged_pairs(
    gold_pair_flags: Path | None = None,
    reviews_clean_path: Path | None = None,
) -> set[tuple[int, int]]:
    """Đọc tập cặp (user_id, product_id) bị gắn cờ dup_content hoặc promo."""
    if gold_pair_flags is not None and gold_pair_flags.exists():
        pf = pd.read_parquet(gold_pair_flags)
        sub = pf[pf["dup_content"].astype(bool) | pf["promo"].astype(bool)]
        return set(zip(sub["user_id"].astype(int), sub["product_id"].astype(int)))

    rev_path = reviews_clean_path or recsys_config.REVIEWS_CLEAN_PATH
    if not rev_path.exists():
        raise FileNotFoundError(f"Không tìm thấy bảng cờ Gold hay {rev_path}")
    rev = pd.read_parquet(
        rev_path,
        columns=["review_id", "user_id", "product_id", "title", "content", "created_at", "had_phone"],
    )
    pf = flags_by_pair(flag_reviews(rev))
    sub = pf[pf["dup_content"] | pf["promo"]]
    return set(zip(sub["user_id"].astype(int), sub["product_id"].astype(int)))


def _single_blas_ctx():
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(1, "blas")
    except ImportError:
        return contextlib.nullcontext()


def run_single_config(
    cfg: RunConfig,
    splits_dir: str | Path,
    results_root: str | Path,
    flagged_pairs: set[tuple[int, int]] | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Hàm thuần chạy một RunConfig; hỗ trợ bỏ qua nếu đã có summary.json (resume trên Colab)."""
    results_path = Path(results_root)
    summary_file = results_path / cfg.stage / cfg.run_id / "summary.json"
    if skip_existing and summary_file.exists():
        cached = json.loads(summary_file.read_text(encoding="utf-8"))
        cached["skipped"] = True
        return cached

    t0 = time.perf_counter()
    sp_dir = Path(splits_dir) / f"{cfg.star}_{cfg.k}core"
    train = pd.read_parquet(sp_dir / "train.parquet")
    eval_df = pd.read_parquet(sp_dir / f"{cfg.eval_split}.parquet")

    warm = set(train["user_id"].astype(int).unique())
    train_fit = train
    if cfg.mitigation == "filter" and flagged_pairs:
        pairs = list(zip(train["user_id"].astype(int), train["product_id"].astype(int)))
        keep = [p not in flagged_pairs for p in pairs]
        train_fit = train[keep].reset_index(drop=True)

    with _single_blas_ctx():
        model = get_model(cfg.model, cfg.params).fit(train_fit, seed=cfg.seed)

    truth = truth_from_df(eval_df[eval_df["user_id"].isin(warm)])
    users = sorted(truth)
    recs = model.recommend(users, k=max(recsys_config.TOP_KS))
    pu = evaluate(recs, truth, warm_users=warm, ks=recsys_config.TOP_KS)

    summary = {
        "run_id": cfg.run_id,
        "stage": cfg.stage,
        "config": cfg.to_dict(),
        "n_users": int(len(pu)),
        "mean": {c: float(pu[c].mean()) for c in pu.columns if "@" in c},
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "skipped": False,
    }
    save_run(cfg.stage, cfg.run_id, pu, summary, root=results_path)
    return summary


def run_configs_sequential(
    configs: list[RunConfig],
    splits_dir: str | Path,
    results_root: str | Path,
    flagged_pairs: set[tuple[int, int]] | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Chạy tuần tự danh sách RunConfig (dùng làm baseline so sánh)."""
    t0 = time.perf_counter()
    runs = [
        run_single_config(cfg, splits_dir, results_root, flagged_pairs=flagged_pairs, skip_existing=skip_existing)
        for cfg in configs
    ]
    out = {
        "mode": "sequential",
        "n_configs": len(configs),
        "n_skipped": sum(1 for r in runs if r.get("skipped")),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "runs": runs,
    }
    out_path = Path(results_root) / "runner_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def run_configs_spark(
    spark: "SparkSession",
    configs: list[RunConfig],
    splits_dir: str | Path,
    results_root: str | Path,
    flagged_pairs: set[tuple[int, int]] | None = None,
    skip_existing: bool = True,
    num_slices: int | None = None,
) -> dict[str, Any]:
    """Phân phối danh sách RunConfig qua Spark RDD (broadcast flagged_pairs, không capture SparkSession)."""
    t0 = time.perf_counter()
    splits_str = str(Path(splits_dir).resolve())
    results_str = str(Path(results_root).resolve())
    root_str = str(recsys_config.ROOT.resolve())

    bc_flags = spark.sparkContext.broadcast(flagged_pairs)
    slices = num_slices or max(1, min(len(configs), spark.sparkContext.defaultParallelism * 2))

    def _worker(cfg: RunConfig) -> dict[str, Any]:
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from bigdata.runner import run_single_config as _run_one

        return _run_one(
            cfg,
            splits_dir=splits_str,
            results_root=results_str,
            flagged_pairs=bc_flags.value,
            skip_existing=skip_existing,
        )

    try:
        runs = spark.sparkContext.parallelize(configs, numSlices=slices).map(_worker).collect()
    finally:
        bc_flags.unpersist()

    out = {
        "mode": "spark",
        "master": spark.sparkContext.master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "n_configs": len(configs),
        "n_skipped": sum(1 for r in runs if r.get("skipped")),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "runs": runs,
    }
    out_path = Path(results_root) / "runner_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "Spark runner xong %d RunConfig trong %.2fs (skipped=%d)",
        len(configs),
        out["elapsed_s"],
        out["n_skipped"],
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chạy phân tán RunConfig trên Spark (IE212)")
    ap.add_argument("--suite", default="pilot", choices=["pilot", "stage1"])
    ap.add_argument("--mode", default="spark", choices=["spark", "sequential"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-resume", action="store_true", help="Chạy lại dù đã có summary.json")
    ap.add_argument("--master", default="local[*]")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = resolve_paths(data_dir=args.data_dir, results_dir=args.results_dir)
    splits_dir = paths["splits_dir"] if paths["splits_dir"].exists() else recsys_config.SPLITS_DIR
    configs = pilot_configs() if args.suite == "pilot" else stage1_configs()
    flagged = None
    if any(c.mitigation == "filter" for c in configs):
        rev_fallback = paths["silver_reviews"] if paths["silver_reviews"].exists() else recsys_config.REVIEWS_CLEAN_PATH
        flagged = load_flagged_pairs(paths["gold_pair_flags"], rev_fallback)

    if args.mode == "sequential":
        run_configs_sequential(
            configs, splits_dir, paths["results_dir"], flagged_pairs=flagged, skip_existing=not args.no_resume
        )
        return 0

    spark = build_spark(app_name=f"Tiki-IE212-Runner-{args.suite}", master=args.master)
    try:
        run_configs_spark(
            spark,
            configs,
            splits_dir=splits_dir,
            results_root=paths["results_dir"],
            flagged_pairs=flagged,
            skip_existing=not args.no_resume,
        )
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
