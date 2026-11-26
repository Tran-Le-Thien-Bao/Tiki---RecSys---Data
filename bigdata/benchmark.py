"""Benchmark hiệu năng Big Data: Pandas vs PySpark (1x..10x), Spark Tuning & Core Scaling (IE212).

Chạy:
    python -m bigdata.benchmark --scales 1 5 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bigdata.config import build_spark, resolve_paths
from bigdata.etl import spark_flag_reviews, spark_flags_by_pair, spark_kcore, spark_to_interactions
from bigdata.runner import load_flagged_pairs, pilot_configs, run_configs_spark
from recsys import config as recsys_config
from recsys.mitigation.flags import flag_reviews, flags_by_pair
from recsys.prep.build_interactions import to_interactions
from recsys.prep.kcore import kcore

log = logging.getLogger(__name__)

NAVY, TEAL, ORANGE, GREY = "#14213D", "#0F7C8C", "#F2A65A", "#9AA5B4"


def _rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        return 0.0


def generate_scaled_reviews(
    reviews_clean_parquet: Path,
    out_dir: Path,
    scale: int,
    write_jsonl: bool = False,
) -> tuple[Path, Path | None]:
    """Nhân bản bảng reviews_clean theo hệ số scale (giữ phân bố thời gian, sao, nội dung)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / f"reviews_{scale}x.parquet"
    js_path = (out_dir / f"reviews_{scale}x.jsonl") if write_jsonl else None

    base = pd.read_parquet(reviews_clean_parquet)
    if scale == 1:
        if not pq_path.exists():
            base.to_parquet(pq_path, index=False)
        if write_jsonl and js_path is not None and not js_path.exists():
            base.to_json(js_path, orient="records", lines=True, force_ascii=False)
        return pq_path, js_path

    if not pq_path.exists():
        chunks = [base]
        stride = 10_000_000
        for rep in range(1, scale):
            part = base.copy()
            part["review_id"] = part["review_id"] + rep * stride
            part["user_id"] = part["user_id"] + rep * stride
            chunks.append(part)
        scaled = pd.concat(chunks, ignore_index=True)
        scaled.to_parquet(pq_path, index=False)
        if write_jsonl and js_path is not None:
            scaled.to_json(js_path, orient="records", lines=True, force_ascii=False)
        del scaled, chunks
    elif write_jsonl and js_path is not None and not js_path.exists():
        pd.read_parquet(pq_path).to_json(js_path, orient="records", lines=True, force_ascii=False)

    return pq_path, js_path


def _run_pandas_pipeline(pq_path: Path) -> dict[str, Any]:
    tracemalloc.start()
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        rev = pd.read_parquet(pq_path)
        inter = to_interactions(rev)
        rf = flag_reviews(rev)
        pf = flags_by_pair(rf)
        core = kcore(inter, 2)
        dt = round(time.perf_counter() - t0, 3)
        _, peak_bytes = tracemalloc.get_traced_memory()
        rss_after = _rss_mb()
        return {
            "status": "ok",
            "elapsed_s": dt,
            "peak_py_mb": round(peak_bytes / (1024 * 1024), 2),
            "rss_mb": max(rss_before, rss_after),
            "n_rows": int(len(rev)),
            "n_2core": int(len(core)),
            "n_flagged_pairs": int((pf["dup_content"] | pf["promo"]).sum()),
        }
    except MemoryError:
        return {"status": "OOM", "elapsed_s": None, "peak_py_mb": None, "rss_mb": _rss_mb()}
    finally:
        tracemalloc.stop()


def _run_spark_pipeline(
    pq_path: Path,
    master: str = "local[*]",
    shuffle_partitions: int | None = None,
    adaptive: bool = True,
    use_checkpoint: bool = True,
    input_format: str = "parquet",
) -> dict[str, Any]:
    spark = build_spark(
        app_name=f"Tiki-Bench-{input_format}",
        master=master,
        shuffle_partitions=shuffle_partitions,
        adaptive=adaptive,
    )
    t0 = time.perf_counter()
    try:
        df = spark.read.parquet(str(pq_path)) if input_format == "parquet" else spark.read.json(str(pq_path))
        inter = spark_to_interactions(df)
        rf = spark_flag_reviews(df)
        pf = spark_flags_by_pair(rf)
        core = spark_kcore(inter, 2, use_checkpoint=use_checkpoint)
        n_core = int(core.count())
        n_flagged = int(pf.filter("dup_content OR promo").count())
        dt = round(time.perf_counter() - t0, 3)
        return {
            "status": "ok",
            "elapsed_s": dt,
            "rss_mb": _rss_mb(),
            "n_2core": n_core,
            "n_flagged_pairs": n_flagged,
        }
    finally:
        spark.stop()


def benchmark_pandas_vs_spark(
    reviews_clean_parquet: Path,
    work_dir: Path,
    scales: list[int] | None = None,
    master: str = "local[*]",
) -> list[dict[str, Any]]:
    """So sánh thời gian và bộ nhớ giữa Pandas và PySpark ở các quy mô dữ liệu 1x, 5x, 10x."""
    scales = scales or [1, 5, 10]
    rows: list[dict[str, Any]] = []
    for sc in scales:
        pq_path, _ = generate_scaled_reviews(reviews_clean_parquet, work_dir, scale=sc, write_jsonl=False)
        log.info("Benchmark Pandas vs Spark ở mức %dx...", sc)
        pd_res = _run_pandas_pipeline(pq_path)
        sp_res = _run_spark_pipeline(pq_path, master=master, adaptive=True, use_checkpoint=True)
        rows.append({"scale": sc, "pandas": pd_res, "pyspark": sp_res})
    return rows


def benchmark_spark_tuning(
    reviews_clean_parquet: Path,
    work_dir: Path,
    scale: int = 5,
    master: str = "local[*]",
) -> list[dict[str, Any]]:
    """Đo tác động tối ưu Spark: JSONL 200 partitions -> Parquet 200p -> Parquet Tuned + AQE + Checkpoint."""
    pq_path, js_path = generate_scaled_reviews(reviews_clean_parquet, work_dir, scale=scale, write_jsonl=True)
    tuned_parts = max(4, (os.cpu_count() or 2) * 2)
    configs = [
        ("naive_jsonl_200p", js_path, "json", 200, False, False),
        ("parquet_200p", pq_path, "parquet", 200, False, False),
        ("parquet_tuned", pq_path, "parquet", tuned_parts, True, True),
    ]
    rows: list[dict[str, Any]] = []
    for label, path, fmt, parts, adaptive, ckpt in configs:
        log.info("Benchmark Spark tuning: %s (scale=%dx)...", label, scale)
        res = _run_spark_pipeline(
            path,
            master=master,
            shuffle_partitions=parts,
            adaptive=adaptive,
            use_checkpoint=ckpt,
            input_format=fmt,
        )
        rows.append({"config": label, "scale": scale, "shuffle_partitions": parts, "adaptive": adaptive, **res})
    return rows


def benchmark_core_scaling(
    splits_dir: Path,
    work_dir: Path,
    flagged_pairs: set[tuple[int, int]] | None,
    cores_list: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Đo speedup chạy phân tán RunConfig khi tăng số CPU cores (local[1], local[2], local[4])."""
    max_c = os.cpu_count() or 2
    if cores_list is None:
        cores_list = [c for c in (1, 2, 4) if c <= max_c] or [1]
    cfgs = pilot_configs(ks=(0,), seeds=(0,))
    rows: list[dict[str, Any]] = []
    t_base: float | None = None
    for c in cores_list:
        spark = build_spark(app_name=f"Tiki-Scale-c{c}", master=f"local[{c}]", shuffle_partitions=max(4, c * 2))
        try:
            out = run_configs_spark(
                spark,
                cfgs,
                splits_dir=splits_dir,
                results_root=work_dir / f"scaling_c{c}",
                flagged_pairs=flagged_pairs,
                skip_existing=False,
                num_slices=max(c, len(cfgs)),
            )
            dt = float(out["elapsed_s"])
            if t_base is None:
                t_base = dt
            speedup = round(t_base / dt, 3) if dt > 0 else 0.0
            rows.append(
                {
                    "cores": c,
                    "n_configs": len(cfgs),
                    "elapsed_s": dt,
                    "speedup": speedup,
                    "efficiency": round(speedup / c, 3),
                }
            )
        finally:
            spark.stop()
    return rows


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7585")
    ax.spines["bottom"].set_color("#6B7585")
    ax.tick_params(colors=NAVY)


def plot_benchmarks(summary: dict[str, Any], out_dir: Path) -> list[Path]:
    """Vẽ 3 biểu đồ PNG cho báo cáo IE212 từ kết quả benchmark."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelcolor": NAVY})
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # 1. Pandas vs PySpark
    pvs = summary.get("pandas_vs_spark") or []
    if pvs:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        labels = [f"{r['scale']}x" for r in pvs]
        x = np.arange(len(labels))
        w = 0.35
        pd_t = [r["pandas"].get("elapsed_s") or 0.0 for r in pvs]
        sp_t = [r["pyspark"].get("elapsed_s") or 0.0 for r in pvs]
        pd_m = [r["pandas"].get("peak_py_mb") or 0.0 for r in pvs]
        sp_m = [r["pyspark"].get("rss_mb") or 0.0 for r in pvs]

        axes[0].bar(x - w / 2, pd_t, width=w, label="Pandas", color=ORANGE)
        axes[0].bar(x + w / 2, sp_t, width=w, label="PySpark", color=TEAL)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Thời gian (giây)")
        axes[0].set_title("Thời gian ETL + Flags + 2-core", color=NAVY)
        axes[0].legend(frameon=False)
        _style_ax(axes[0])

        axes[1].bar(x - w / 2, pd_m, width=w, label="Pandas (peak Python)", color=ORANGE)
        axes[1].bar(x + w / 2, sp_m, width=w, label="PySpark (RSS)", color=TEAL)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Bộ nhớ (MB)")
        axes[1].set_title("Mức sử dụng bộ nhớ theo quy mô", color=NAVY)
        axes[1].legend(frameon=False)
        _style_ax(axes[1])

        fig.tight_layout()
        p1 = out_dir / "bd1_pandas_vs_spark.png"
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        created.append(p1)

    # 2. Spark Tuning
    tun = summary.get("spark_tuning") or []
    if tun:
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        names = [r["config"] for r in tun]
        vals = [r.get("elapsed_s") or 0.0 for r in tun]
        colors = [GREY, ORANGE, TEAL][: len(names)]
        bars = ax.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}s", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", color=NAVY)
        ax.set_ylabel("Thời gian (giây)")
        ax.set_title("Tối ưu Spark: JSONL 200p vs Parquet 200p vs Parquet Tuned", color=NAVY)
        _style_ax(ax)
        fig.tight_layout()
        p2 = out_dir / "bd2_spark_tuning.png"
        fig.savefig(p2, dpi=200)
        plt.close(fig)
        created.append(p2)

    # 3. Core Scaling
    cs = summary.get("core_scaling") or []
    if cs:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        cores = [r["cores"] for r in cs]
        times = [r["elapsed_s"] for r in cs]
        speedups = [r["speedup"] for r in cs]

        axes[0].plot(cores, times, marker="o", color=TEAL, lw=2)
        axes[0].set_xticks(cores)
        axes[0].set_xlabel("Số CPU cores")
        axes[0].set_ylabel("Thời gian (giây)")
        axes[0].set_title("Thời gian chạy RunConfig theo số cores", color=NAVY)
        _style_ax(axes[0])

        axes[1].plot(cores, speedups, marker="o", color=TEAL, lw=2, label="Thực tế")
        axes[1].plot(cores, cores, ls="--", color=GREY, lw=1.2, label="Lý tưởng (tuyến tính)")
        axes[1].set_xticks(cores)
        axes[1].set_xlabel("Số CPU cores")
        axes[1].set_ylabel("Speedup (T₁ / Tₙ)")
        axes[1].set_title("Hệ số tăng tốc (Strong Scaling)", color=NAVY)
        axes[1].legend(frameon=False)
        _style_ax(axes[1])

        fig.tight_layout()
        p3 = out_dir / "bd3_core_scaling.png"
        fig.savefig(p3, dpi=200)
        plt.close(fig)
        created.append(p3)

    return created


def run_all_benchmarks(
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    scales: list[int] | None = None,
    tuning_scale: int | None = None,
    master: str = "local[*]",
) -> dict[str, Any]:
    """Chạy trọn bộ benchmark IE212 và xuất JSON + 3 biểu đồ PNG."""
    paths = resolve_paths(data_dir=data_dir, results_dir=results_dir)
    rev_pq = paths["silver_reviews"] if paths["silver_reviews"].exists() else recsys_config.REVIEWS_CLEAN_PATH
    splits_dir = paths["splits_dir"] if paths["splits_dir"].exists() else recsys_config.SPLITS_DIR
    w_dir = Path(work_dir) if work_dir is not None else (paths["data_dir"] / "bigdata" / "tmp_bench")

    scales = scales or [1, 5, 10]
    t_scale = tuning_scale if tuning_scale is not None else (5 if 5 in scales else scales[-1])
    flagged = load_flagged_pairs(paths["gold_pair_flags"], rev_pq)

    summary = {
        "cpu_count": os.cpu_count(),
        "scales": scales,
        "pandas_vs_spark": benchmark_pandas_vs_spark(rev_pq, w_dir, scales=scales, master=master),
        "spark_tuning": benchmark_spark_tuning(rev_pq, w_dir, scale=t_scale, master=master),
        "core_scaling": benchmark_core_scaling(splits_dir, w_dir, flagged_pairs=flagged),
    }

    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    out_json = paths["results_dir"] / "benchmark_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figs = plot_benchmarks(summary, paths["figures_dir"])
    log.info("Đã ghi %s và %d biểu đồ tại %s", out_json, len(figs), paths["figures_dir"])
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark Big Data (IE212)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--work-dir", default=None, help="Thư mục tạm cục bộ (/content/tmp_bench trên Colab)")
    ap.add_argument("--scales", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--tuning-scale", type=int, default=None)
    ap.add_argument("--master", default="local[*]")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_all_benchmarks(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        work_dir=args.work_dir,
        scales=args.scales,
        tuning_scale=args.tuning_scale,
        master=args.master,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
