"""Pilot (docs/DO_AN.md mục 4.6): Popular, ItemKNN, ALS trên train gốc và train đã bỏ tương tác bị gắn cờ
(dup_content ∪ promo). Val/test giữ nguyên. Báo Jaccard@10 và ΔNDCG@10 kèm khoảng tin cậy, đánh giá trên
validation, user warm (warm theo train GỐC để hai phiên bản so trên cùng tập user).

Quy tắc định trước: giữ RQ3 nếu khoảng tin cậy của ΔNDCG@10 không chứa 0 HOẶC Jaccard@10 trung bình < 0,9.

Chạy:  python -m recsys.experiments.pilot --k 0 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from recsys import config
from recsys.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from recsys.evaluation.metrics import evaluate, jaccard_at_k, truth_from_df
from recsys.evaluation.results import save_run
from recsys.mitigation.flags import flag_reviews, flags_by_pair, summarize
from recsys.models.als import ALS
from recsys.models.itemknn import ItemKNN
from recsys.models.popular import Popular
from recsys.schema import RunConfig

log = logging.getLogger(__name__)
MODELS = {"popular": Popular, "itemknn": ItemKNN, "als": ALS}


def load_split(star: str, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = config.split_dir(star, k)
    return pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "val.parquet")


def pair_flags() -> tuple[pd.DataFrame, dict]:
    rev = pd.read_parquet(config.REVIEWS_CLEAN_PATH,
                          columns=["review_id", "user_id", "product_id", "title", "content", "created_at", "had_phone"])
    rf = flag_reviews(rev)
    pf = flags_by_pair(rf)
    return pf, summarize(rf, pf)


def run_config(cfg: RunConfig, train: pd.DataFrame, val: pd.DataFrame, warm: set[int]) -> tuple[pd.DataFrame, dict]:
    """Một cấu hình: huấn luyện, gợi ý top-20 cho user warm có trong val, trả metric theo user + gợi ý."""
    model = MODELS[cfg.model](**cfg.params).fit(train, seed=cfg.seed)
    truth = truth_from_df(val[val["user_id"].isin(warm)])
    users = sorted(truth)
    recs = model.recommend(users, k=max(config.TOP_KS))
    return evaluate(recs, truth, warm_users=warm, ks=config.TOP_KS), recs


def run(ks: list[int], star: str = "all", seed: int = 0) -> dict:
    t0 = time.time()
    pf, flag_summary = pair_flags()
    flagged = pf[pf["dup_content"] | pf["promo"]][["user_id", "product_id"]]
    report: dict = {"flags": flag_summary, "rule": "giữ RQ3 nếu CI(ΔNDCG@10) không chứa 0 hoặc Jaccard@10 TB < 0,9",
                    "eval_split": "val", "configs": {}}
    for k in ks:
        train, val = load_split(star, k)
        drop = train.merge(flagged, on=["user_id", "product_id"], how="left", indicator=True)["_merge"] == "both"
        train_f = train[~drop.to_numpy()].reset_index(drop=True)
        warm = set(train["user_id"].unique())
        lost_users = len(warm - set(train_f["user_id"].unique()))
        key = f"{star}_{k}core"
        report["configs"][key] = {"train_pairs": int(len(train)), "train_pairs_removed": int(drop.sum()),
                                  "warm_users_lost_all_history": int(lost_users), "models": {}}
        log.info("%s: bỏ %d / %d cặp train (%.2f%%)", key, drop.sum(), len(train), 100 * drop.mean())
        for name in MODELS:
            out = {}
            per = {}
            for variant, tr in (("goc", train), ("loc", train_f)):
                cfg = RunConfig(stage="pilot", model=name, star=star, k=k, seed=seed,
                                mitigation=None if variant == "goc" else "filter")
                t = time.time()
                pu, recs = run_config(cfg, tr, val, warm)
                save_run("pilot", cfg.run_id, pu, {"config": cfg.to_dict(), "n_users": len(pu),
                                                   "mean": {c: float(pu[c].mean()) for c in pu.columns if "@" in c}})
                per[variant] = (pu, recs)
                log.info("  %-8s %-3s NDCG@10=%.4f  (%.1fs)", name, variant, pu["ndcg@10"].mean(), time.time() - t)
            (pa, ra), (pb, rb) = per["goc"], per["loc"]
            assert (pa["user_hash"].to_numpy() == pb["user_hash"].to_numpy()).all()
            users = sorted(ra)
            jac = np.array([jaccard_at_k(ra[u], rb[u], config.JACCARD_K) for u in users])
            d = paired_bootstrap(pa["ndcg@10"].to_numpy(), pb["ndcg@10"].to_numpy(), seed=config.BOOTSTRAP_SEED)
            j = bootstrap_ci(jac, seed=config.BOOTSTRAP_SEED)
            keep = d.distinguishable or j.mean < config.PILOT_JACCARD_THRESHOLD
            out = {"n_users": int(len(users)),
                   "ndcg@10_goc": bootstrap_ci(pa["ndcg@10"]).to_dict(),
                   "ndcg@10_loc": bootstrap_ci(pb["ndcg@10"]).to_dict(),
                   "delta_ndcg@10": d.to_dict(), "jaccard@10": j.to_dict(), "rule_keep_rq3": bool(keep)}
            report["configs"][key]["models"][name] = out
            log.info("  %-8s ΔNDCG@10=%+.5f [%+.5f, %+.5f]  Jaccard@10=%.4f [%.4f, %.4f]  → giữ RQ3: %s",
                     name, d.mean, d.lo, d.hi, j.mean, j.lo, j.hi, keep)
    report["elapsed_s"] = round(time.time() - t0, 1)
    out_path = config.RESULTS_DIR / "pilot" / "pilot_summary.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Đã ghi %s", out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pilot RQ3")
    ap.add_argument("--k", nargs="+", type=int, default=[0], choices=list(config.KCORE_LEVELS))
    ap.add_argument("--star", default="all", choices=list(config.STAR_VARIANTS))
    args = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.k, args.star)
    return 0


if __name__ == "__main__":
    sys.exit(main())
