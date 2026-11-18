"""Vẽ hình cho báo cáo giữa kỳ từ dữ liệu đã xử lý (chỉ số liệu tổng hợp).

Chạy:  python -m recsys.notebooks.figures   → results/figures/*.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from recsys import config

NAVY, TEAL, ORANGE, GREY = "#14213D", "#0F7C8C", "#F2A65A", "#9AA5B4"
OUT = config.RESULTS_DIR / "figures"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#6B7585", "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def _bars(ax, labels, vals, hi, fmt="{:.1f}%"):
    bars = ax.bar(labels, vals, color=[TEAL if i == hi else GREY for i in range(len(vals))])
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                    fontsize=10, color=NAVY, xytext=(0, 2), textcoords="offset points")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def main() -> None:
    inter = pd.read_parquet(config.INTERACTIONS_PATH)
    summ = json.loads(config.PREP_SUMMARY_PATH.read_text(encoding="utf-8"))

    # 1. phân bố sao
    rc = summ["rating_counts"]; tot = sum(rc.values())
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _bars(ax, [f"{k} sao" for k in rc], [100 * v / tot for v in rc.values()], 4)
    ax.set_title(f"Phân bố số sao ({tot:,} review)".replace(",", "."), color=NAVY)
    _save(fig, "h1_phan_bo_sao.png")

    # 2. số tương tác / user
    upu = inter.groupby("user_id").size()
    b = pd.cut(upu, [0, 1, 2, 3, 5, 10, 10**9], labels=["1", "2", "3", "4–5", "6–10", ">10"]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _bars(ax, list(b.index), list(100 * b.values / b.sum()), 0)
    ax.set_xlabel("Số tương tác của một người dùng")
    ax.set_title("Tỉ lệ người dùng theo số tương tác", color=NAVY)
    _save(fig, "h2_tuong_tac_moi_user.png")

    # 3. tương tác theo tháng + mốc cắt
    ts = pd.to_datetime(inter["ts"], unit="s")
    m = ts.dt.to_period("M").value_counts().sort_index()
    m = m[m.index >= pd.Period("2018-01", "M")]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(m.index.to_timestamp(), m.values, color=TEAL, alpha=0.25, lw=0)
    ax.plot(m.index.to_timestamp(), m.values, color=TEAL, lw=1.5)
    for key, lab in (("t_val", "t_val (80%)"), ("t_test", "t_test (90%)")):
        t = pd.to_datetime(summ["cutoffs"][key], unit="s")
        ax.axvline(t, color=ORANGE, lw=1.5, ls="--")
        ax.annotate(f"{lab}\n{t:%d/%m/%Y}", (t, m.max() * 0.92), xytext=(4, 0), textcoords="offset points", fontsize=9, color=NAVY)
    ax.set_ylabel("Tương tác / tháng")
    ax.set_title("Số tương tác theo tháng (từ 2018) và hai mốc chia dữ liệu", color=NAVY)
    _save(fig, "h3_theo_thoi_gian.png")

    # 4. user warm theo mức k-core
    ks = [0, 2, 3, 5]
    warm = [summ["splits"][f"all_{k}core"]["val"]["n_warm_users"] for k in ks]
    tr = [summ["splits"][f"all_{k}core"]["train"]["n_users"] for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    _bars(axes[0], [f"{k}-core" for k in ks], [t / 1000 for t in tr], 0, "{:.1f}k")
    axes[0].set_title("User trong train", color=NAVY)
    _bars(axes[1], [f"{k}-core" for k in ks], [w / 1000 for w in warm], 0, "{:.2f}k")
    axes[1].set_title("User warm ở validation", color=NAVY)
    _save(fig, "h4_kcore_warm.png")

    # 5. kết quả pilot
    p = json.loads((config.RESULTS_DIR / "pilot" / "pilot_summary.json").read_text(encoding="utf-8"))
    cfgs = list(p["configs"])
    names = ["popular", "itemknn", "als"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    w = 0.38
    for ci, cfg in enumerate(cfgs):
        for mi, n in enumerate(names):
            d = p["configs"][cfg]["models"][n]["delta_ndcg@10"]
            x = mi + (ci - 0.5) * w
            col = TEAL if ci == 0 else ORANGE
            axes[0].errorbar(x, d["mean"] * 1000, yerr=[[(d["mean"] - d["lo"]) * 1000], [(d["hi"] - d["mean"]) * 1000]],
                             fmt="o", color=col, capsize=4, label=cfg if mi == 0 else None)
            j = p["configs"][cfg]["models"][n]["jaccard@10"]
            axes[1].bar(x, j["mean"], w * 0.9, color=col, label=cfg if mi == 0 else None)
    axes[0].axhline(0, color=GREY, lw=1)
    axes[0].set_xticks(range(3), ["Popular", "ItemKNN", "ALS"])
    axes[0].set_ylabel("ΔNDCG@10 (×10⁻³)")
    axes[0].set_title("Bỏ tương tác bị gắn cờ: ΔNDCG@10, CI 95%", color=NAVY)
    axes[0].legend(frameon=False, fontsize=9)
    nf = p.get("als_noise_floor", {})
    for ci, cfg in enumerate(cfgs):
        if cfg in nf:
            v = nf[cfg]["jaccard@10_seed0_vs_seed1"]["mean"]
            x = 2 + (ci - 0.5) * w
            axes[1].plot([x - w * 0.45, x + w * 0.45], [v, v], color=NAVY, lw=2, ls=":")
    axes[1].axhline(config.PILOT_JACCARD_THRESHOLD, color=GREY, lw=1, ls="--")
    axes[1].set_xticks(range(3), ["Popular", "ItemKNN", "ALS"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Jaccard@10 gốc vs lọc (chấm: nhiễu seed ALS)", color=NAVY)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, "h5_pilot.png")
    print("Đã vẽ:", sorted(x.name for x in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
