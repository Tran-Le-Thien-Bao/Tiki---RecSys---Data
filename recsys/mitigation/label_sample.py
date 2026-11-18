"""Gán nhãn tay để tính λ (docs/DO_AN.md mục 4.5.5).

1) Lấy mẫu:   python -m recsys.mitigation.label_sample sample
   → data/labels/mau_gan_nhan.csv (mở bằng Excel; KHÔNG commit, KHÔNG gửi ra ngoài nhóm)
   Điền cột `nhan`: 1 = rác/không đáng tin, 0 = bình thường. Cột `nguoi_gan`: tên người gán.
2) Tính λ:     python -m recsys.mitigation.label_sample lambda
   → in precision từng dấu hiệu (số tổng hợp) để chép vào config.TRUST_LAMBDA.
3) Gán chéo:   python -m recsys.mitigation.label_sample cheo   → data/labels/mau_gan_nhan_cheo.csv (30 dòng/dấu hiệu)
   Độ đồng thuận: python -m recsys.mitigation.label_sample kappa
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from recsys import config
from recsys.mitigation.flags import flag_reviews

SAMPLE_PATH = config.LABELS_DIR / "mau_gan_nhan.csv"


def make_sample(n_per_flag: int = 100, seed: int = 2026) -> pd.DataFrame:
    """Mỗi dấu hiệu lấy ngẫu nhiên tối đa n review bị gắn cờ (seed cố định); một review có thể thuộc nhiều dấu hiệu."""
    rev = pd.read_parquet(config.REVIEWS_CLEAN_PATH,
                          columns=["review_id", "user_id", "product_id", "rating", "title", "content", "created_at", "had_phone"])
    rf = flag_reviews(rev)
    rows = []
    for flag in config.FLAG_NAMES:
        ids = rf.loc[rf[flag], "review_id"]
        pick = ids.sample(n=min(n_per_flag, len(ids)), random_state=seed)
        rows.append(pd.DataFrame({"dau_hieu": flag, "review_id": pick.to_numpy()}))
    s = pd.concat(rows, ignore_index=True).merge(rev[["review_id", "rating", "title", "content"]], on="review_id")
    s["nhan"] = ""
    s["nguoi_gan"] = ""
    return s.sample(frac=1, random_state=seed).reset_index(drop=True)   # trộn để người gán không đoán theo thứ tự


def compute_lambda(path=SAMPLE_PATH) -> dict[str, dict]:
    """precision = tỉ lệ nhãn 1 trong các review đã gán của từng dấu hiệu."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["nhan"].astype(str).str.strip().isin(["0", "1"])]
    out = {}
    for flag, g in df.groupby("dau_hieu"):
        y = g["nhan"].astype(int)
        out[flag] = {"n_da_gan": int(len(g)), "precision": round(float(y.mean()), 3)}
    return out


CROSS_PATH = config.LABELS_DIR / "mau_gan_nhan_cheo.csv"


def make_cross(n_per_flag: int = 30) -> pd.DataFrame:
    """Bản gán nhãn chéo: n dòng đầu của mỗi dấu hiệu trong file mẫu, cột nhãn để trống."""
    s = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    c = s.groupby("dau_hieu", group_keys=False).head(n_per_flag).copy()
    c["nhan"] = ""
    c["nguoi_gan"] = ""
    return c


def cohen_kappa(path_a=SAMPLE_PATH, path_b=CROSS_PATH) -> dict:
    """Độ đồng thuận giữa hai người gán trên các dòng cả hai đã gán (0/1)."""
    a = pd.read_csv(path_a, encoding="utf-8-sig")
    b = pd.read_csv(path_b, encoding="utf-8-sig")
    m = a.merge(b, on=["dau_hieu", "review_id"], suffixes=("_a", "_b"))
    ok = m["nhan_a"].astype(str).str.strip().isin(["0", "1"]) & m["nhan_b"].astype(str).str.strip().isin(["0", "1"])
    m = m[ok]
    if m.empty:
        return {"n": 0}
    ya, yb = m["nhan_a"].astype(int), m["nhan_b"].astype(int)
    po = float((ya == yb).mean())
    pe = float(ya.mean() * yb.mean() + (1 - ya.mean()) * (1 - yb.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": int(len(m)), "dong_y": round(po, 3), "kappa": round(kappa, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["sample", "lambda", "cheo", "kappa"])
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args(argv)
    for s_ in (sys.stdout, sys.stderr):
        if hasattr(s_, "reconfigure"):
            s_.reconfigure(encoding="utf-8")
    if a.cmd == "sample":
        if SAMPLE_PATH.exists():
            print(f"Đã có {SAMPLE_PATH} — không ghi đè (xóa file thủ công nếu muốn lấy lại mẫu).")
            return 1
        s = make_sample(a.n)
        config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
        s.to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
        print(f"Đã ghi {len(s)} dòng vào {SAMPLE_PATH}; số dòng theo dấu hiệu: {s['dau_hieu'].value_counts().to_dict()}")
    elif a.cmd == "cheo":
        if CROSS_PATH.exists():
            print(f"Đã có {CROSS_PATH} — không ghi đè.")
            return 1
        c = make_cross()
        c.to_csv(CROSS_PATH, index=False, encoding="utf-8-sig")
        print(f"Đã ghi {len(c)} dòng vào {CROSS_PATH}")
    elif a.cmd == "kappa":
        print(cohen_kappa())
    else:
        print(compute_lambda())
    return 0


if __name__ == "__main__":
    sys.exit(main())
