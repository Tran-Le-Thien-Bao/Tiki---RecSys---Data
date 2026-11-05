"""Test prep trên dữ liệu giả nhỏ (không đọc data/raw)."""
from __future__ import annotations

import pandas as pd
import pytest

from recsys.prep.build_interactions import clean_reviews, compare_expected, to_interactions
from recsys.prep.kcore import kcore
from recsys.prep.split import apply_star, make_split, temporal_split, time_cutoffs


def _raw_reviews() -> pd.DataFrame:
    rows = [
        # review_id, user, product, rating, title, content, created_at
        (1, 1, 10, 5, "Tốt", "gọi 0912345678 nhé", 300),
        (1, 1, 10, 5, "Tốt", "gọi 0912345678 nhé", 300),     # trùng review_id
        (2, 1, 10, 2, "", "lần hai", 100),                      # cùng cặp, sớm hơn → được giữ
        (3, 2, 10, 4, "", "mail a@b.com", 200),
        (4, 2, 11, 5, "", "", 200),
        (5, 2, 11, 3, "", "", 200),                             # hòa created_at → review_id 4 thắng
    ]
    df = pd.DataFrame(rows, columns=["review_id", "user_id", "product_id", "rating", "title", "content", "created_at"])
    df["thank_count"] = 0
    df["is_verified_purchase"] = True
    return df


def test_clean_reviews_dedupe_and_scrub():
    r = clean_reviews(_raw_reviews())
    assert len(r) == 5
    row1 = r[r.review_id == 1].iloc[0]
    assert row1.content == "gọi [sdt] nhé" and row1.had_phone and not row1.had_email
    row3 = r[r.review_id == 3].iloc[0]
    assert row3.content == "mail [email]" and row3.had_email


def test_to_interactions_earliest_review():
    inter = to_interactions(clean_reviews(_raw_reviews())).set_index(["user_id", "product_id"])
    assert len(inter) == 3
    assert inter.loc[(1, 10)].tolist()[:3] == [100, 2, 2]     # ts, rating, n_reviews
    assert inter.loc[(2, 11), "rating"] == 5 and inter.loc[(2, 11), "n_reviews"] == 2
    assert not inter["is_fake"].any()


def test_kcore_cascade():
    # u3 chỉ có 1 tương tác → bị loại → sp C còn 1 → bị loại → u2 còn 1 (B) ...
    pairs = [(1, "A"), (1, "B"), (2, "A"), (2, "B"), (2, "C"), (3, "C")]
    df = pd.DataFrame(pairs, columns=["user_id", "product_id"])
    out = kcore(df, 2)
    assert set(map(tuple, out.to_numpy())) == {(1, "A"), (1, "B"), (2, "A"), (2, "B")}
    assert len(kcore(df, 0)) == 6
    assert kcore(df, 3).empty


def _inter(rows):
    df = pd.DataFrame(rows, columns=["user_id", "product_id", "ts", "rating"]).astype(
        {"user_id": "int64", "product_id": "int64", "ts": "int64", "rating": "int8"})
    df["n_reviews"] = pd.Series([1] * len(df), dtype="int16")
    df["is_fake"] = False
    return df


def test_time_cutoffs_and_split():
    df = _inter([(u, 100 + u, t, 5) for u, t in zip(range(10), range(0, 100, 10))])
    t_val, t_test = time_cutoffs(df, 0.8, 0.9)
    assert (t_val, t_test) == (72, 81)
    tr, va, te = temporal_split(df, t_val, t_test)
    assert tr.ts.max() < t_val <= va.ts.min() and va.ts.max() < t_test <= te.ts.min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_apply_star():
    df = _inter([(1, 1, 0, 5), (1, 2, 1, 3), (2, 1, 2, 4)])
    assert len(apply_star(df, "all")) == 3
    assert apply_star(df, "pos45")["rating"].min() >= 4
    with pytest.raises(ValueError):
        apply_star(df, "x")


def test_make_split_warm_cold_and_item_filter():
    df = _inter([
        (1, 10, 1, 5), (1, 11, 2, 5), (2, 10, 3, 5), (2, 11, 4, 5),   # train
        (1, 12, 10, 5),     # val: sp 12 không có trong train → bỏ
        (2, 111, 11, 5),    # val: sp 111 không có trong train → bỏ
        (3, 10, 12, 5),     # val: user 3 cold
        (2, 12, 20, 2),     # test: sp 12 → bỏ
        (4, 11, 21, 5),     # test: user 4 cold
    ])
    sp = make_split(df, "all", 2, t_val=10, t_test=20)
    assert len(sp.train) == 4
    assert sp.val["product_id"].tolist() == [10]
    assert sp.stats["val"]["n_interactions_dropped_item_not_in_train"] == 2
    assert sp.stats["val"]["n_cold_users"] == 1 and sp.stats["val"]["n_warm_users"] == 0
    assert sp.stats["test"]["n_users"] == 1 and sp.stats["test"]["n_cold_users"] == 1
    assert sp.stats["train"]["median_interactions_per_user"] == 2.0


def test_compare_expected():
    assert compare_expected({"a": 1, "b": 2}, {"a": 1}) == []
    assert len(compare_expected({"a": 1}, {"a": 2, "b": 3})) == 2
