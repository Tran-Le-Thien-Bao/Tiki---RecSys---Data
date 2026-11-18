"""Test metric, bootstrap và 3 mô hình nền trên dữ liệu giả."""
from __future__ import annotations

import math

import numpy as np
import pytest

from recsys.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from recsys.evaluation.metrics import (coverage, evaluate, hit_at_k, jaccard_at_k, kendall_tau, ndcg_at_k,
                                       prediction_shift, recall_at_k)
from recsys.models.als import ALS
from recsys.models.itemknn import ItemKNN
from recsys.models.popular import Popular
from recsys.schema import validate_per_user


# ---------------------------------------------------------------- metric (tính tay)
def test_recall_ndcg_hand():
    rec, truth = [5, 1, 9], {1, 9, 7}
    assert recall_at_k(rec, truth, 3) == pytest.approx(2 / 3)
    # hit ở vị trí 2 và 3: DCG = 1/log2(3) + 1/log2(4); IDCG (3 hit, K=3) = 1 + 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3) + 0.5
    assert ndcg_at_k(rec, truth, 3) == pytest.approx(dcg / (1 + 1 / math.log2(3) + 0.5))
    assert ndcg_at_k([1, 9], {1, 9}, 10) == pytest.approx(1.0)
    assert recall_at_k([], {1}, 10) == 0.0 and math.isnan(recall_at_k([1], set(), 10))


def test_hit_jaccard_coverage_shift_tau():
    assert hit_at_k([1, 2, 3], {3}, 3) == 1.0 and hit_at_k([1, 2, 3], {3}, 2) == 0.0
    assert jaccard_at_k([1, 2, 3], [2, 3, 4], 3) == pytest.approx(2 / 4)
    assert jaccard_at_k([], [], 10) == 1.0
    assert coverage({1: [1, 2], 2: [2, 3]}, 10, 2) == pytest.approx(0.3)
    assert prediction_shift(np.array([[1.0, 2.0]]), np.array([[2.0, 4.0]])) == pytest.approx(1.5)
    assert kendall_tau({"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}) == pytest.approx(1.0)
    assert kendall_tau({"a": 1, "b": 2, "c": 3}, {"a": 30, "b": 20, "c": 10}) == pytest.approx(-1.0)


def test_evaluate_schema():
    df = evaluate({1: [10, 11], 2: []}, {1: {10}, 2: {12}, 3: {10}}, warm_users={1, 2})
    validate_per_user(df)
    assert len(df) == 3 and df["is_warm"].sum() == 2
    assert df["ndcg@10"].tolist() == [1.0, 0.0, 0.0]   # user 3 không có trong recs → 0


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_constant_and_seed():
    ci = bootstrap_ci(np.full(50, 0.3), seed=1)
    assert ci.mean == pytest.approx(0.3) and ci.lo == pytest.approx(0.3) and ci.hi == pytest.approx(0.3)
    x = np.random.default_rng(0).random(200)
    assert bootstrap_ci(x, seed=7) == bootstrap_ci(x, seed=7)
    assert bootstrap_ci(x, seed=7).lo < x.mean() < bootstrap_ci(x, seed=7).hi


def test_paired_bootstrap():
    a = np.random.default_rng(1).random(300)
    same = paired_bootstrap(a, a.copy(), seed=3)
    assert same.mean == 0 and not same.distinguishable
    up = paired_bootstrap(a, a + 0.1, seed=3)
    assert up.mean == pytest.approx(0.1) and up.distinguishable
    with pytest.raises(ValueError):
        paired_bootstrap(a, a[:10])


# ---------------------------------------------------------------- mô hình
@pytest.mark.parametrize("model_cls", [Popular, ItemKNN, ALS])
def test_models_run_and_reproducible(model_cls, toy_interactions):
    kw = {"factors": 4, "iterations": 3} if model_cls is ALS else ({"k": 2} if model_cls is ItemKNN else {})
    m1 = model_cls(**kw).fit(toy_interactions, seed=0)
    m2 = model_cls(**kw).fit(toy_interactions, seed=0)
    r1, r2 = m1.recommend([1, 2, 3], k=2), m2.recommend([1, 2, 3], k=2)
    assert r1 == r2
    seen1 = set(toy_interactions.loc[toy_interactions.user_id == 1, "product_id"])
    assert not set(r1[1]) & seen1


def test_popular_cold_and_weight(toy_interactions):
    m = Popular().fit(toy_interactions)
    assert m.recommend([999], k=1) == {999: [10]}
    w = np.where(toy_interactions["product_id"] == 10, 0.05, 1.0)
    assert Popular().fit(toy_interactions, sample_weight=w).recommend([999], k=1)[999] != [10]


def test_itemknn_cosine_hand(toy_interactions):
    m = ItemKNN(k=10).fit(toy_interactions)
    i11, i12 = m.item_index[11], m.item_index[12]
    # cột 11 và 12 đều được user 1 và 3 tương tác, không ai khác → cosine = 1
    assert m.sim[i11, i12] == pytest.approx(1.0)
