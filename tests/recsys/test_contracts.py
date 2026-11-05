"""Test các quy ước dùng chung: config, schema, Recommender cơ sở, ghi kết quả."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from recsys import config
from recsys.evaluation.results import hash_user, save_run
from recsys.models.base import Recommender
from recsys.schema import RunConfig, SchemaError, validate_interactions, validate_per_user


# ---------------------------------------------------------------- config
def test_config_consistent():
    assert 0 < config.VAL_QUANTILE < config.TEST_QUANTILE < 1
    assert 0 in config.KCORE_LEVELS
    assert set(config.TRUST_LAMBDA) == set(config.FLAG_NAMES)
    assert 0 < config.TRUST_EPS < 1
    assert config.FAKE_USER_ID_START < 0
    assert config.split_dir("all", 2).name == "all_2core"
    with pytest.raises(ValueError):
        config.split_dir("all", 4)


# ---------------------------------------------------------------- schema
def test_validate_interactions_ok(toy_interactions):
    validate_interactions(toy_interactions)


def test_validate_interactions_duplicate_pair(toy_interactions):
    bad = pd.concat([toy_interactions, toy_interactions.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="trùng"):
        validate_interactions(bad)


def test_validate_interactions_fake_flag_mismatch(toy_interactions):
    bad = toy_interactions.copy()
    bad.loc[0, "is_fake"] = True
    with pytest.raises(SchemaError, match="is_fake"):
        validate_interactions(bad)


def test_validate_per_user_forbids_user_id():
    df = pd.DataFrame({"user_id": [1], "user_hash": ["a"], "is_warm": [True], "n_truth": [1]})
    with pytest.raises(SchemaError, match="quyền riêng tư"):
        validate_per_user(df)


def test_validate_per_user_metric_names():
    ok = pd.DataFrame({"user_hash": ["a"], "is_warm": [True], "n_truth": [1], "ndcg@10": [0.5]})
    validate_per_user(ok)
    with pytest.raises(SchemaError):
        validate_per_user(ok.rename(columns={"ndcg@10": "ndcg10"}))


def test_run_id_stable():
    a = RunConfig(stage="stage2", model="als", k=2, seed=1, attack="random", attack_level=0.03)
    b = RunConfig(stage="stage2", model="als", k=2, seed=1, attack="random", attack_level=0.03)
    assert a.run_id == b.run_id == "als__all_2core__s1__val__random0.03"


# ---------------------------------------------------------------- Recommender cơ sở
class _ItemSum(Recommender):
    """Mô hình giả: điểm của item = tổng cột R (giống Popular có trọng số)."""

    name = "itemsum"

    def _fit(self, R, train, item_features, seed):
        self.pop = np.asarray(R.sum(axis=0)).ravel()

    def _score(self, user_idx, item_idx):
        return np.tile(self.pop[item_idx], (len(user_idx), 1))


def test_recommend_excludes_seen_and_ties(toy_interactions):
    m = _ItemSum().fit(toy_interactions)
    # tổng: 10→4, 11→2, 12→2, 13→1, 14→2 ; user 1 đã xem 10,11,12
    assert m.recommend([1], k=2) == {1: [14, 13]}
    # user 4 đã xem 10; hòa 11/12/14 → product_id nhỏ trước
    assert m.recommend([4], k=3) == {4: [11, 12, 14]}
    assert m.recommend([4], k=2, exclude_seen=False) == {4: [10, 11]}


def test_cold_user_empty_when_unsupported(toy_interactions):
    m = _ItemSum().fit(toy_interactions)
    assert m.recommend([999], k=3) == {999: []}
    assert np.isnan(m.score([999])).all()


def test_sample_weight_changes_R(toy_interactions):
    w = np.where(toy_interactions["is_fake"], 0.05, 1.0)
    m = _ItemSum().fit(toy_interactions, sample_weight=w)
    j = m.item_index[14]
    assert m.pop[j] == pytest.approx(1.05)


def test_score_unknown_item_is_neg_inf(toy_interactions):
    m = _ItemSum().fit(toy_interactions)
    s = m.score([1], item_ids=[10, 777])
    assert np.isfinite(s[0, 0]) and s[0, 1] == -np.inf


# ---------------------------------------------------------------- ghi kết quả
def test_save_run(tmp_path):
    per_user = pd.DataFrame({
        "user_hash": [hash_user(1), hash_user(2)],
        "is_warm": [True, False],
        "n_truth": np.array([2, 1], dtype="int32"),
        "ndcg@10": [0.5, 0.0],
    })
    out = save_run("pilot", "demo", per_user, {"mean_ndcg@10": np.float64(0.25)}, root=tmp_path)
    assert (out / "per_user.parquet").exists()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == {"mean_ndcg@10": 0.25}
    assert hash_user(1) != hash_user(2) and len(hash_user(1)) == 16
