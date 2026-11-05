import pandas as pd

from analysis.report import k_core, summarize


def test_k_core_iterative():
    # u1,u2 cùng xem i1,i2 → 2-core. u3 chỉ xem i3 → bị loại, kéo theo i3 bị loại.
    df = pd.DataFrame({
        "user": ["u1", "u1", "u2", "u2", "u3", "u1"],
        "product_id": [1, 2, 1, 2, 3, 3],
    })
    core = k_core(df, 2)
    assert set(core["user"]) == {"u1", "u2"}
    assert set(core["product_id"]) == {1, 2}
    assert len(core) == 4


def test_k_core_counts_pairs_once_and_can_be_empty():
    df = pd.DataFrame({"user": ["u1", "u1"], "product_id": [1, 1]})
    assert k_core(df, 2).empty


def test_summarize():
    df = pd.DataFrame({
        "review_id": [1, 2, 3, 3],
        "user": ["a", "a", "b", "b"],
        "product_id": [1, 2, 1, 1],
    })
    s = summarize(df)
    assert s["reviews"] == 3
    assert s["users"] == 2 and s["items"] == 2
    assert s["users_>=2_items"] == 1
    assert s["density"] == 3 / 4
