"""Test bộ gắn cờ trên dữ liệu giả."""
from __future__ import annotations

import pandas as pd

from recsys.mitigation.flags import flag_burst, flag_dup_content, flag_promo, flag_reviews, flags_by_pair


def _rev(rows):
    df = pd.DataFrame(rows, columns=["review_id", "user_id", "product_id", "title", "content", "created_at"])
    df["had_phone"] = False
    return df


def test_dup_content_threshold():
    long_ = "Sản phẩm rất tốt giao hàng nhanh"          # > 20 ký tự
    rows = [(i, i, 1, "", long_ if i < 5 else "khác " * 6, 0) for i in range(6)]
    rows.append((99, 99, 1, "", "  SẢN PHẨM rất tốt   giao hàng nhanh ", 0))   # chuẩn hóa trùng
    rows.append((100, 100, 1, "", "ngắn", 0))
    f = flag_dup_content(_rev(rows))
    assert f.iloc[:5].all() and f.iloc[6] and not f.iloc[5] and not f.iloc[7]


def test_promo_keyword_and_phone():
    df = _rev([(1, 1, 1, "", "Ib mình qua ZALO nhé", 0), (2, 2, 1, "", "hàng đẹp", 0), (3, 3, 1, "", "hàng đẹp", 0)])
    df.loc[2, "had_phone"] = True
    assert flag_promo(df).tolist() == [True, False, True]


def test_burst_same_product_and_window():
    rows = [(1, 1, 10, "", "", 0), (2, 1, 10, "", "", 100)]                       # cùng sp 2 lần
    rows += [(10 + i, 2, 20 + i, "", "", i * 60) for i in range(5)]               # 5 sp trong 4 phút
    rows += [(20 + i, 3, 30 + i, "", "", i * 3600 * 5) for i in range(5)]         # 5 sp rải rác
    f = flag_burst(_rev(rows), same_product_min=2, n_products=5, window_min=60)
    assert f.iloc[:2].all() and f.iloc[2:7].all() and not f.iloc[7:].any()


def test_flags_by_pair_any():
    rf = flag_reviews(_rev([(1, 1, 10, "", "zalo", 0), (2, 1, 10, "", "ok", 5), (3, 2, 10, "", "ok", 0)]))
    pf = flags_by_pair(rf).set_index(["user_id", "product_id"])
    assert pf.loc[(1, 10), "promo"] and pf.loc[(1, 10), "burst"] and not pf.loc[(2, 10)].any()
