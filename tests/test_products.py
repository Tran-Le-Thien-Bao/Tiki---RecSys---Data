import pytest

from crawler.products import category_id, list_products, parse_products


@pytest.mark.parametrize("value,expected", [
    ("https://tiki.vn/dien-thoai-may-tinh-bang/c1789", 1789),
    ("https://tiki.vn/nha-sach-tiki/c8322?page=2", 8322),
    ("tiki.vn/lam-dep-suc-khoe/c1520/", 1520),
    ("1789", 1789),
    (1789, 1789),
])
def test_category_id(value, expected):
    assert category_id(value) == expected


@pytest.mark.parametrize("bad", ["https://tiki.vn/abc", "https://tiki.vn/sp-p12345.html", ""])
def test_category_id_invalid(bad):
    with pytest.raises(ValueError):
        category_id(bad)


def test_parse_products():
    rows = parse_products({"data": [
        {"id": 1, "name": "A", "brand_name": "B", "price": 100, "rating_average": 4.5, "review_count": 7,
         "seller_id": 9},
        {"name": "thiếu id"},
    ]}, 1789)
    assert rows == [{"id": 1, "name": "A", "brand": "B", "price": 100, "rating_average": 4.5,
                     "review_count": 7, "category": 1789}]


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get_json(self, path, params):
        self.calls.append(params["page"])
        return {"data": self.pages[params["page"] - 1], "paging": {"last_page": len(self.pages)}}


def test_list_products_filters_and_max_pages():
    pages = [[{"id": 1, "review_count": 10}, {"id": 2, "review_count": 1}], [{"id": 3, "review_count": 5}],
             [{"id": 4, "review_count": 50}]]
    client = FakeClient(pages)
    ids = [p["id"] for p in list_products(client, 1, max_pages=2, min_reviews=5)]
    assert ids == [1, 3]
    assert client.calls == [1, 2]
