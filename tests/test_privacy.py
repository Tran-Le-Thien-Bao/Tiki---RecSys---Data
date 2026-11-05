from pathlib import Path

import pytest

from crawler.privacy import load_or_create_key, pseudo, scrub

KEY = b"k" * 32


def test_pseudo_deterministic_and_16_hex():
    h = pseudo(123456, KEY)
    assert h == pseudo("123456", KEY)
    assert len(h) == 16
    int(h, 16)
    assert "123456" not in h


def test_pseudo_depends_on_key():
    assert pseudo(1, KEY) != pseudo(1, b"x" * 32)
    assert pseudo(1, KEY) != pseudo(2, KEY)


@pytest.mark.parametrize("text", [
    "Gọi 0912345678 nhé",
    "sdt 0912 345 678",
    "lh 091.234.5678",
    "+84912345678 shop",
    "84 912 345 678",
])
def test_scrub_phone(text):
    out = scrub(text)
    assert "[sdt]" in out
    assert not any(ch.isdigit() for ch in out)


def test_scrub_email_and_keep_normal_text():
    assert scrub("mail me: a.b+c@gmail.com ok") == "mail me: [email] ok"
    assert scrub("Giao hàng 2 ngày, giá 150000đ") == "Giao hàng 2 ngày, giá 150000đ"
    assert scrub(None) == ""


def test_key_created_and_reused(tmp_path: Path):
    p = tmp_path / "secrets" / "k.key"
    k1 = load_or_create_key(p)
    assert len(k1) == 32
    assert load_or_create_key(p) == k1


def test_key_rejected_inside_forbidden_dir(tmp_path: Path):
    with pytest.raises(ValueError):
        load_or_create_key(tmp_path / "data" / "k.key", forbidden_dirs=[tmp_path / "data"])
