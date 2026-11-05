"""Ẩn danh người dùng (HMAC) và che thông tin cá nhân trong văn bản."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# SĐT Việt Nam: 0xxx hoặc +84/84xxx, 9–10 chữ số sau đầu số, cho phép dấu cách/chấm/gạch.
PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")


def load_or_create_key(path: str | os.PathLike[str], forbidden_dirs: list[Path] | None = None) -> bytes:
    """Đọc khóa HMAC; nếu chưa có thì tạo 32 byte ngẫu nhiên. Từ chối nếu khóa nằm trong repo/data."""
    key_path = Path(path).expanduser().resolve()
    for d in forbidden_dirs or []:
        d = Path(d).expanduser().resolve()
        if key_path == d or d in key_path.parents:
            raise ValueError(f"Khóa HMAC không được nằm trong {d}")
    if key_path.exists():
        key = key_path.read_bytes()
        if len(key) < 16:
            raise ValueError(f"Khóa HMAC tại {key_path} quá ngắn")
        return key
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    key_path.write_bytes(key)
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return key


def pseudo(customer_id: int | str, key: bytes) -> str:
    """Mã giả danh: HMAC-SHA256(customer_id), lấy 16 ký tự hex đầu."""
    return hmac.new(key, str(customer_id).encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def scrub(text: str | None) -> str:
    """Che email thành [email] và số điện thoại thành [sdt]."""
    if not text:
        return ""
    text = EMAIL_RE.sub("[email]", text)
    return PHONE_RE.sub("[sdt]", text)
