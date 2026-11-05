"""Client HTTP cho API Tiki: giới hạn tốc độ, retry, dừng khi bị chặn."""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://tiki.vn"

# Chỉ các endpoint đã kiểm tra.
ALLOWED_PATHS = frozenset({"/api/v2/products", "/api/v2/reviews"})

MIN_DELAY = 1.5
JITTER = 1.0
MAX_RETRIES = 4


class CrawlBlocked(RuntimeError):
    """Bị chặn (401/403 hoặc nhận HTML) — phải dừng cào ngay."""


class TikiClient:
    """Gọi API Tiki tuần tự, một luồng, có độ trễ tối thiểu giữa các request."""

    def __init__(
        self,
        contact: str,
        delay: float = MIN_DELAY,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not contact or "@" not in contact:
            raise ValueError("CRAWLER_CONTACT phải là email liên hệ hợp lệ")
        # Không cho phép giảm độ trễ dưới mức tối thiểu.
        self.delay = max(delay, MIN_DELAY)
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": f"UIT-RecSys-Research/0.1 (academic research project; contact: {contact})",
            "Accept": "application/json",
        })
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def _throttle(self) -> None:
        """Chờ đủ delay + ngẫu nhiên 0–1 giây kể từ request trước."""
        if self._last is not None:
            wait = self.delay + random.uniform(0, JITTER) - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    @staticmethod
    def _retry_after(resp: requests.Response, attempt: int) -> float:
        """Thời gian chờ theo Retry-After, nếu không có thì 5 * 2**i."""
        value = resp.headers.get("Retry-After")
        if value:
            try:
                return max(float(value), 0.0)
            except ValueError:
                pass
        return 5.0 * 2 ** attempt

    def get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET một endpoint được phép, trả về JSON."""
        if path not in ALLOWED_PATHS:
            raise ValueError(f"Endpoint không được phép: {path}")
        url = BASE_URL + path
        for attempt in range(MAX_RETRIES + 1):
            self._throttle()
            resp = self.session.get(url, params=params, timeout=30)
            status = resp.status_code

            if status in (401, 403):
                raise CrawlBlocked(f"HTTP {status} tại {path} {params} — dừng cào")
            if status == 429 or status >= 500:
                if attempt == MAX_RETRIES:
                    resp.raise_for_status()
                wait = self._retry_after(resp, attempt)
                log.warning("HTTP %s tại %s, chờ %.0fs (lần %d/%d)", status, path, wait, attempt + 1, MAX_RETRIES)
                self._sleep(wait)
                continue
            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "")
            if "json" not in ctype.lower():
                raise CrawlBlocked(f"Nhận '{ctype}' thay vì JSON tại {path} — có thể bị chặn, dừng cào")
            try:
                return resp.json()
            except ValueError as exc:
                raise CrawlBlocked(f"Phản hồi không phải JSON tại {path} — dừng cào") from exc
        raise RuntimeError("unreachable")
