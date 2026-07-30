"""Unit tests for the data collector's response validation.

No network access: ``requests.get`` is stubbed.
"""

from __future__ import annotations

import pytest

from air_quality import collect
from air_quality.collect import MINIMUM_DAYS, fetch_json, validate_hourly


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise collect.requests.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


def hourly_payload(hours: int = MINIMUM_DAYS * 24) -> dict:
    times = [f"2026-01-01T{h:02d}:00" for h in range(hours)]
    return {"hourly": {"time": times, "pm2_5": [10.0] * hours}}


def test_fetch_json_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = hourly_payload()
    monkeypatch.setattr(collect.requests, "get", lambda *a, **k: FakeResponse(payload))
    assert fetch_json("http://example.test", {}) == payload


def test_fetch_json_rejects_payload_without_hourly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collect.requests, "get", lambda *a, **k: FakeResponse({"daily": {}})
    )
    with pytest.raises(ValueError, match="hourly"):
        fetch_json("http://example.test", {})


def test_fetch_json_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        collect.requests, "get", lambda *a, **k: FakeResponse({}, status=500)
    )
    with pytest.raises(collect.requests.HTTPError):
        fetch_json("http://example.test", {})


def test_validate_accepts_a_sufficient_window() -> None:
    assert validate_hourly(hourly_payload(), "air quality") == MINIMUM_DAYS * 24


def test_validate_rejects_too_few_days() -> None:
    with pytest.raises(ValueError, match="at least"):
        validate_hourly(hourly_payload(hours=24 * 10), "air quality")


def test_validate_rejects_duplicate_timestamps() -> None:
    payload = hourly_payload()
    payload["hourly"]["time"][5] = payload["hourly"]["time"][4]
    with pytest.raises(ValueError, match="duplicate"):
        validate_hourly(payload, "air quality")


def test_validate_rejects_mismatched_series_lengths() -> None:
    payload = hourly_payload()
    payload["hourly"]["pm2_5"] = payload["hourly"]["pm2_5"][:-3]
    with pytest.raises(ValueError, match="mismatched"):
        validate_hourly(payload, "air quality")


def test_validate_rejects_missing_time_series() -> None:
    with pytest.raises(ValueError, match="time"):
        validate_hourly({"hourly": {"pm2_5": [1.0]}}, "air quality")
