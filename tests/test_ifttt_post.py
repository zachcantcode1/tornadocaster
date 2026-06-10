import sys
from argparse import Namespace
from datetime import date

import pytest

from post_ifttt import (
    _nadocast_valid_period,
    _spc_valid_period,
    build_forecast_command,
    build_webhook_url,
    default_tweet_text,
)


def test_build_webhook_url_prefers_full_url():
    assert build_webhook_url("https://example.test/hook", None, None) == "https://example.test/hook"


def test_build_webhook_url_from_event_and_key():
    assert build_webhook_url(None, "post_tornado", "abc123") == (
        "https://maker.ifttt.com/trigger/post_tornado/with/key/abc123"
    )


def test_build_webhook_url_requires_configuration():
    with pytest.raises(ValueError):
        build_webhook_url(None, "post_tornado", None)


def test_build_forecast_command_for_spc_tornado():
    args = Namespace(output="out.png", map_style="dark", product="spc-tornado")

    assert build_forecast_command(args) == [
        sys.executable,
        "forecast_now.py",
        "--output",
        "out.png",
        "--map-style",
        "dark",
        "--spc-day1",
        "tornado",
    ]


def test_default_tweet_text_for_nadocast(monkeypatch):
    async def fake_default_post_text(product):
        assert product == "nadocast"
        return (
            "NADOCast Tornado Guidance\n"
            "Generated: 10:00 AM CDT Jun 10\n"
            "Run: 7:00 AM CDT Jun 10\n"
            "Valid: 9:00 AM CDT Jun 10 - 6:00 AM CDT Jun 11"
        )

    monkeypatch.setattr("post_ifttt.default_post_text", fake_default_post_text)

    text = default_tweet_text("nadocast")

    assert "NADOCast Tornado Guidance" in text
    assert "Generated:" in text
    assert "Run:" in text
    assert "Valid:" in text


def test_nadocast_valid_period_uses_chicago_time():
    assert _nadocast_valid_period(date(2026, 6, 10), 12, "f02-23") == (
        "9:00 AM CDT Jun 10 - 6:00 AM CDT Jun 11"
    )


def test_spc_valid_period_uses_chicago_time():
    assert _spc_valid_period("2026-06-10T13:00:00+00:00", "2026-06-11T12:00:00+00:00") == (
        "8:00 AM CDT Jun 10 - 7:00 AM CDT Jun 11"
    )
