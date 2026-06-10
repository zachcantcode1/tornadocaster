from argparse import Namespace

import pytest

from autopost import _safe_filename, load_state, save_state, webhook_url_for


def test_state_round_trip(tmp_path):
    state_file = tmp_path / "state.json"
    save_state(state_file, {"nadocast": "run-a"})

    assert load_state(state_file) == {"nadocast": "run-a"}


def test_missing_state_file_returns_empty(tmp_path):
    assert load_state(tmp_path / "missing.json") == {}


def test_safe_filename_strips_unsafe_characters():
    assert _safe_filename("2026-06-10T13:00:00+00:00 / name") == "2026-06-10T13_00_00_00_00_name"


def test_webhook_url_prefers_explicit_value(monkeypatch):
    args = Namespace(nadocast_webhook_url="https://example.test/nadocast", spc_tornado_webhook_url=None)

    assert webhook_url_for("nadocast", args) == "https://example.test/nadocast"


def test_webhook_url_reads_product_env(monkeypatch):
    monkeypatch.setenv("IFTTT_SPC_TORNADO_WEBHOOK_URL", "https://example.test/spc")
    args = Namespace(nadocast_webhook_url=None, spc_tornado_webhook_url=None)

    assert webhook_url_for("spc-tornado", args) == "https://example.test/spc"


def test_webhook_url_requires_product_env(monkeypatch):
    monkeypatch.delenv("IFTTT_NADOCAST_WEBHOOK_URL", raising=False)
    args = Namespace(nadocast_webhook_url=None, spc_tornado_webhook_url=None)

    with pytest.raises(ValueError):
        webhook_url_for("nadocast", args)
