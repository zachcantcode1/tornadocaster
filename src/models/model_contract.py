import json
from pathlib import Path


def load_contract(contract_path: str) -> dict:
    return json.loads(Path(contract_path).read_text())


def select_model_path(contract: dict, event: str, window: str) -> str:
    window_key = {
        "f2-13": "f2_13_model_path",
        "f13-24": "f13_24_model_path",
        "f24-35": "f24_35_model_path",
    }.get(window)
    if window_key is None:
        raise ValueError("window must be one of f2-13, f13-24, f24-35")

    row = next((r for r in contract.get("models", []) if r.get("event") == event), None)
    if row is None:
        raise ValueError(f"event `{event}` not found in model contract")
    return row[window_key]
