from src.models.model_contract import load_contract, select_model_path


def test_contract_load_and_select():
    contract = load_contract("artifacts/upstream/model_contract.json")
    path = select_model_path(contract, event="tornado", window="f13-24")
    assert "tornado" in path
    assert path.endswith(".model")
