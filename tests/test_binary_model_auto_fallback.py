import asyncio

from main import run_pipeline


def test_binary_model_auto_fallback_without_julia():
    result = asyncio.run(
        run_pipeline(
            mock_mode=True,
            contract_path="artifacts/upstream/model_contract.json",
            event="tornado",
            window="f13-24",
            inference_backend="auto",
        )
    )
    assert result["status"] == "success"
