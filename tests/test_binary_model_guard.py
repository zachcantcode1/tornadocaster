import pytest

from src.models.lgb_bridge import JuliaToLightGBMBridge


def test_binary_model_rejected():
    model_path = "artifacts/upstream/models/676_trees_loss_0.0012007512.model"
    with pytest.raises(ValueError, match="MemoryConstrainedTreeBoosting"):
        JuliaToLightGBMBridge.load_model_from_file(model_path)
