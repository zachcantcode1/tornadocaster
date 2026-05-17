import numpy as np
import lightgbm as lgb
import pytest
from src.models.lgb_bridge import JuliaToLightGBMBridge

def test_lgbm_bridge_inference():
    X_train = np.array([[4.0, 0.0], [8.0, 0.0]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.float32)
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression', 
        'num_leaves': 2, 
        'max_depth': 1,
        'min_data_in_leaf': 1,
        'min_data_in_bin': 1
    }
    booster_orig = lgb.train(params, dtrain, num_boost_round=1)
    
    mock_lgbm_text = booster_orig.model_to_string()
    
    booster = JuliaToLightGBMBridge.load_model_from_json(mock_lgbm_text)
    
    X_mock = np.array([
        [4.0, 0.0],
        [8.0, 0.0]
    ], dtype=np.float32)
    
    predictions = booster.predict(X_mock)
    print("\nInference Output:", predictions)
    
    assert len(predictions) == 2
    assert predictions[0] != predictions[1]
    
    print("Step 3 Model Bridge: Python inference matches expected behavior via string deserialization.")

if __name__ == "__main__":
    test_lgbm_bridge_inference()
