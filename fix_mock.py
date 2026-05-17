import numpy as np
import lightgbm as lgb
X_dummy = np.array([[0.0], [6000.0]], dtype=np.float32)
y_dummy = np.array([0.01, 0.30], dtype=np.float32)
dtrain = lgb.Dataset(X_dummy, label=y_dummy)
params = {
    'objective': 'regression',
    'max_depth': 2,
    'num_leaves': 3,
    'min_data_in_bin': 1,
    'min_data_in_leaf': 1
}
mock_booster = lgb.train(params, dtrain, num_boost_round=5)
print(mock_booster.predict([[-100], [5000]]))
