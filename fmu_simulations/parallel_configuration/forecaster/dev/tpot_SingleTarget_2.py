import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -56539.4095929883
exported_pipeline = make_pipeline(
    StandardScaler(),
    SelectFwe(score_func=f_regression, alpha=0.033),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=1.0, min_samples_leaf=15, min_samples_split=15, n_estimators=100)),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="huber", max_depth=5, max_features=0.9000000000000001, min_samples_leaf=12, min_samples_split=18, n_estimators=100, subsample=0.45)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
