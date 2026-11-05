"""Group-separated logistic calibration with explicit fit provenance."""
import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_calibrator(x, y, fit_groups, evaluation_groups):
    if set(fit_groups) & set(evaluation_groups):
        raise ValueError('Calibration groups overlap evaluation groups')
    model = LogisticRegression(C=1, max_iter=1000, random_state=58021)
    model.fit(np.asarray(x), y)
    return model
