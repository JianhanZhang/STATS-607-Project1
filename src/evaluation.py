'''Implement functions for evaluating model performance.'''

import numpy as np

def compute_rmse(predicted, actual):
    """Compute the root mean squared error (RMSE) between predictions and observations.

    Parameters
    ----------
    predicted : array-like
        Predicted values.
    actual : array-like
        Observed values.

    Returns
    -------
    float
        RMSE between ``predicted`` and ``actual``.
    """
    return np.sqrt(np.mean((predicted - actual) * (predicted - actual)))