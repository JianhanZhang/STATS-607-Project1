import numpy as np

def compute_rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual) * (predicted - actual)))