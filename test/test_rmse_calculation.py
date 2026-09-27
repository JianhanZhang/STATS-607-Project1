import numpy as np
from src.evaluation import compute_rmse


def test_compute_rmse_equal_values():
    '''Test case with equal predicted and actual values'''
    predicted = np.array([3, 4, 5])
    actual = np.array([3, 4, 5])
    assert compute_rmse(predicted, actual) == 0.0


def test_compute_rmse_different_values():
    '''Test case with different predicted and actual values'''
    predicted = np.array([3, 4, 5])
    actual = np.array([1, 2, 3])
    expected_rmse = ((2**2 + 2**2 + 2**2) / 3) ** 0.5
    assert np.isclose(compute_rmse(predicted, actual), expected_rmse)

def test_compute_rmse_negative_values():
    '''Test case with predicted and actual values being negative'''
    predicted = np.array([-1, -2, -3])
    actual = np.array([-1, -2, -3])
    assert compute_rmse(predicted, actual) == 0.0

def test_compute_rmse_mixed_values():
    '''Test case with predicted and actual values being a mix of positive and negative numbers'''
    predicted = np.array([1, -1, 0])
    actual = np.array([0, -1, 1])
    expected_rmse = ((1**2 + 0**2 + (-1)**2) / 3) ** 0.5
    assert np.isclose(compute_rmse(predicted, actual), expected_rmse)