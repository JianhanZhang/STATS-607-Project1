import pandas as pd
import numpy as np
from src.models.brismf import BRISMF
from src.models.iterative_svd import IterativeSVD


def test_brismf_prediction_output():
    # Load a small portion of the dataset for testing
    full = pd.read_csv('data/data.csv').drop(columns=['Unnamed: 0'])[-1000:]
    train = full[:800]
    test = full[-200:]
    X_train = train[['movie_id', 'user_id']]
    y_train = train['rating']
    X_test = test[['movie_id', 'user_id']]
    y_test = test['rating']

    # Initialize and fit BRISMF
    model = BRISMF(K=2, epochs=5)
    model.fit(X_train, y_train)

    # Get predictions
    predictions = model.predict(X_test)

    # Check if the predictions are of the same length as the input
    assert (len(predictions) == len(y_test))

    # Check if the predictions are stored in a numpy array
    assert isinstance(predictions, np.ndarray)


def test_iterative_svd_prediction_output():
    # Load a small portion of the dataset for testing
    full = pd.read_csv('data/data.csv').drop(columns=['Unnamed: 0'])[-1000:]
    train = full[:800]
    test = full[-200:]
    X_train = train[['movie_id', 'user_id']]
    y_train = train['rating']
    X_test = test[['movie_id', 'user_id']]
    y_test = test['rating']

    # Initialize and fit IterativeSVD
    model = IterativeSVD(k=2, num_epochs=5)
    model.fit(X_train, y_train)

    # Get predictions
    predictions = model.predict(X_test)

    # Check if the predictions are of the same length as the input
    assert (len(predictions) == len(y_test))

    # Check if the predictions are stored in a numpy array
    assert isinstance(predictions, np.ndarray)