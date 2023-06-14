from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    new_X = np.concatenate((X, X))
    new_y = np.concatenate((y, y))
    test_set_size = int(np.ceil(X.shape[0] / cv))
    train_set_size = int(X.shape[0] - test_set_size)
    train_score = 0
    test_score = 0
    for i in range(cv):
        train_set = new_X[i:i + train_set_size, :]
        test_set = new_X[i + train_set_size:i + train_set_size + test_set_size, :]
        estimator.fit(train_set, new_y[i:i + train_set_size])
        pred = estimator.predict(train_set)
        train_score += scoring(new_y[i:i + train_set_size], pred)
        test_score += scoring(new_y[i + train_set_size:i + train_set_size + test_set_size], estimator.predict(test_set))
    test_score /= cv
    train_score /= cv
    return train_score, test_score
