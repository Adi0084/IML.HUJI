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
    indices = np.arange(X.shape[0])
    np.array_split(indices, cv)

    train_scores, validation_scores = [], []

    fold_size = X.shape[0] // cv
    for fold in range(cv):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size

        fold_indices = indices[start_idx:end_idx]
        train_indices = np.delete(indices, fold_indices)

        fit = estimator.fit(X[train_indices], y[train_indices])

        train_score = scoring(y[train_indices], fit.predict(X[train_indices]))
        validation_score = scoring(y[fold_indices], fit.predict(X[fold_indices]))

        train_scores.append(train_score)
        validation_scores.append(validation_score)

    train_score_avg = float(np.mean(train_scores))
    validation_score_avg = float(np.mean(validation_scores))

    return train_score_avg, validation_score_avg



