from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error as mse
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    # raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    # raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    train_ridge_scores = np.zeros(n_evaluations)
    val_ridge_scores = np.zeros(n_evaluations)
    train_lasso_scores = np.zeros(n_evaluations)
    val_lasso_scores = np.zeros(n_evaluations)

    rng = np.linspace(0.001, 2, n_evaluations)

    for i in range(n_evaluations):
        ridge = RidgeRegression(lam=rng[i])
        lasso = Lasso(alpha=rng[i], max_iter=5000)
        train_ridge_scores[i], val_ridge_scores[i] = cross_validate(ridge, train_X, train_y, mse)
        train_lasso_scores[i], val_lasso_scores[i] = cross_validate(lasso, train_X, train_y, mse)

    model_names = ["Ridge", "Lasso"]
    train_scores = [train_ridge_scores, train_lasso_scores]
    val_scores = [val_ridge_scores, val_lasso_scores]

    for i in range(len(model_names)):
        fig = go.Figure([
            go.Scatter(x=rng, y=train_scores[i], mode="lines", name=f"{model_names[i]} Train Scores",
                       marker=dict(color="black"), line=dict(color="black", width=2)),
            go.Scatter(x=rng, y=val_scores[i], mode="lines", name=f"{model_names[i]} Validation Scores",
                       marker=dict(color="red", line=dict(color="black", width=2)))])
        fig.update_layout(
            title=rf"$\textbf{{{model_names[i]} train and validation errors for {n_samples} samples}}$",
            xaxis=dict(title=r'$\lambda$', title_font=dict(size=16)),
            yaxis=dict(title="Error", title_font=dict(size=16)),
            title_x=0.5, title_font=dict(size=24), margin=dict(t=100))
        fig.show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    model_names.append('Least Squares')
    ridge_lamda_ind = np.argmin(val_ridge_scores)
    lasso_lamda_ind = np.argmin(val_lasso_scores)

    regs = [RidgeRegression(lam=rng[ridge_lamda_ind]), Lasso(alpha=rng[lasso_lamda_ind]), LinearRegression()]

    for i, (model_name, lambda_index) in enumerate(zip(model_names, [ridge_lamda_ind, lasso_lamda_ind, None])):
        regression = regs[i]
        regression.fit(train_X, train_y)
        pred = regression.predict(test_X)
        mse_val = round(mse(y_true=test_y, y_pred=pred), 3)

        print(f"\t\t------ {model_name.upper()} ------")
        if lambda_index is not None:
            print(f"\tOptimal regularization parameter {rng[lambda_index]}")
        print(f"\tMSE: {mse_val}")
        print()


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()

