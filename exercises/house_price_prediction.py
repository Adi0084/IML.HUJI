import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

from typing import NoReturn
from typing import Optional
from typing import Tuple

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

pio.templates.default = "simple_white"

UNNECESSARY_COL = ["id", "date", "lat", "long", "sqft_living15", "sqft_lot15"]
POSITIVE_COL = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
NON_NEGATIVE_COL = ["sqft_basement", "yr_renovated"]
COLS_RANGE = {'view': range(5), 'condition': range(1, 6),
              "grade": range(1, 14), 'waterfront': range(0, 2)}


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = X.drop(UNNECESSARY_COL, axis='columns')
    X = X.replace('nan', np.nan)
    if y is not None:
        data = pd.concat([X, y], axis='columns')
        data = data[data['price'] > 0]
        data = preprocess_data_helper(data, y)

    else:
        data = X
        data = preprocess_data_helper(data)

    data = pd.get_dummies(data, columns=['zipcode'])

    if y is not None:
        y = data.pop("price")
        return data, y
    return data


def preprocess_data_helper(X: pd.DataFrame,  y: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Preprocess the data by removing outliers and replacing missing values.

    Parameters
    ----------
    X : pd.DataFrame
        The input data.

    train_data : bool
        Whether or not the data is training data.

    Returns
    -------
    pd.DataFrame
        The preprocessed data.
    """
    if y is not None:
        X = X.dropna().drop_duplicates()

        for col in COLS_RANGE.keys():
            X = X[X[col].isin(COLS_RANGE[col])]

        for col in POSITIVE_COL:
            X = X[X[col] > 0]

        for col in NON_NEGATIVE_COL:
            X = X[X[col] >= 0]

    else:
        X_mean = X.mean()

        for col in COLS_RANGE.keys():
            X.loc[X[col].isna(), col] = X_mean[col]
            X.loc[~X[col].isin(COLS_RANGE[col]), col] = X_mean[col]

        for col in POSITIVE_COL:
            X.loc[X[col].isna(), col] = X_mean[col]
            X.loc[X[col] <= 0, col] = X_mean[col]

        for col in NON_NEGATIVE_COL:
            X.loc[X[col].isna(), col] = X_mean[col]
            X.loc[X[col] < 0, col] = X_mean[col]
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        if 'zipcode_' not in feature:
            cov = X[feature].cov(y) / (np.std(X[feature]) * np.std(y))
            go.Figure(go.Scatter(x=X[feature], y=y, mode="markers",
                                 marker=dict(color="blue")),
                      layout=dict(
                          title=f"Correlation Between {feature} "
                                f"Values and Response <br>Pearson "
                                f"Correlation {cov}",
                          xaxis_title="value of " + feature,
                          yaxis_title="price")).write_image(
                output_path + f"_{feature}.png", width=1000, height=700)


def fit_model(fit_train_X: pd.DataFrame , fit_train_y: pd.Series, fit_test_X: pd.DataFrame, fit_test_y: pd.Series):
    """
    For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
     1) Sample p% of the overall training data
     2) Fit linear model (including intercept) over sampled set
     3) Test fitted model over test set
     4) Store average and variance of loss over test set
    Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    Returns:
    NoReturn
    """

    percentage = list(range(10, 101))
    mse = np.zeros((len(percentage)))
    std = np.zeros((len(percentage)))

    # For every percentage p in 10%, 11%, ..., 100%
    for i, p in enumerate(percentage):
        temp_mse_results = np.zeros(10)
        # repeat the following 10 times
        for j in range(10):
            # Sample p% of the overall training data
            p_train_X = fit_train_X.sample(frac=(p/100))
            p_train_y = fit_train_y.loc[p_train_X.index]

            lr_fitted = LinearRegression(include_intercept=True)
            temp_mse_results[j] = lr_fitted.fit(p_train_X, p_train_y).loss(fit_test_X, fit_test_y)

        mse[i] = temp_mse_results.mean()
        std[i] = temp_mse_results.std()

        fig = go.Figure(
            [go.Scatter(x=percentage, y=mse, mode="markers+lines", name="MSE",
                        marker=dict(color="black"), line=dict(color='black')),
             go.Scatter(x=percentage, y=mse + (2 * std), mode='lines', name='MSE + 2 * std',
                        line=dict(color='lightgrey')),
             go.Scatter(x=percentage, y=mse - (2 * std), mode='lines', name='MSE - 2 * std',
                        line=dict(color='lightgrey'))],
            layout=go.Layout(
                title="MSE vs size",
                xaxis=dict(title="p%"), yaxis=dict(title='MSE test')))
        fig.write_image('MSE vs size.png')


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    df = pd.read_csv("../datasets/house_prices.csv")
    df = df[df['price'] != np.nan]
    y = df.pop('price')
    train_X, train_y, test_X, test_y = split_train_test(df, y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, 'pc')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_model(train_X, train_y, test_X, test_y)



