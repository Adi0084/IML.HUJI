import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from plotly.subplots import make_subplots
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc
from utils import custom
from IMLearn.metrics import misclassification_error, mean_square_error
from IMLearn.model_selection import cross_validate
import plotly.graph_objects as go

MODULES = ["L1", "L2"]



def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}",
                                      title_x=0.5))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])
        return

    return callback, values, weights


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):

    for i, m in enumerate([L1, L2]):
        fig = go.Figure()

        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), tol=1e-5,
                                 max_iter=1000, out_type="last",
                                 callback=callback)

            gd.fit(m(weights=np.copy(init)), init, init)
            # Question 1 - Decent Path
            plot_descent_path(module=m, descent_path=np.array(weights),
                              title=f"For Norm: {MODULES[i]} and eta: {eta}").show()

            fig.add_trace(
                go.Scatter(x=np.arange(len(values)), y=values,
                           mode='lines+markers', name=f"{MODULES[i]}, eta={eta}"))

            # Question 3 - Convergence Rates
            print(f"\tLowest Losses for {MODULES[i]}:")
            print(f"\tEta={eta}, min={np.min(values)}")

        fig.update_layout(title=f"{MODULES[i]} norm as a function of the GD iteration",
                          xaxis_title="Iteration Number",
                          yaxis_title="Convergence Rate",
                          title_x=0.5).show()



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)



def fit_logistic_regression():
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    l1_val_errors, l2_val_errors = [], []
    lam_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    gd = GradientDescent()
    lr = LogisticRegression(solver=gd)
    lr.fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_train, lr.predict_proba(X_train))

    c = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))

    alpha_best = thresholds[np.argmax(tpr - fpr)]
    lr.alpha_ = alpha_best
    # Question 9
    print(f"best alpha is:\n {alpha_best},\n and the test error is:\n {lr.loss(X_test, y_test)}")

    gd1 = GradientDescent(learning_rate=FixedLR(0.002))
    gd2 = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20)

    for lam_val in lam_values:
        print(f"lam is: {lam_val}")
        lrl1 = LogisticRegression(solver=gd1, penalty="l1", lam=lam_val, alpha=0.5)

        lrl2 = LogisticRegression(solver=gd2, penalty="l2", lam=lam_val, alpha=0.5)

        l1_train_score, l1_val_score = cross_validate(lrl1, X_train, y_train, misclassification_error)

        l2_train_score, l2_val_score = cross_validate(lrl2, X_train, y_train, misclassification_error)
        l2_val_errors.append(l2_val_score)
        l1_val_errors.append(l1_val_score)

    l1_lam_best, l2_lam_best = lam_values[np.argmin(l1_val_errors)], lam_values[np.argmin(l2_val_errors)]

    lrl1 = LogisticRegression(solver=gd, penalty="l1", lam=l1_lam_best, alpha=0.5)
    lrl2 = LogisticRegression(solver=gd, penalty="l2", lam=l2_lam_best, alpha=0.5)

    lrl1.fit(X_train, y_train)
    lrl2.fit(X_train, y_train)
    # Question 10 -
    print(f"best lambda for {MODULES[0]} is:\n  {l1_lam_best},\n and the test error is:\n {lrl1.loss(X_test, y_test)}")
    # Question 11 -
    print(f"best lambda for {MODULES[1]} is:\n {l2_lam_best},\n and the test error is:\n {lrl2.loss(X_test, y_test)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
