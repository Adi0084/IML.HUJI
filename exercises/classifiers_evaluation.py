import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from math import atan2, pi

PATH_TO_DATA = "..\\datasets"

MODE = 'markers'


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for name, file in [("Linearly Separable", "linearly_separable.npy"),
                       ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"{PATH_TO_DATA}\{file}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(perceptron: Perceptron, x: np.ndarray, y_: int):
            losses.append(perceptron._loss(X, y))

        perceptron = Perceptron(callback=callback)
        perceptron._fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.plot(losses)
        plt.xlabel("Number of Iterations")
        plt.ylabel("loss")
        plt.title(f"The loss of data that is: {name}")
        plt.show()
        plt.savefig(f"Perceptron fit {name}.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", line=dict(color="black"))


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for name, data in [("Gaussian1", "gaussian1.npy"),
                       ("Gaussian2", "gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset(f"{PATH_TO_DATA}\{data}")

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda.fit(X, y)
        gnb.fit(X, y)
        lda_pred, gnb_pred = lda._predict(X), gnb._predict(X)
        from IMLearn.metrics import accuracy
        acc_gnb = round(100 * accuracy(y, gnb_pred), 3)
        acc_lda = round(100 * accuracy(y, lda_pred), 3)

        plot_comparison(name, X, y, gnb_pred, lda_pred, acc_gnb, acc_lda, gnb, lda)


def plot_comparison(name, X, y, gnb_pred, lda_pred, acc_gnb, acc_lda, gnb, lda):
    """
    Plot a comparison of Gaussian Naive Bayes and LDA predictions.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"GNB with acc={acc_gnb}%",
        f"LDA with acc={acc_lda}%"))

    preds = [gnb_pred, lda_pred]
    models = [gnb, lda]
    for i in range(2):
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode=MODE,
                                 marker=dict(color=preds[i], symbol=class_symbols[y])), row=1, col=i + 1)

        fig.add_trace(go.Scatter(x=models[i].mu_[:, 0], y=models[i].mu_[:, 1], mode=MODE,
                                 marker=dict(symbol="x", color="black", size=15)), row=1, col=i + 1)

    for i in range(3):
        fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=1)
        fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(title=f"Comparing Gaussian Classifiers on - {name} dataset",
                      title_x=0.5, width=800, height=400, showlegend=False)

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
