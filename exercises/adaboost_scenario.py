import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, iterations=n_learners)
    adaboost._fit(train_X, train_y)

    preds = []
    test_loss, train_loss = [], []
    for num_learners in range(1, n_learners):
        preds.append(adaboost.partial_predict(test_X, num_learners))
        test_loss.append(adaboost.partial_loss(test_X, test_y, num_learners))
        train_loss.append(adaboost.partial_loss(train_X, train_y, num_learners))

    # test_scatter = go.Scatter(x=np.arange(len(test_loss)), y=test_loss, mode='markers+lines', name='Test Set')
    # train_scatter = go.Scatter(x=np.arange(len(test_loss)), y=train_loss, mode='markers+lines', name='Train Set')
    # fig = go.Figure([test_scatter, train_scatter],
    #                 layout=go.Layout(title=r"$\text{Loss as function of weak learners used in AdaBoost model} $",
    #                                  xaxis_title="$\\text{Number of weak learners}$",
    #                                  yaxis_title="$\\text{loss over dataset}$",
    #                                  height=400
    #                                  ))
    # fig.show()
    plt.plot(range(1, n_learners), train_loss, label='train')
    plt.plot(range(1, n_learners), test_loss, label='test')
    plt.title('Error - Train and test')
    plt.legend()
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # raise NotImplementedError()

    # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
