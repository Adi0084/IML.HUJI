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
    adaboost.fit(train_X, train_y)

    test_loss, train_loss = [], []
    for num_learners in range(1, n_learners):
        test_loss.append(adaboost.partial_loss(test_X, test_y, num_learners))
        train_loss.append(adaboost.partial_loss(train_X, train_y, num_learners))

    min_loss = np.inf
    size_l = np.inf

    for ind, loss in enumerate(test_loss):
        if loss < min_loss:
            min_loss = loss
            size_l = ind

    plt.plot(range(1, n_learners), train_loss, label='train error')
    plt.plot(range(1, n_learners), test_loss, label='test error')
    plt.title('AdaBoost Error - Train and Test')
    plt.xlabel('Number of fitted models')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    model_info = [(t, accuracy(test_y, adaboost.partial_predict(test_X, t))) for t in T]
    model_names = [f'{t} Learners with Accuracy: {acc}' for t, acc in model_info]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                        horizontal_spacing=.03, vertical_spacing=.1)

    scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                         marker=dict(color=test_y,
                                     colorscale=[custom[0], custom[-1]],
                                     line=dict(color="black", width=1)))
    for i, t in enumerate(T):
        pred = lambda X: adaboost.partial_predict(X, t)
        fig.add_traces([decision_surface(pred, lims[0], lims[1], showscale=False), scatter],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision boundaries}}$", margin=dict(t=100), title_x=0.5)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    acc = accuracy(test_y, adaboost.partial_predict(test_X, size_l))
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[rf"$\textbf{{{f'Ensemble size: {size_l}, Accuracy: {acc}'}}}$"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, size_l), lims[0], lims[1], showscale=False),
                    scatter])

    fig.update_layout(title=rf"$\textbf{{Best performing ensemble}}$", margin=dict(t=100), title_x=0.5)
    fig.show()

    # Question 4: Decision surface with weighted samples
    D = (adaboost.D_ / np.max(adaboost.D_))*5
    scatter.marker['size'] = D
    scatter.marker['color'] = train_y
    scatter.x, scatter.y = train_X[:, 0], train_X[:, 1]
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        rf"$\textbf{{{f'{n_learners} Learners, with point size proportional to its weight'}}}$"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, n_learners), lims[0], lims[1],
                                     showscale=False), scatter])

    fig.update_layout(title=rf"$\textbf{{Last iteration}}$", margin=dict(t=100), title_x=0.5)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
