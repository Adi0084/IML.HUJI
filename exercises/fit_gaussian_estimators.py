from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    X = np.random.normal(loc=mu, scale=sigma, size=1000)
    fit = UnivariateGaussian().fit(X)
    print((np.round(fit.mu_, 3), np.round(fit.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    err = np.zeros((100, 2))
    for i, num_of_samples in enumerate(range(10, 1001, 10)):
        samples = np.random.choice(X, size=num_of_samples)
        fit = UnivariateGaussian().fit(samples)
        loss_mu = np.abs(mu - fit.mu_)
        err[i] = num_of_samples, loss_mu
    plt.plot(err.T[0], err.T[1])
    plt.xlabel('Number of samples')
    plt.ylabel('Absolute Error')
    plt.title('Deviation of Sample Mean Estimation')
    plt.show()
    plt.close()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = fit.pdf(X)
    plt.scatter(X, pdfs)
    plt.xlabel('X')
    plt.ylabel('Empirical probability estimation')
    plt.title('Empirical pdf using fitted model')
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = [0, 0, 4, 0]
    cov = np.array([[1, .2, 0, .5], [.2, 2, 0, 0], [0, 0, 1, 0], [.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)
    fit = MultivariateGaussian().fit(X)
    print(np.round(fit.mu_, 3))
    print(np.round(fit.cov_, 3))
    fit.pdf(X)

    # Question 5 - Likelihood evaluation
    log_likelihood = np.zeros((200, 200))
    fs = np.linspace(-10, 10, log_likelihood.shape[0])
    for i, f1 in enumerate(fs):
        for j, f3 in enumerate(fs):
            log_likelihood[i, j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, X)
    go.Figure(go.Heatmap(x=fs, y=fs, z=log_likelihood),
              layout=dict(title="Log Likelihood as function of f1, f3",
                          xaxis_title="f1",
                          yaxis_title="f3")).show()

    # Question 6 - Maximum likelihood
    res = fs[list(np.unravel_index(log_likelihood.argmax(), log_likelihood.shape))]
    print(np.round(res, 3))


if __name__ == '__main__':
    # np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
