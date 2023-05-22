from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / y.shape[0]
        self.mu_ = np.zeros((counts.shape[0], X.shape[1]))
        self.vars_ = np.zeros((counts.shape[0], X.shape[1]))

        for i in self.classes_:
            samples_i = X[y == i]
            self.mu_[i, :] = np.mean(samples_i, axis=0)
            self.vars_[i, :] = np.var(samples_i, axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred = np.zeros((self.classes_.shape[0], X.shape[0]))
        for i in range(self.classes_.shape[0]):
            x_m = X - self.mu_[i]
            sum_val = np.sum((x_m ** 2 / self.vars_[i] + np.log(self.vars_[i])), axis=1)
            pred[i] = np.log(self.pi_[i]) - 0.5 * sum_val

        return np.argmax(pred, axis=0)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros((self.classes_.shape[0], X.shape[0]))

        for i in range(self.classes_.shape[0]):
            x_m = X - self.mu_[i]
            squared_diff_var = (x_m ** 2) / (2 * self.vars_[i])
            denominator = np.sqrt(2 * np.pi * self.vars_[i])
            likelihoods[i] = self.pi_[i] * np.exp(-squared_diff_var) / denominator

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
