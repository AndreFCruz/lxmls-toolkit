import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian", smooth=False):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = smooth
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes, counts = np.unique(y, return_counts=True)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # ----------
        # Solution to Exercise 1

        # class priors' estimates are the relative frequencies
        for i, c in enumerate(classes):
            prior[i] = counts[c] / n_docs

        # maximum likelihood estimate
        for k in range(n_classes):      # y_k
            denominator = x[y.flatten() == k, :].sum()
            numerator = x[y.flatten() == k, :].sum(0)

            if self.smooth:
                likelihood[:, k] = (self.smooth_param + numerator) / (self.smooth_param * n_words + denominator)
            else:
                likelihood[:, k] = numerator / denominator

        # End solution to Exercise 1
        # ----------

        params = np.zeros((n_words+1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
