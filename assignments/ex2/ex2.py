#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:08:22 2017

@author: yazar
"""
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy import optimize


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _RSS(theta, X, y):
    # number of training examples
    m = len(y)
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    prediction = np.dot(X, theta)
    mean_error = prediction - y

    return 1/(2*m) * np.sum(np.power(mean_error, 2))


def _logisticCostFunc(theta, X, y):
    """ compute cost for logistic regression

        Parameters:
        -----------
        theta : ndarray, shape (n_features,)
            Regression coefficients

        X : {array-like}, shape (n_samples, n_features)
            Training data. Should include intercept.

        y : ndarray, shape (n_samples,)
            Target values

        Returns
        -------
        cost : float
            cost evaluation using logistic cost function
    """

    # number of training examples
    m = len(y)
    y = y.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    J = 1/m * (np.dot(-y.T, np.log(sigmoid(np.dot(X, theta)))) -
               np.dot((1-y.T), np.log(1-sigmoid(np.dot(X, theta)))))

    return J


def compute_gradient(theta, X, y):
    """ Compute gradient. This will be passed to minimization functions

        Parameters:
        -----------
        theta : ndarray, shape (n_features,)
            Regression coefficients

        X : {array-like}, shape (n_samples, n_features)
            Training data. Should include intercept.

        y : ndarray, shape (n_samples,)
            Target values

        method : cost calculation method, default to 'RSS'
                Only RSS is supported for now

        Returns
        -------
        cost : float
    """
    m = len(y)
    y = y.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    grad = 1/m * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)

    return grad.ravel()


def compute_cost(theta, X, y, method='RSS'):
    """ Compute cost to be used in gradient descent

        Parameters:
        -----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Should include intercept.

        y : ndarray, shape (n_samples,)
            Target values

        theta : ndarray, shape (n_features,)
            Regression coefficients

        method : cost calculation method, default to 'RSS'
                Only RSS is supported for now

        Returns
        -------
        cost : float
    """
    print("cost method is {0}".format(method))
    if method == 'RSS':
        return _RSS(theta, X, y)
    elif method == 'logistic':
        return _logisticCostFunc(theta, X, y)
    else:
        raise ValueError("only 'RSS' and 'Logistic' methods are supported.")


def normalEqn(X, y):
    """ Computes the closed-form solution to linear regression
        using the normal equations.

        Parameters:
        -----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Should include intercept.

        y : ndarray, shape (n_samples,)
            Target values

        Returns
        -------
        theta : {array-like}, shape (n_features,)
    """

    theta = np.dot(np.dot(linalg.inv(np.dot(X.T, X)), X.T), y)

    return theta


def gradient_descent(X, y, theta, learning_rate, num_iters, cost_func='RSS',
                     ):
    """ Performs gradient descent to learn theta
        theta = gradient_descent(x, y, theta, alpha, num_iters) updates theta
                by taking num_iters gradient steps with learning rate alpha

        Parameters:
        -----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Should include intercept.

        y : ndarray, shape (n_samples,)
            Target values

        theta : ndarray, shape (n_features,)
            Regression coefficients

        learning_rate : float
            Controls the speed of convergence, a.k.a alpha

        cost_func : cost calculation method, default to 'RSS'
                Only RSS is supported for now

        num_iters : int
            Number of iterations

        Returns
        -------
        calculated theta : ndarray, shape (n_features,)
            Regression coefficients that minimize the cost function

        cost : ndarray, shape (num_iters,)
            Cost calculated for each iteration
    """

    print("running gradient descent algorithm...")

    # Initialize some useful values
    m = len(y)  # number of training examples
    cost = np.zeros((num_iters,))
    y = y.reshape(-1, 1)

    for i in range(num_iters):

        # Perform a single gradient step on the parameter vector
        prediction = np.dot(X, theta)  # m size vector
        mean_error = prediction - y  # m size vector
        theta = theta - learning_rate/m * np.dot(X.T, mean_error)

        # Save the cost J in every iteration
        cost[i] = compute_cost(X, y, theta)

    return theta, cost


if __name__ == "__main__":

    # load data
    train = pd.read_csv('ex2data1.txt',
                        names=['x1', 'x2', 'y'])  # pylint: disable-msg=C0103
    print('train.shape = {0}'.format(train.shape))
    X = train.ix[:, train.columns != 'y'].get_values()
    y = train['y'].get_values()
    print('Shape of X,y = ({0},{1})'.format(X.shape, y.shape))

    # scatter plot of admitted and non-admitted exam scores
    X_admitted = train.ix[train['y'] == 1, train.columns != 'y'].get_values()
    X_not_admitted = train.ix[train['y'] == 0,
                              train.columns != 'y'].get_values()
    admitted = plt.scatter(X_admitted[:, 0], X_admitted[:, 1],
                           color='b', marker='+')
    not_admitted = plt.scatter(X_not_admitted[:, 0], X_not_admitted[:, 1],
                               color='r', marker='o')
    plt.xlabel('exam 1 score')
    plt.ylabel('exam 2 score')
    plt.legend([admitted, not_admitted], ['admitted', 'Not admitted'])
    plt.show()

    # scale the input to zero mean and standard deviation of 1
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    print(X_scaled[:3, :])

    # add intercept term to X_scaled
    X_scaled = np.concatenate((np.ones((X.shape[0], 1)), X_scaled), axis=1)
    print(X_scaled.shape)
    print(X_scaled[:3, :])

    # Init Theta
    theta = np.zeros((3, ))

    print('cost = {}'.format(compute_cost(theta, X_scaled, y, 'logistic')))
    print('gradient = {}'.format(compute_gradient(theta, X_scaled, y)))

    theta_optimized = optimize.fmin_bfgs(_logisticCostFunc, theta,
                                         fprime=compute_gradient,
                                         args=(X_scaled, y))

    print('optimized theta with bfgs= {}'.format(theta_optimized))
