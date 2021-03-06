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
from sklearn import metrics


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


def plotData(X, y, ax1=None):
    train_y = pd.DataFrame(y, columns=['y'])
    train = pd.concat((pd.DataFrame(X), train_y), axis=1)
    # scatter plot of admitted and non-admitted exam scores
    X_admitted = train.ix[train['y'] == 1, train.columns != 'y'].get_values()
    X_not_admitted = train.ix[train['y'] == 0,
                              train.columns != 'y'].get_values()

    if ax1 is None:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
    admitted = ax1.scatter(X_admitted[:, 0], X_admitted[:, 1],
                           color='b', marker='+')
    not_admitted = ax1.scatter(X_not_admitted[:, 0], X_not_admitted[:, 1],
                               color='r', marker='o')
    plt.xlabel('exam 1 score')
    plt.ylabel('exam 2 score')
    plt.legend([admitted, not_admitted], ['admitted', 'Not admitted'])


def plotDecisionBoundary(theta, X, y):
    """Plots the data points X and y into a new figure with
        the decision boundary defined by theta

    plotDecisionBoundary(theta, X,y) plots the data points with + for the
    positive examples and o for the negative examples. X is assumed to be
    a either
    1) Mx3 matrix, where the first column is an all-ones column for the
      intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
        
    # Plot Data
    plotData(X[:,1:3], y, ax1)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1])-0.1,  np.max(X[:, 1])+0.1])

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (np.dot(theta[1], plot_x) + theta[0])

        # Plot, and adjust axes for better viewing
        ax1.plot(plot_x, plot_y)
    
    plt.show()


if __name__ == "__main__":

    # load data
    train = pd.read_csv('ex2data1.txt',
                        names=['x1', 'x2', 'y'])  # pylint: disable-msg=C0103
    print('train.shape = {0}'.format(train.shape))
    X = train.ix[:, train.columns != 'y'].get_values()
    y = train['y'].get_values()
    print('Shape of X,y = ({0},{1})'.format(X.shape, y.shape))

    plotData(X, y)

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

    # calculate minimum cost
    print('minimum cost: {0}'.format(
            _logisticCostFunc(theta_optimized, X_scaled, y)))

    # plot data wiht a decision boundary
    plotDecisionBoundary(theta_optimized, X_scaled, y)

    # estimate admission probability for a student
    x_sample = scaler.transform(np.array([45, 85]).reshape(1, -1))

    # add intercept term to X_sample
    x_sample = np.concatenate((np.ones((1, 1)), x_sample), axis=1)
    prob = sigmoid(np.dot(x_sample, theta_optimized))
    print('For a student with scores 45 and 85, we predict an admission '
          'probability of {}'.format(prob))

    # compute accuracy on our training set
    prediction = sigmoid(np.dot(X_scaled, theta_optimized)) >= 0.5
    print('Accuracy is {0}'.format(sum(prediction == y)/y.shape[0]))
    print('Accuracy is {0}'.format(metrics.accuracy_score(y, prediction)))
    
    print('classification report:')
    print(metrics.classification_report(y, prediction))
