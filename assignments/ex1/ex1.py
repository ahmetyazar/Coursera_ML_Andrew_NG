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
import matplotlib as mpl


def _RSS(X, y, theta):
    # number of training examples
    m = len(y)

    prediction = np.dot(X, theta)

    mean_error = prediction - y

    return 1/(2*m) * np.sum(np.power(mean_error, 2))


def compute_cost(X, y, theta, method='RSS'):
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
    if method != 'RSS':
        raise ValueError("only 'RSS' method is supported.")

    return _RSS(X, y, theta)


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
        theta = theta - alpha/m * np.dot(X.T, mean_error)

        # Save the cost J in every iteration
        cost[i] = compute_cost(X, y, theta)

    return theta, cost


if __name__ == "__main__":
    train = pd.read_csv('ex1data2.txt',
                        names=['x1', 'x2', 'y'])  # pylint: disable-msg=C0103
    print('train.shape = {0}'.format(train.shape))
    X = train.ix[:, train.columns != 'y'].get_values()
    y = train['y'].get_values()
    print('Shape of X,y = ({0},{1})'.format(X.shape, y.shape))

    # scale the input to zero mean and standard deviation of 1
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    print(X_scaled[:3, :])

    # add intercept term to X_scaled
    X_scaled = np.concatenate((np.ones((X.shape[0], 1)), X_scaled), axis=1)
    print(X_scaled.shape)
    print(X_scaled[:3, :])

    ##############################################
    #       Gradient descent                     #
    ##############################################

    print('Running gradient descent ...\n')

    # Choose some alpha value
    alpha = 1
    num_iters = 400

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    theta, cost = gradient_descent(X_scaled, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.plot(range(cost.shape[0]), cost)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")
    plt.show()

    # Display gradient descent's result
    print('Theta computed from gradient descent: {}'.format(theta))

    # Estimate the price of a 1650 sq-ft, 3 br house
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.
    X_sample = np.array([1650, 3]).reshape(1, 2)
    X_sample_scaled = scaler.transform(X_sample)
    X_sample_scaled = np.concatenate((np.ones((1, 1)),
                                      X_sample_scaled), axis=1)

    price = np.dot(X_sample_scaled, theta)
    print('Predicted price of a 1650 sq-ft, 3 br house'
          'using gradient descent:{}'.format(price))

    ##############################################
    #        Normal Equation                    ##
    ##############################################

    print('Running normal equation ...\n')

    # Calculate the parameters from the normal equation
    theta = normalEqn(X_scaled, y)
    print('Theta computed from the normal equations: {}'.format(theta))

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.dot(X_sample_scaled, theta)
    print('Predicted price of a 1650 sq-ft, 3 br house'
          'using gradient descent:{}'.format(price))
