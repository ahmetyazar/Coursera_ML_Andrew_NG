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
               np.dot((1-y).T, np.log(1 - sigmoid(np.dot(X, theta)))))

    return np.asscalar(J)



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

        Returns
        -------
        gradient : ndarray, shape (n_features,)
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


def plotDecisionBoundary(theta, X, y, poly=False):
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
    plotData(X, y, ax1)

    if not poly:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1])-0.1,  np.max(X[:, 1])+0.1])

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (np.dot(theta[1], plot_x) + theta[0])

        # Plot, and adjust axes for better viewing
        ax1.plot(plot_x, plot_y)
    else:
        # Here is the grid range
        u_orig = np.linspace(-1, 1.5, 50)
        v_orig = np.linspace(-1, 1.5, 50)

        # create a dataframe with all combinations of u and v
        u = u_orig.repeat(len(u_orig)).reshape(-1, 1)
        v = v_orig.reshape(-1, 1).repeat(len(v_orig), axis=1).T.reshape(-1, 1)
        df = pd.DataFrame(np.concatenate((u, v), axis=1), columns=['c1', 'c2'])

        # create polynomial features
        poly = polynomial_features(df, columns=['c1', 'c2'], degree=6)

        # add intercept
        X = np.concatenate((np.ones((len(poly), 1)),
                            poly.get_values()), axis=1)

        # Evaluate z = theta*x over the grid
        z = np.dot(X, theta)

        z = z.reshape(len(u_orig), len(v_orig)) #  important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        CS = plt.contour(u_orig, v_orig, z)
        plt.clabel(CS, inline=1, fontsize=10)

    plt.show()


def polynomial_features(df, columns, degree=2, include_bias=False,
                        copy=True):
    """ Convert the columns provided a san input to polynomial features.
        Columns in the input dataframe not part of "columns" parameters
        are not touched.

        Parameters
        ----------
        df : pandas DataFrame
            The training/test input samples.
        columns: list of columns where polynomial features will be generated
        degree: degree of the polynomial that will be generated
        include_bias: include bias or not
        copy: leave the input dataframe intact when generating output dataframe

        Returns
        -------
        Dataframe with "columns" replaced with polynomials
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree, include_bias=include_bias)

    X = df[columns].get_values()
    X_poly = poly.fit_transform(X)

    target_feature_names = []
    power_columns = [zip(df[columns], p) for p in poly.powers_]

    for power_column in power_columns:
        powers = []
        for pair in power_column:
            if pair[1] != 0:
                if pair[1] == 1:
                    powers.append('{}'.format(pair[0]))
                else:
                    powers.append('{}^{}'.format(pair[0], pair[1]))
        target_feature_names.append('x'.join(powers))

    df_poly = pd.DataFrame(X_poly, columns=target_feature_names)

    if copy:
        df_output = df.copy()
    else:
        df_output = df

    df_output.drop(columns, axis=1, inplace=True)
    df_output.reset_index(inplace=True)

    return pd.concat((df_output, df_poly), axis=1)

Nfeval = 1

def callbackF(Xi):
    global Nfeval
    global X_scaled, y
    g = compute_gradient(Xi, X_scaled, y)
    print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval,
           _logisticCostFunc(Xi, X_scaled, y), g[0], g[1], g[2]))
    Nfeval += 1

if __name__ == "__main__":

    # load data
    train = pd.read_csv('ex2data2.txt',
                        names=['x1', 'x2', 'y'])  # pylint: disable-msg=C0103
    print('train.shape = {0}'.format(train.shape))

    X = train.ix[:, train.columns != 'y'].get_values()
    y = train['y'].get_values()
    print('Shape of X,y = ({0},{1})'.format(X.shape, y.shape))

    plotData(X, y)

    # add additional features
    train_poly = polynomial_features(train.ix[:, train.columns != 'y'],
                                     ['x1', 'x2'], degree=6)

    # scale the input to zero mean and standard deviation of 1
    # scaler = preprocessing.StandardScaler()
    # scaler.fit(train_poly.get_values())
    # X_scaled = scaler.transform(train_poly.get_values())

    # commented out scaling. It creates very small values causing issues with 
    # minimization
    X_scaled = train_poly.get_values()
    
    # add intercept term to X_scaled
    X_scaled = np.concatenate((np.ones((X.shape[0], 1)),
                               X_scaled), axis=1)
    print('Shape of X_scaled: {0}'.format(X_scaled.shape))

    # Init Theta
    theta = np.zeros((X_scaled.shape[1], ))

    print('cost = {}'.format(compute_cost(theta, X_scaled, y, 'logistic')))
    print('gradient = {}'.format(compute_gradient(theta, X_scaled, y)))

    [xopt, fopt, gopt, Bopt, func_calls, grad_calls, 
     warnflg]  = optimize.fmin_bfgs(_logisticCostFunc, theta,
                                         fprime=compute_gradient,
                                         args=(X_scaled, y), 
                                         callback=callbackF, 
                                         maxiter=2000, 
                                         full_output=True, 
                                         retall=False)

    print('optimized theta with bfgs= {}'.format(xopt))

    # calculate minimum cost
    # print('minimum cost: {0}'.format(
    #        _logisticCostFunc(theta_optimized, X_scaled, y)))

    # plot data wiht a decision boundary
    plotDecisionBoundary(xopt, X, y, poly=True)
