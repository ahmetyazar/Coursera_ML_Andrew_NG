import numpy as np
import pandas as pd
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy import optimize
from sklearn import preprocessing

def computeRSS(theta, X, y, reg=0):
    # number of training examples
    m = len(y)
    if hasattr(theta, 'reshape'):
        theta = theta.reshape(-1, 1)
    if hasattr(y, 'reshape'):    
        y = y.reshape(-1, 1)

    prediction = np.dot(X, theta)
    mean_error = prediction - y

    RSS = 1/(2*m) * (np.sum(np.power(mean_error, 2)) +
                     reg * np.sum(np.power(theta[1:], 2)))

    return np.asscalar(RSS)


def computeGradient(theta, X, y, reg=0):
    # number of training examples
    m = len(y)

    # print("Number of samples in computeGradient(): {}".format(m))
    if hasattr(theta, 'reshape'):
        theta = theta.reshape(-1, 1)
    if hasattr(y, 'reshape'):
        y = y.reshape(-1, 1)

    prediction = np.dot(X, theta)
    mean_error = prediction - y

    grad = 1/m * (np.dot(X.T, mean_error) +
                  reg * np.concatenate((np.zeros((1, 1)), theta[1:]), axis=0))

    return grad.ravel()

Nfeval = 1


def callbackF(Xi):
    global Nfeval
    global X, y
    g = computeGradient(Xi, X, y)
    print('{0:3d}\t{1: 3.4f}\t{2: 3.4f}\t{3: 3.4f}'.format(Nfeval,
          computeRSS(Xi, X, y), g[0], g[1]))
    Nfeval += 1


def learningCurve(X, y, Xval, yval, reg=0):
    """ LEARNINGCURVE Generates the train and cross validation set errors needed
    %to plot a learning curve
    %   [error_train, error_val] = ...
    %       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
    %       cross validation set errors for a learning curve. In particular,
    %       it returns two vectors of the same length - error_train and
    %       error_val. Then, error_train(i) contains the training error for
    %       i examples (and similarly for error_val(i)).
    %
    %   In this function, you will compute the train and test errors for
    %   dataset sizes from 1 up to m. In practice, when working with larger
    %   datasets, you might want to do this in larger intervals.
    """

    # Number of training examples
    m = len(X)

    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return training errors in
#               error_train and the cross validation errors in error_val.
#               i.e., error_train(i) and
#               error_val(i) should give you the errors
#               obtained after training on i examples.
#
# Note: You should evaluate the training error on the first i training
#       examples (i.e., X(1:i, :) and y(1:i)).
#
#       For the cross-validation error, you should instead evaluate on
#       the _entire_ cross validation set (Xval and yval).
#
# Note: If you are using your cost function (linearRegCostFunction)
#       to compute the training and cross validation error, you should
#       call the function with the lambda argument set to 0.
#       Do note that you will still need to use lambda when running
#       the training to obtain the theta parameters.

    for i in range(m):
        xopt = optimize.minimize(computeRSS, theta, method='BFGS',
                                 jac=computeGradient, args=(X[:i+1, :],
                                                            y[:i+1], reg),
                                 options={'gtol': 1e-6, 'disp': False})
        error_train[i] = computeRSS(xopt.x, X[:i+1, :], y[:i+1], 0)
        error_val[i] = computeRSS(xopt.x, Xval, yval, 0)

    return error_train, error_val


if __name__ == "__main__":
    # load data
    raw = loadmat('ex5data1.mat')
    X = raw['X']
    y = raw['y'].ravel()
    Xval = raw['Xval']
    yval = raw['yval'].ravel()
    Xtest = raw['Xtest']
    ytest = raw['ytest'].ravel()
    print("Size of the training set:{}".format(len(X)))
    print("Size of the valuation set:{}".format(len(Xval)))
    print("Size of the test set:{}".format(len(Xtest)))

    # Plot training data
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(X, y, c='r', marker='x', linewidths=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    m = len(X)
    # add intercept of ones to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    Xval = np.concatenate((np.ones((len(Xval), 1)), Xval), axis=1)

    # Compute RSS and gradient
    theta = np.array([1, 1]).ravel()
    J = computeRSS(theta, X, y, 1)
    grad = computeGradient(theta, X, y, 1)

    print('Cost at theta = [1 ; 1]: {} \n(this value should be '
          'about 303.993192)\n'.format(J))
    print('Gradient at theta = [1 ; 1]:  [{0}; {1}]\n(this value should be '
          'about [-15.303016; 598.250744])\n'.format(grad[0], grad[1]))

    print('\n##########################################################')
    print('CALCULATE MINIMUM VALUE USING BFGS\n')

    #  Train linear regression with regularization parameter lambda = 0
    print('{0}\t{1}\t{2}\t{3}'.format('iter', 'cost',
          'grad[0]', 'grad[1]'))
    xopt = optimize.minimize(computeRSS, theta, method='BFGS',
                             jac=computeGradient, args=(X, y),
                             callback=callbackF,
                             options={'gtol': 1e-6, 'disp': True})

    print('optimized theta with bfgs= {}'.format(xopt.x))

    # Plot, and adjust axes for better viewing
    ax1.plot(raw['X'], np.dot(X, xopt.x))
    plt.show()

    #  Part 5: Learning Curve for Linear Regression
    #  Write Up Note: Since the model is underfitting the data, we expect to
    #                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
    #

    print('\n##########################################################')
    print('LEARNING CURVE FOR LINEAR MODEL\n')
    reg = 0
    error_train, error_val = learningCurve(X, y, Xval, yval, reg)

    fig1 = plt.figure()
    ax2 = fig1.add_subplot(111)
    ax2.plot(np.array(range(m))+1, error_train,
             np.array(range(m))+1, error_val)
    plt.title('Learning curve for linear regression')
    ax2.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{0}\t{1:3.4f}\t{2:3.4f}'.format(i+1,
              np.asscalar(error_train[i]), np.asscalar(error_val[i])))

    print('\n##########################################################')
    print('POLYNOMIAL REGRESSION\n')

    degree = 8
    scaler = preprocessing.StandardScaler()

    # Map X onto Polynomial Features and Normalize
    poly = preprocessing.PolynomialFeatures(degree, include_bias=False)
    X_poly = poly.fit_transform(raw['X'])
    scaler.fit(X_poly)
    X_poly = scaler.transform(X_poly)
    X_poly = np.concatenate((np.ones((len(X_poly), 1)),
                             X_poly), axis=1)  # Add Ones

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = poly.fit_transform(raw['Xtest'])
    X_poly_test = scaler.transform(X_poly_test)
    X_poly_test = np.concatenate((np.ones((len(X_poly_test), 1)),
                                  X_poly_test), axis=1)  # Add Ones

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = poly.fit_transform(raw['Xval'])
    X_poly_val = scaler.transform(X_poly_val)
    X_poly_val = np.concatenate((np.ones((len(X_poly_val), 1)),
                                 X_poly_val), axis=1)  # Add Ones

    print('Normalized Training Example 1:\n')
    print(X_poly[0, :])

    print('\nNormalized Testing Example 1:\n')
    print(X_poly_test[0, :])
    
    print('\nNormalized cross validation Example 1:\n')
    print(X_poly_val[0, :])
    
    print('\n##########################################################')
    print('LEARNING CURVE POLYNOMIAL REGRESSION\n')
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with 
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.
    #
   
    reg = 1
    initial_theta = np.zeros((X_poly.shape[1],))
    xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                             jac=computeGradient, args=(X_poly, y, reg),
                             options={'gtol': 1e-6, 'disp': False})

    # Plot training data and fit
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.scatter(raw['X'], y, c='r', marker='x', linewidths=1.5)
    
    # plot polynomial fit
    #x_plt = np.array(np.arange(min(raw['X'])-15, max(raw['X'])+25, 0.05))
    #x_plt_poly = poly.fit_transform(x_plt)
    #x_plt_poly = scaler.transform(x_plt_poly)
    #x_plt_poly = np.concatenate((np.ones((len(x_plt_poly), 1)), x_plt_poly), 
    #                            axis=1)
                                   
#    ax4 = fig2.add_subplot(111)
#    ax4.plot(raw['X'], np.dot(x_plt_poly, xopt.x).ravel(),
#             c='r', marker='x', linewidths=1.5)
#    plt.xlabel('Change in water level (x)')
#    plt.ylabel('Water flowing out of the dam (y)')
#    plt.title(print('Polynomial Regression Fit (lambda = {0:1.2f})', reg))


