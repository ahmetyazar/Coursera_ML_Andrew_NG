import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


def computeRSS(theta, X, y, reg=0):
    # number of training examples
    m = len(y)
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    prediction = np.dot(X, theta)
    mean_error = prediction - y

    RSS = 1/(2*m) * (np.sum(np.power(mean_error, 2)) +
                     reg * np.sum(np.power(theta[1:], 2)))

    return RSS


def computeGradient(theta, X, y, reg=0):
    # number of training examples
    m = len(y)
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    prediction = np.dot(X, theta)
    mean_error = prediction - y

    grad = 1/m * (np.dot(X.T, mean_error) +
                  reg * np.concatenate((np.zeros((1, 1)), theta[2:]), axis=0))

    return grad

# load data
raw = loadmat('ex5data1.mat')
X = raw['X']
y = raw['y']
Xval = raw['Xval']
yval = raw['yval']
Xtest = raw['Xtest']
ytest = raw['ytest']
print("Size of the training set:{}".format(len(X)))
print("Size of the valuation set:{}".format(len(Xval)))
print("Size of the test set:{}".format(len(Xtest)))

# Plot training data
plt.scatter(X, y, c='r', marker='x', linewidths=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

# add intercept of ones to X
m = len(X)
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Compute RSS and gradient
theta = np.array([1, 1])
J = computeRSS(theta, X, y, 1)
grad = computeGradient(theta, X, y, 1)

print('Cost at theta = [1 ; 1]: {} \n(this value should be '
      'about 303.993192)\n'.format(J))
print('Gradient at theta = [1 ; 1]:  [{0}; {1}]\n(this value should be '
      'about [-15.303016; 598.250744])\n'.format(grad[0], grad[1]))
