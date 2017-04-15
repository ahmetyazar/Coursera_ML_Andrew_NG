import numpy as np
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

    # You need to return these values correctly
    error_train = np.zeros((len(X), 1))
    error_val = np.zeros((len(Xval), 1))

    initial_theta = np.zeros((X.shape[1],))

    for i in range(m):
        xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                                 jac=computeGradient, args=(X[:i+1, :],
                                                            y[:i+1], reg),
                                 options={'gtol': 1e-6, 'disp': False})
        error_train[i] = computeRSS(xopt.x, X[:i+1, :], y[:i+1], 0)
        error_val[i] = computeRSS(xopt.x, Xval, yval, 0)

    return error_train, error_val


def learningCurveWithShuffle(X, y, Xval, yval, reg=0, samplingSize=50):
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

    # You need to return these values correctly
    error_train = np.zeros((len(X), 1))
    error_val = np.zeros((len(Xval), 1))

    initial_theta = np.zeros((X.shape[1],))

    for i in range(m):
        for j in range(samplingSize):
            train_idx = np.random.choice(len(X), size=j+1)
            cv_idx = np.random.choice(len(Xval), size=j+1)
            xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                                     jac=computeGradient, 
                                     args=(X[train_idx, :], y[train_idx], reg),
                                     options={'gtol': 1e-6, 'disp': False})
            error_train[i] += computeRSS(xopt.x, X[train_idx, :], y[train_idx], 0)
            error_val[i] += computeRSS(xopt.x, Xval[cv_idx, :], yval[cv_idx], 0)
        
        error_train[i] = error_train[i] / samplingSize
        error_val[i] = error_val[i] / samplingSize

    return error_train, error_val


def validationCurve(X, y, Xval, yval):
    """ VALIDATIONCURVE Generate the train and validation errors needed to
        plot a validation curve that we can use to select lambda
        %   [lambda_vec, error_train, error_val] = ...
        %       VALIDATIONCURVE(X, y, Xval, yval) returns the train
        %       and validation errors (in error_train, error_val)
        %       for different values of lambda. You are given the training 
        set (X, y) and validation set (Xval, yval).
     """

    # Selected values of lambda
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # You need to return these variables correctly.
    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    initial_theta = np.zeros((X.shape[1],))
    for i in range(len(lambda_vec)):
        xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                             jac=computeGradient, args=(X, y, lambda_vec[i]),
                             options={'gtol': 1e-6, 'disp': False})
        error_train[i] = xopt.fun
        error_val[i] = computeRSS(xopt.x, Xval, yval, lambda_vec[i])
    
    return lambda_vec, error_train, error_val


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
             np.array(range(m))+1, error_val[:m])
    plt.title('Learning curve for linear regression')
    ax2.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.xticks(np.arange(1,m+1,2))
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
    print('LEARNING CURVE POLYNOMIAL REGRESSION WITH LAMBDA = 0\n')
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with 
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.
    #
   
    print('lambda = 0\n')
    reg = 0
    initial_theta = np.zeros((X_poly.shape[1],))
    xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                             jac=computeGradient, args=(X_poly, y, reg),
                             options={'gtol': 1e-6, 'disp': False})

    # Plot training data and fit
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.scatter(raw['X'], y, c='r', marker='x', linewidths=1.5)
    
    # plot polynomial fit
    x_plt = np.array(np.arange(min(raw['X'])-15,
                               max(raw['X'])+25, 0.05)).reshape(-1,1)
    print(x_plt.shape)
    x_plt_poly = poly.fit_transform(x_plt)
    print(x_plt_poly.shape)
    x_plt_poly = scaler.transform(x_plt_poly)
    print(x_plt_poly.shape)
    x_plt_poly = np.concatenate((np.ones((len(x_plt_poly), 1)), x_plt_poly), 
                                axis=1)
    print(x_plt_poly.shape)         
    print(xopt.x.shape)                  
    ax4 = fig2.add_subplot(111)
    ax4.plot(x_plt, np.dot(x_plt_poly, xopt.x))
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = {0:1.2f})'.format(reg))

    # Learning curve for polynomial regression
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, reg)

    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111)
    ax5.plot(np.arange(len(X_poly))+1, error_train,
             np.arange(len(X_poly))+1, error_val[:len(X_poly)])
    plt.title('Learning curve for polynomial regression with '
              'lambda {0}'.format(reg))
    ax5.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    # plt.xticks(np.arange(1,m+1,2))
    plt.ylabel('Error')

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{0}\t{1:3.4f}\t{2:3.4f}'.format(i+1,
              np.asscalar(error_train[i]), np.asscalar(error_val[i])))

   
    print('\n##########################################################')
    print('LEARNING CURVE POLYNOMIAL REGRESSION WITH LAMBDA = 1\n')
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with 
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.
    #
   
    print('lambda = 1\n')
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
    x_plt = np.array(np.arange(min(raw['X'])-15,
                               max(raw['X'])+25, 0.05)).reshape(-1,1)
    print(x_plt.shape)
    x_plt_poly = poly.fit_transform(x_plt)
    print(x_plt_poly.shape)
    x_plt_poly = scaler.transform(x_plt_poly)
    print(x_plt_poly.shape)
    x_plt_poly = np.concatenate((np.ones((len(x_plt_poly), 1)), x_plt_poly), 
                                axis=1)
    print(x_plt_poly.shape)         
    print(xopt.x.shape)                  
    ax4 = fig2.add_subplot(111)
    ax4.plot(x_plt, np.dot(x_plt_poly, xopt.x))
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = {0:1.2f})'.format(reg))

    # Learning curve for polynomial regression
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, reg)

    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111)
    ax5.plot(np.arange(len(X_poly))+1, error_train,
             np.arange(len(X_poly))+1, error_val[:len(X_poly)])
    plt.title('Learning curve for polynomial regression with '
              'lambda {0}'.format(reg))
    ax5.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    # plt.xticks(np.arange(1,m+1,2))
    plt.ylabel('Error')

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{0}\t{1:3.4f}\t{2:3.4f}'.format(i+1,
              np.asscalar(error_train[i]), np.asscalar(error_val[i])))



    print('\n##########################################################')
    print('Validation for Selecting Lambda\n')
    # test various values of lambda on a validation set and 
    # use this to select the "best" lambda value.

    lambda_vec, error_train, error_val = validationCurve(X_poly, y,
                                                         X_poly_val, yval)
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(111)
    ax6.plot(lambda_vec, error_train, lambda_vec, error_val)
    ax6.legend(['Train', 'Cross Validation'])
    plt.title('Validation for Selecting Lambda')
    # plt.xticks(lambda_vec)
    plt.xlabel('lambda')
    plt.ylabel('Error')
    
    print('lambda\tTrain Error\tValidation Error\n')
    for i in range(len(lambda_vec)):
        	print(' {0}\t{1:3.5f}\t\t{2:3.5f}\n'.format(lambda_vec[i],
               np.asscalar(error_train[i]), np.asscalar(error_val[i])))
 
    print('\n##########################################################')
    print('Compute the test error using the best value of λ\n')  
    best_lambda = 0.3  
    initial_theta = np.zeros((X_poly.shape[1],))
    xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                             jac=computeGradient,
                             args=(X_poly, y, best_lambda),
                             options={'gtol': 1e-6, 'disp': False})
    test_error = computeRSS(xopt.x, X_poly_test, ytest, best_lambda)
    print('For lambda={0}, test error is {1:3.5f}'.format(best_lambda,
          test_error))
 
    best_lambda = 1
    initial_theta = np.zeros((X_poly.shape[1],))
    xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                             jac=computeGradient,
                             args=(X_poly, y, best_lambda),
                             options={'gtol': 1e-6, 'disp': False})
    test_error = computeRSS(xopt.x, X_poly_test, ytest, best_lambda)
    print('For lambda={0}, test error is {1:3.5f}'.format(best_lambda,
          test_error))
  
    best_lambda = 3
    initial_theta = np.zeros((X_poly.shape[1],))
    xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                             jac=computeGradient,
                             args=(X_poly, y, best_lambda),
                             options={'gtol': 1e-6, 'disp': False})
    test_error = computeRSS(xopt.x, X_poly_test, ytest, best_lambda)
    print('For lambda={0}, test error is {1:3.5f}'.format(best_lambda,
          test_error))   
    
    
    
    print('\n##########################################################')
    print('LEARNING CURVE POLYNOMIAL REGRESSION WITH LAMBDA = 0.3\n')
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with 
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.
    #
   
    print('lambda = 0.3\n')
    reg = 0.3
    initial_theta = np.zeros((X_poly.shape[1],))
    xopt = optimize.minimize(computeRSS, initial_theta, method='BFGS',
                             jac=computeGradient, args=(X_poly, y, reg),
                             options={'gtol': 1e-6, 'disp': False})

    # Plot training data and fit
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.scatter(raw['X'], y, c='r', marker='x', linewidths=1.5)
    
    # plot polynomial fit
    x_plt = np.array(np.arange(min(raw['X'])-15,
                               max(raw['X'])+25, 0.05)).reshape(-1,1)
    print(x_plt.shape)
    x_plt_poly = poly.fit_transform(x_plt)
    print(x_plt_poly.shape)
    x_plt_poly = scaler.transform(x_plt_poly)
    print(x_plt_poly.shape)
    x_plt_poly = np.concatenate((np.ones((len(x_plt_poly), 1)), x_plt_poly), 
                                axis=1)
    print(x_plt_poly.shape)         
    print(xopt.x.shape)                  
    ax4 = fig2.add_subplot(111)
    ax4.plot(x_plt, np.dot(x_plt_poly, xopt.x))
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = {0:1.2f})'.format(reg))

    # Learning curve for polynomial regression
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, reg)

    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111)
    ax5.plot(np.arange(len(X_poly))+1, error_train,
             np.arange(len(X_poly))+1, error_val[:len(X_poly)])
    plt.title('Learning curve for polynomial regression with '
              'lambda {0}'.format(reg))
    ax5.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    # plt.xticks(np.arange(1,m+1,2))
    plt.ylabel('Error')

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{0}\t{1:3.4f}\t{2:3.4f}'.format(i+1,
              np.asscalar(error_train[i]), np.asscalar(error_val[i])))


    
    
    print('\n##########################################################')
    print('Learning curves with randomly selected examples\n')     
    # In practice, especially for small training sets, when you plot learning curves
    # to debug your algorithms, it is often helpful to average across multiple sets
    # of randomly selected examples to determine the training error and cross
    # validation error.
    
    # Learning curve for polynomial regression
    reg = 3
    averagingSize = 50
    error_train, error_val = learningCurveWithShuffle(X_poly, y, 
                                                      X_poly_val, yval, 
                                                      reg, averagingSize)

    fig5 = plt.figure()
    ax7 = fig5.add_subplot(111)
    ax7.plot(np.arange(len(X_poly))+1, error_train,
             np.arange(len(X_poly))+1, error_val[:len(X_poly)])
    plt.title('Learning curve for polynomial regression with '
              'lambda {0} using averaging of 50 samples'.format(reg))
    ax7.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    # plt.xticks(np.arange(1,len(X_poly)+1,2))
    plt.ylabel('Error')

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{0}\t{1:3.4f}\t{2:3.4f}'.format(i+1,
              np.asscalar(error_train[i]), np.asscalar(error_val[i])))
    
    
    reg = 1
    averagingSize = 50
    error_train, error_val = learningCurveWithShuffle(X_poly, y, 
                                                      X_poly_val, yval, 
                                                      reg, averagingSize)

    fig5 = plt.figure()
    ax7 = fig5.add_subplot(111)
    ax7.plot(np.arange(len(X_poly))+1, error_train,
             np.arange(len(X_poly))+1, error_val[:len(X_poly)])
    plt.title('Learning curve for polynomial regression with '
              'lambda {0} using averaging of 50 samples'.format(reg))
    ax7.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    # plt.xticks(np.arange(1,len(X_poly)+1,2))
    plt.ylabel('Error')

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{0}\t{1:3.4f}\t{2:3.4f}'.format(i+1,
              np.asscalar(error_train[i]), np.asscalar(error_val[i])))
    

    reg = 0.03
    averagingSize = 50
    error_train, error_val = learningCurveWithShuffle(X_poly, y, 
                                                      X_poly_val, yval, 
                                                      reg, averagingSize)

    fig5 = plt.figure()
    ax7 = fig5.add_subplot(111)
    ax7.plot(np.arange(len(X_poly))+1, error_train,
             np.arange(len(X_poly))+1, error_val[:len(X_poly)])
    plt.title('Learning curve for polynomial regression with '
              'lambda {0} using averaging of 50 samples'.format(reg))
    ax7.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    # plt.xticks(np.arange(1,len(X_poly)+1,2))
    plt.ylabel('Error')

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{0}\t{1:3.4f}\t{2:3.4f}'.format(i+1,
              np.asscalar(error_train[i]), np.asscalar(error_val[i])))
    