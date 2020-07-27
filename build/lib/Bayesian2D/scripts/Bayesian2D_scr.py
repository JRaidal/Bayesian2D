

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:56:37 2020

@author: jraidal
"""


from datetime import datetime
import itertools
import numpy
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import uniform
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF, ExpSineSquared, RationalQuadratic, WhiteKernel
import sys


# Define objective function

def functions(x, y, function):
    '''
    Defines and evaluates functions

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.
    function : str
        Either the name of one of the built in functions or a custom 2D function.

    Returns
    -------
    TYPE float
        Returns function value based on input x and y \.

    '''
    if function == 'Beale':
      #Beale
        return (1.5-x+x*y)**2+(2.25-x+x*y**2)**2+(2.625-x+x*y**3)**2
    if function == 'Goldstein-Price':
      #Goldstein-Price
        return (1+(x+y+1)**2 * (19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2 * (18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    if function == 'Rosenbrock':
      #Rosenbrock
        return ((1 - x)**2 + 100*(y - x**2)**2)
    if function == 'Ackley':
      #Ackley
        return -20*numpy.exp(-0.2*numpy.sqrt(0.5*(x**2 + y**2)))-numpy.exp(0.5*(numpy.cos(2*numpy.pi*x) + numpy.cos(2*numpy.pi*y)))+numpy.exp(1)+20
    else:
        return eval(function)


# Surrogate model
def surrogate(model, XY): 
    '''
    Predicts the mean and standard deviation of points using Gaussian processes

    Parameters
    ----------
    model : sklearn.gaussian_process
        Some Gaussian process model.
    XY : numpy array
        Array of x and y coordinates.

    Returns
    -------
    array, array
        Returns mean and standard deviation arrays for evaluated points.

    '''
    return model.predict(XY, return_std=True)

# Maximum probability of improvement acquisition function
def acquisition(XY, x_bounds, y_bounds, e, model, max_min):
    '''
    Creates sample points and finds the one most likely to improve the function when
    evaluating.

    Parameters
    ----------
    XY : numpy array
        Array of all points evaluated so far.
    x_bounds : list
        Two element list of x-axis boundaries for the function.
    y_bounds : list
        Two element list of y-axis boundaries for the function.
    e : float
        Exploration parameter.
    model : sklearn.gaussian_process
        Some Gaussian process model.
    max_min : str
        Specifies whether the algorithm is searching for maxima or minima.

    Returns
    -------
    X_best : float
        x-coordinate of point with maximum probability of improvement.
    Y_best : float
        y-coordinate of point with maximum probability of improvement.

    '''
    # Unpack bounds
    x1, x2 = x_bounds
    y1, y2 = y_bounds
    
    # Find the best surrogate mean found so far
    z_surrogate, _ = surrogate(model, XY)
    if max_min == 'maximum':
        best = numpy.max(z_surrogate)
    if max_min == 'minimum':
        best = numpy.min(z_surrogate)
    
    # Create random sample points
    Xsamples = ([])
    Ysamples = ([])
    for i in range(100):
        a = uniform(x1,x2)
        Xsamples.append(a)
        b = uniform(y1,y2)
        Ysamples.append(b)
    Xsamples = numpy.array(Xsamples)
    Ysamples = numpy.array(Ysamples)
    XYsamples=numpy.vstack((Xsamples, Ysamples)).T
    
    # Find the mean and standard deviation of the sample points
    mu, std = surrogate(model, XYsamples)
    
    # Calculate the maximum probability of improvement
    r=(mu-best)
    c=(r)/(std+1e-9)
    with catch_warnings():
        # Ignore scaling warnings (not true)
        simplefilter("ignore")
        c= preprocessing.scale(c)  
    scores=norm.cdf(c - e)
    
    # Find point with best score
    if max_min == 'maximum':
          index_max = (numpy.argwhere(scores == numpy.max(scores)))
    if max_min == 'minimum':
          index_max = (numpy.argwhere(scores == numpy.min(scores)))
    
    ix_max = index_max[0,0]
    X_max, Y_max = XYsamples[ix_max]
    X_best = float(X_max)
    Y_best = float(Y_max)
    
    return X_best, Y_best

# plot real observations
def plot(plot_func, x_bounds, y_bounds):
    '''
    Plots the function which is optimized

    Parameters
    ----------
    plot_func : function
        The function to be plotted.
    x_bounds : list
        Two element list of x-axis boundaries for the function.
    y_bounds : list
        Two element list of y-axis boundaries for the function.

    Returns
    -------
    None.

    '''
    #Unpack bounds
    x1, x2 = x_bounds
    y1, y2 = y_bounds
    
    #Plot the function that is optimized
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Xsamples = numpy.linspace(x1, x2, 500)
    Ysamples = numpy.linspace(y1, y2, 500)
    X, Y = numpy.meshgrid(Xsamples, Ysamples)
    Z = plot_func(X, Y)
    ax.plot_surface(X,Y,Z, cmap = 'jet' )
    ax.view_init(45, 45)
    plt.show()


# Set domain and evaluations per cycle
def initial_points(starting_n, opt_func, x_bounds, y_bounds):
    '''
    Picks and evaluates random initial points from the function

    Parameters
    ----------
    starting_n : int
        Number of initial points picked.
    opt_func : function
        The function from which the points are picked.
    x_bounds : list
        Two element list of x-axis boundaries for the function.
    y_bounds : list
        Two element list of y-axis boundaries for the function.

    Returns
    -------
    XY : numpy array
        x and y coordinates for the inital points.
    z : numpy array
        Function values for inital points.

    '''
    #Unpack bounds
    x1, x2 = x_bounds
    y1, y2 = y_bounds
    
    #Pick random points within bounds
    X = ([])
    for i in range(0, starting_n):
        a = uniform(x1, x2)
        X.append(a)
    Y = ([])
    for i in range(0, starting_n):
        b = uniform(y1, y2)
        Y.append(b)
    X=numpy.array(X)
    Y=numpy.array(Y)
    XY=numpy.vstack((X, Y)).T
    z= opt_func(X, Y)
    
    return XY, z

    
def fit_model(model, data_input, data_output):
    '''
    Fits new data to the model

    Parameters
    ----------
    model : sklearn.gaussian_process
        Some Gaussian process model.
    data_input : numpy array
        x and y coordinates to be fitted.
    data_output : numpy array
        Corresponding function values to the x and y coordinates.

    Returns
    -------
    None.

    '''
    model.fit(data_input, data_output)

    
def optimize(opt_func, aquisition_func, starting_n, x_bounds, y_bounds, iterations, e, model, max_min):
    '''
    

    Parameters
    ----------
    opt_func : function
        Function that is optimized.
    aquisition_func : function
        Function used to pick points to evaluate.
    starting_n : int
        Initial number of random points evaluated.
    x_bounds : list
        Two element list of x-axis boundaries for the function.
    y_bounds : list
        Two element list of y-axis boundaries for the function.
    iterations : int
        Number of times optimization is run.
    e : float
        Exploration parameter.
    model : sklearn.gaussian_process
        Some Gaussian process model.
    max_min : str
        Specifies whether the algorithm is searching for maxima or minima.

    Returns
    -------
    XY : numpy array
        Array of all points evaluated.
    z : numpy array
        Value of all points evaluated.

    '''
    #Unpack inital points
    XY, z = initial_points(starting_n, opt_func, x_bounds, y_bounds)
    # Perform the optimization process
    
    for i in range(iterations):
        # Select the next point to sample
        x, y, *rest= aquisition_func(XY, x_bounds, y_bounds, e, model, max_min)
        XYmingi = numpy.array(([x, y]))
        XYmingi = XYmingi.reshape(1, -1)
        
        # Sample the point
        actual = opt_func(x, y)
        
        # Show process
        print(f'{i+1}/{iterations} completed')
        print(f'Currently evaluating x=%.3f y=%.3f with value of z=%.4f' % (x, y, actual))
        
        # Add the data to the dataset
        XYnew_element = numpy.array(([x, y]))
        XY=numpy.vstack((XY, XYnew_element))
        z = list(z)
        z.append(actual)
        z = numpy.array(z)
        
        #Show current best result
        if max_min == 'maximum':
            z_best = numpy.max(z)
        if max_min == 'minimum':
            z_best = numpy.min(z)
        print(f'Current {max_min} found is z = {z_best}')
        
        # Update the model with new data
        fit_model(model, XY, z)
    
    return XY, z
    
def results(opt_function, acquisition, starting_n, x_bounds, y_bounds, iterations, e, model, max_min): 
    '''
    Returns the results of the optimization process

    Parameters
    ----------
    opt_function : function
        Function that is optimized.
    aquisition_func : function
        Function used to pick points to evaluate.
    starting_n : int
        Initial number of random points evaluated.
    x_bounds : list
        Two element list of x-axis boundaries for the function.
    y_bounds : list
        Two element list of y-axis boundaries for the function.
    iterations : int
        Number of times optimization is run.
    e : float
        Exploration parameter.
    model : sklearn.gaussian_process
        Some Gaussian process model.
    max_min : str
        Specifies whether the algorithm is searching for maxima or minima.

    Returns
    -------
    x : float
        x-coordinate of minimum/maximum found.
    y : float
        y-coordinate of minimum/maximum found.
    z_best : float
        Value of the function at x, y coordinates found.

    '''
    #Start timer
    startTime = datetime.now()
    #Optimize and unpack XY and corresponding z
    XY, z = optimize(opt_function, acquisition, starting_n, x_bounds, y_bounds, iterations, e, model, max_min)
    # Find best result
    if max_min == 'maximum':
        index = (numpy.argwhere(z == numpy.max(z)))
        z_best = numpy.max(z)
        ix = index[0, 0]
    if max_min == 'minimum':
        index = (numpy.argwhere(z == numpy.min(z)))
        z_best = numpy.min(z)
        ix = index[0,0]
    x, y = XY[ix]
    
    #Unpack all XY and z evaluated and plot as scatter plot
    Xfinal=([])
    Yfinal=([])
    for x, y in XY:
        Xfinal.append(x)
        Yfinal.append(y)
    Xfinal=numpy.array(Xfinal)
    Yfinal=numpy.array(Yfinal)
    Z = z
    ax = plt.axes(projection='3d')
    ax.scatter(Xfinal, Yfinal, Z, linewidth=0.5)
    ax.view_init(45, 45)
    plt.show()
    
    #Print results and time taken
    print('')
    print('The %s found is at x=%f, y=%f with a value of z=%f' % (max_min, x, y, z_best))
    print('Time elapsed', datetime.now() - startTime)
    print('Time per iteration', (datetime.now() - startTime)/iterations)
    return x, y, z_best

def Bayesian2D(x_bounds, y_bounds, starting_n, iterations, max_min, exploration, function = 'Rosenbrock' ):
    '''
    Combines all the functions in the package to find the maximum/minimum of any 2D
    function specified.

    Parameters
    ----------
    x_bounds : list
        Two element list of x-axis boundaries for the function.
    y_bounds : list
        Two element list of y-axis boundaries for the function.
    starting_n : int
        Initial number of random points evaluated.
    iterations : int
        Number of times optimization is run.
    max_min : str
        Specifies whether the algorithm is searching for maxima or minima.
    exploration : float
        Exploration parameter.
    function : str, optional
        Either the name of one of the built in functions or a custom 2D function. 
        The default is 'Rosenbrock'.

    Returns
    -------
    function
        Result function.

    '''
    def objective(x, y):
        '''
        Returns only the function to be optimized so that it needn't be 
        specified each time'
    
        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
    
        Returns
        -------
        function
            The function to be optimized.
    
        '''
        return functions(x, y, function)
    #Randomize seed
    numpy.random.seed()
      
    #Set parameters for surrogate model and aquisition function
    e=exploration
    model = GaussianProcessRegressor(kernel= Matern(), alpha = 1e-10)
    #Create initial random set of points
    XY_initial, z_initial = initial_points(starting_n, objective, x_bounds, y_bounds)
    
    #fit initial points to model
    fit_model(model, XY_initial, z_initial)
    
    #Plot function before optimization
    plot(objective, x_bounds, y_bounds)
    
    #Find results
    
    return results(objective, acquisition, starting_n, x_bounds, y_bounds, iterations, e, model, max_min)

