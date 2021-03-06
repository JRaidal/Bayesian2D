# Bayesian2D

This package implements Bayesian optimization in Python for any 2D function. It uses Gaussian regression to create a surrogate function and the Maximum Probability of Improvement aquisition function to pick points to evaluate, thus finding the specified extremum of the function in only a few hundred evaluations.

# How to install

The package can simply be installed with 'pip install Bayesian2D'.

# How to use

The package contains two directories- tools and tests. The tools folder contains all the separate python functions used by the algorithm, with the Bayesian2D function being the main function of the package.

To optimize your function just import 'from Bayesian2D.tools import Bayesian2D'. The function takes as an input the function you wish to optimize and the bounds in which you wish to search for the extremum (there are a few built in named functions such as 'Beale' or 'Ackley' with the Rosenbrock function being the default but custom functions can also be inserted). The function also requires you to specify the number of initial points evaluated, the number of optimization cycles run, the number of random points evaluated by the surrogate function each cycle, the exploration constant and whether you want to find the maximum or minimum.

# Testing

Unit tests for all the functions used can be found in the aforementioned tests directory.
