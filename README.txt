# Bayesian2D

This package enables finding the minimum or maximum point of any 2D function within the bounds specified. The package uses Bayesian optimization with the Maximum Probability of Improvement aquisition function. The algorithm makes educated guesses about which points in the function are smaller (if searching for a minimum) than the current point and evaluates them, creating a surrogate function using Gaussian Regression.
