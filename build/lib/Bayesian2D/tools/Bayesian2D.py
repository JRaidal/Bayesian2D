def Bayesian2D(x_bounds, y_bounds, starting_n, n_random, iterations, max_min, exploration, function = 'Rosenbrock' ):
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
    n_random : int
        Number of random points to be created.
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
    
    x, y, z_best = results(objective, acquisition, starting_n, n_random, x_bounds, y_bounds, iterations, e, model, max_min)
    
    return x, y, z_best
