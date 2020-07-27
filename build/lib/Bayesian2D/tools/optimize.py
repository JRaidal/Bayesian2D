def optimize(opt_func, aquisition_func, starting_n, n_random, x_bounds, y_bounds, iterations, e, model, max_min):
    '''
    

    Parameters
    ----------
    opt_func : function
        Function that is optimized.
    aquisition_func : function
        Function used to pick points to evaluate.
    starting_n : int
        Initial number of random points evaluated.
    n_random : int
        Number of random points to be created.
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
        x, y, *rest= aquisition_func(XY, x_bounds, y_bounds, e, model, max_min, n_random)
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
