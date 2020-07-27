def create_random(x_bounds, y_bounds, n_random):
    '''
    Uniformly creates the specified number of random points within specified bounds.

    Parameters
    ----------
    x_bounds : list
        Two element list of x-axis boundaries for the function.
    y_bounds : list
        Two element list of y-axis boundaries for the function.
    n_random : int
        Number of random points to be created.

    Returns
    -------
    XYsamples : numpy array
        Array of created random sample points.

    '''
    # Unpack bounds
    x1, x2 = x_bounds
    y1, y2 = y_bounds
    
    # Create random sample points
    Xsamples = uniform(x1,x2, n_random)
    Ysamples = uniform(y1,y2, n_random)
    XYsamples=numpy.vstack((Xsamples, Ysamples)).T
    
    return XYsamples