def acquisition(XY, x_bounds, y_bounds, e, model, max_min, n_random):
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
    n_random : int
        Number of random points to be created.
    
    Returns
    -------
    X_best : float
        x-coordinate of point with maximum probability of improvement.
    Y_best : float
        y-coordinate of point with maximum probability of improvement.

    '''

    # Find the best surrogate mean found so far
    z_surrogate, _ = surrogate(model, XY)
    if max_min == 'maximum':
        best = numpy.max(z_surrogate)
    if max_min == 'minimum':
        best = numpy.min(z_surrogate)
    
    #Extract random points
    XYsamples = create_random(x_bounds, y_bounds, n_random)
    
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
