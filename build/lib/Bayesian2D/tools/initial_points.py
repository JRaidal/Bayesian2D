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

