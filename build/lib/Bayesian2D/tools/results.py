def results(opt_function, acquisition, starting_n, n_random, x_bounds, y_bounds, iterations, e, model, max_min): 
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
    XY, z = optimize(opt_function, acquisition, starting_n, n_random, x_bounds, y_bounds, iterations, e, model, max_min)
    # Find best result
    if max_min == 'maximum':
        index = (numpy.argwhere(z == numpy.max(z)))
        z_best = numpy.max(z)
        ix = index[0, 0]
    if max_min == 'minimum':
        index = (numpy.argwhere(z == numpy.min(z)))
        z_best = numpy.min(z)
        ix = index[0,0]
    x_best, y_best = XY[ix]
    
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
    print('The %s found is at x=%f, y=%f with a value of z=%f' % (max_min, x_best, y_best, z_best))
    print('Time elapsed', datetime.now() - startTime)
    print('Time per iteration', (datetime.now() - startTime)/iterations)
    return x_best, y_best, z_best
