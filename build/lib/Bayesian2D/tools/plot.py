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

