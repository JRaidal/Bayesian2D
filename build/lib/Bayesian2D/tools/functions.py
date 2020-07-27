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
        Returns function value based on input x and y.

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
