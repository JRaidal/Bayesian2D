# Surrogate model
def surrogate(model, XY): 
    '''
    Predicts the mean and standard deviation of points using Gaussian processes

    Parameters
    ----------
    model : sklearn.gaussian_process
        Some Gaussian process model.
    XY : numpy array
        Array of x and y coordinates.

    Returns
    -------
    array, array
        Returns mean and standard deviation arrays for evaluated points.

    '''
    return model.predict(XY, return_std=True)
