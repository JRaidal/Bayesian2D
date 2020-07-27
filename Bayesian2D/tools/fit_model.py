def fit_model(model, data_input, data_output):
    '''
    Fits new data to the model

    Parameters
    ----------
    model : sklearn.gaussian_process
        Some Gaussian process model.
    data_input : numpy array
        x and y coordinates to be fitted.
    data_output : numpy array
        Corresponding function values to the x and y coordinates.

    Returns
    -------
    None.

    '''
    model.fit(data_input, data_output)

