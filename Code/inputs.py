def inputs(input_data, neighbors, n_neurons, epochs, lr, t1, t2, depth, max_n_neurons, dim_red, neuron_init):


    # Input data for MDC
    MDC_data = input_data


    # Use of neighbors for the SOM approach
    neighbors = neighbors
    if type(neighbors) != bool:
        print("ERROR: Enter a boolean value for the 'neighbors' parameter")
        exit()


    # Number of neurons
    n_neurons = n_neurons
    if type(n_neurons) != int:
        print("ERROR: Enter an integer value for the 'n_neurons' parameter")
        exit()
    elif n_neurons <= 1 and n_neurons != -1:
        print("ERROR: Enter a value of at least '2' for the 'n_neuron' parameter or '-1' for the automatic selection")
        exit()


    if n_neurons == -1:

        # Number of max neurons
        max_n_neurons = max_n_neurons
        if type(max_n_neurons) != int:
            print("ERROR: Enter an integer value for the 'max_n_neurons' parameter")
            exit()
        elif max_n_neurons <= 2:
            print("ERROR: Enter a value of at least '3' for the 'n_neurmax_n_neurons' parameter")
            exit()
        elif max_n_neurons > 10:
            print("WARNING: Higher values of the 'n_neurmax_n_neurons' parameter will result in higher running times")
            print()
        elif max_n_neurons > len(MDC_data):
            print("ERROR: The value of 'max_n_neurons' cannot be higher than the no. of samples in the dataset")
            exit()


    dim_red = dim_red
    if dim_red != "pca_elbow" and dim_red != "none" and dim_red != "pca_auto" and dim_red != "t-sne" and dim_red != "ica":
        print("ERROR: Refer to the parameters docx for the available dimensionality reduction options")
        exit()


    neuron_init = neuron_init
    if neuron_init != "random" and neuron_init != "points":
        print("ERROR: Enter 'random' or 'points' for the 'neuron_init' parameter value")
        exit()


    # Number of iterations
    epochs = epochs
    if type(epochs) != int:
        print("ERROR: Enter a positive integer value for the 'epochs' parameter")
        exit()
    elif epochs <= 0:
        print("ERROR: The value for the 'epochs' parameter cannot be less than 1")
        exit()
    elif epochs < 1000:
        print("WARNING: A value of at least 500 * n_neurons for the 'epochs' parameter is adviced")
        print()


    # Initial learning rate
    lr_0 = lr
    if lr_0 <= 0:
        print("ERROR: Enter a positive integer value for the 'lr' parameter")
        exit()


    # Constants for the exponential decrease of the learning rate and the neighborhood function
    t1 = t1
    if t1 <= 0:
        print("ERROR: Enter a positive integer value for the 't1' parameter")
        exit()

    t2 = t2
    if t2 <= 0:
        print("ERROR: Enter a positive integer value for the 't2' parameter")
        exit()


    # Number of point combinations for neuron initialization
    depth = depth
    if depth == "auto":
        pass
    elif depth <= 0:
        print("ERROR: Enter a positive integer value or 'auto' for the 'depth' parameter")
        exit()
    elif depth > 500000:
        print("WARNING: A value above 500000 is not recommended for the 'depth' parameter due to very high running times")
        print()



    return MDC_data, neighbors, n_neurons, epochs, lr_0, t1, t2, depth, max_n_neurons
