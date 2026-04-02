def inputs(input_data, neighbors, n_neurons, epochs, lr, t1, t2, depth, max_n_neurons, dim_red, neuron_init):
    """
    Validate and process input parameters for the Clust3D pipeline.

    Returns:
        tuple: Processed (MDC_data, neighbors, n_neurons, epochs, lr_0, t1, t2, depth, max_n_neurons)
    """

    MDC_data = input_data

    # --- Validate neighbors ---
    if not isinstance(neighbors, bool):
        print("ERROR: Enter a boolean value for the 'neighbors' parameter")
        exit()

    # --- Validate number of neurons ---
    if not isinstance(n_neurons, int):
        print("ERROR: Enter an integer value for the 'n_neurons' parameter")
        exit()
    elif n_neurons <= 1 and n_neurons != -1:
        print("ERROR: Enter a value of at least '2' for 'n_neurons' or '-1' for auto-selection")
        exit()

    # --- Validate max number of neurons if auto-selection is used ---
    if n_neurons == -1:
        if not isinstance(max_n_neurons, int):
            print("ERROR: Enter an integer value for the 'max_n_neurons' parameter")
            exit()
        elif max_n_neurons <= 2:
            print("ERROR: Enter a value of at least '3' for 'max_n_neurons'")
            exit()
        elif max_n_neurons > 10:
            print("WARNING: High values for 'max_n_neurons' will increase run time")
        elif max_n_neurons > len(MDC_data):
            print("ERROR: 'max_n_neurons' cannot exceed number of samples in dataset")
            exit()

    # --- Validate neuron initialization method ---
    if neuron_init not in ["random", "points"]:
        print("ERROR: Enter 'random' or 'points' for the 'neuron_init' parameter")
        exit()

    # --- Validate epochs ---
    if not isinstance(epochs, int):
        print("ERROR: Enter a positive integer for the 'epochs' parameter")
        exit()
    elif epochs <= 0:
        print("ERROR: 'epochs' must be at least 1")
        exit()
    elif epochs < 1000:
        print("WARNING: At least 500 * n_neurons for 'epochs' is advised")

    # --- Validate learning rate ---
    lr_0 = lr
    if lr_0 <= 0:
        print("ERROR: Enter a positive number for the 'lr' parameter")
        exit()

    # --- Validate t1 and t2 ---
    if t1 <= 0:
        print("ERROR: Enter a positive integer for 't1'")
        exit()
    if t2 <= 0:
        print("ERROR: Enter a positive integer for 't2'")
        exit()

    # --- Validate depth ---
    if depth != "auto":
        if depth <= 0:
            print("ERROR: Enter a positive integer or 'auto' for the 'depth' parameter")
            exit()
        elif depth > 500000:
            print("WARNING: Depth > 500000 may cause long runtime")

    return MDC_data, neighbors, n_neurons, epochs, lr_0, t1, t2, depth, max_n_neurons
