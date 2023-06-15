"""
Authors: {
            Orestis D. Papagiannopoulos :   orepap@uoi.gr
            Vasileios Pezoulas          :   bpezoulas@gmail.com
            Costas Papaloukas           :   papalouk@uoi.gr
            Dimitrios I. Fotiadis       :   fotiadis@uoi.gr
         }


Institution                             :    University of Ioannina, Ioannina, Greece.
                                             Unit of Medical Technology and Intelligent Information Systems,
                                             Department of Materials Science and Engineering,
                                             University of Ioannina, Ioannina GR45110, Greece.

Contact                                 :    orepap@uoi.gr
GitHub                                  :    https://github.com/Orepap/TMDC

"""



import random
import numpy as np
import pandas as pd
import time
from neuron_init import neurons_initialization


def TMDC(data_file,
        correlation_file,
        n_neurons,
        distance="euclidean",
        imputation="zeros",
        max_n_neurons=8,
        dim_red="pca_auto",
        scaling="minmax",
        neighbors=True,
        epochs=5000,
        lr=0.3,
        neuron_init="points",
        t1=1,
        t2=1,
        verbose=2,
        random_state=random.randint(1, 10000),
        depth=10000):


    # Random state
    rng = np.random.RandomState(random_state)


    t0 = time.time()
    print()

    # Reads and converts the txt file into a Dataframe
    if data_file[-4:] == ".txt":
        try:
            df = pd.read_csv(data_file, delimiter="\t", index_col=0)
            print("Data file loaded")
        except FileNotFoundError:
            print()
            print("Please enter a valid data file")
            input("Press enter to close program")
            exit()


    elif data_file[-4:] == ".csv":
        try:
            df = pd.read_csv(data_file, index_col=0)
            print("Csv file loaded")
        except FileNotFoundError:
            print()
            print("Please enter a valid data file")
            input("Press enter to close program")
            exit()



    imputation = imputation
    from imputation import impute
    df = impute(df, imputation)



    df_transposed = df.transpose()


    # Reads the txt correlation file and prepares it as a Dataframe
    if correlation_file[-4:] == ".txt":
        try:
            df_cor = pd.read_csv(correlation_file, header=None, delimiter=" ")
            print("Correlation file loaded")
            print()
        except FileNotFoundError:
            print()
            print("Please enter a valid correlation file")
            input("Press enter to close program")
            exit()

    if correlation_file[-4:] == ".csv":
        try:
            df_cor = pd.read_csv(correlation_file, header=None, delimiter=",")
            print("Correlation file loaded")
            print()
        except FileNotFoundError:
            print()
            print("Please enter a valid correlation file")
            input("Press enter to close program")
            exit()

    correlation = []
    for i in df_cor.index:
        c = []
        for j in range(len(df_cor.columns)):
            c.append(df_cor.iloc[i][j])
        correlation.append(c)


    data = []
    for cor in correlation:
        d = []
        for sample in cor[1:]:
            d.append(np.array(df_transposed.loc[sample]))
        data.append(d)
    data = np.array(data)




    if distance == "euclidean":
        ord = None
    elif distance == "manhattan":
        ord = 1
    else:
        print("ERROR: Enter 'euclidean' or 'manhattan' for the 'distance' parameter value")
        exit()


    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

    scaling = scaling
    if scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "stadard":
        scaler = StandardScaler()
    else:
        print("ERROR: Enter 'minmax' or 'stadard' for the 'scaling' parameter value")
        exit()

    data = scaler.fit_transform(data)
    data = np.array(data).reshape((len(correlation), len(correlation[0]) - 1, data.shape[1]))


    if t1 == 1 and t2 == 1:
        t1 = int(epochs / 2)
        t2 = int(epochs)


    dim_red = dim_red
    from dim_red import apply_dim_red
    input_data, data_max, data_min = apply_dim_red(dim_red, data, correlation)




    from inputs import inputs
    MDC_data, neighbors, n_neurons, epochs, lr_0, t1, t2, depth, max_n_neurons = inputs(input_data, neighbors, n_neurons, epochs, lr, t1, t2, depth, max_n_neurons, dim_red, neuron_init, verbose)


    if n_neurons == -1:

        print()

        print("The neural network is being trained...")
        print(f"Finding the optimal no. of neurons based on the elbow rule...")
        print("This may take a few minutes")
        print()

        # Neural network training up to max_n_neurons number of neurons to select the best
        max_n_neurons = max_n_neurons

        from auto_neuron_number_selection import get_number_of_neurons
        best = get_number_of_neurons(max_n_neurons, neuron_init, lr_0, MDC_data, neighbors, correlation, data_min, data_max, depth, rng, ord)
        n_neurons = best

        print(f"Best no. of neurons: {n_neurons}")
        print(f"The neural network is being trained with {n_neurons} neurons...")
        print()

        neurons = neurons_initialization(neuron_init, correlation, MDC_data, n_neurons, data_min, data_max, depth, rng, ord)

        std = []
        for neuron in neurons:
            std.append(
                np.mean(np.array([np.linalg.norm(neuron - n, ord=ord) for n in neurons if np.linalg.norm(neuron - n, ord=ord) != 0])))
        std_mean_all = np.mean(std)

        from training import train_MDC
        epochs = n_neurons * 500
        cl_labels, neurons, MDC_data, clusters_data, clusters = train_MDC(epochs, lr_0, t1, t2, neurons, n_neurons, MDC_data, neighbors, std_mean_all, correlation, ord)


    else:

        print()
        print("The neural network is being trained...")
        print()

        neurons = neurons_initialization(neuron_init, correlation, MDC_data, n_neurons, data_min, data_max, depth, rng, ord)

        std = []
        for neuron in neurons:
            std.append(
                np.mean(np.array([np.linalg.norm(neuron - n, ord=ord) for n in neurons if np.linalg.norm(neuron - n, ord=ord) != 0])))
        std_mean_all = np.mean(std)

        from training import train_MDC
        cl_labels, neurons, MDC_data, clusters_data, clusters = train_MDC(epochs, lr_0, t1, t2, neurons, n_neurons, MDC_data, neighbors, std_mean_all, correlation, ord)


    if verbose == 0:
        print()
        print("### RESULTS ###")
        print()
        print(f"The neural network trained in {time.time() - t0} seconds")
        print()

    if verbose == 1:
        print()
        print("### RESULTS ###")
        print()
        print(cl_labels)
        print()
        print(f"The neural network trained in {time.time() - t0} seconds")
        print()

    elif verbose == 2:
        print()
        print("### RESULTS ###")
        print()
        for i in range(n_neurons):

            print(f"In cluster {str(i + 1)}: {list(clusters[str(i+1)])}")
            print()

        print()
        print(f"The neural network trained in {np.round(time.time() - t0, 0)} seconds")
        print()

    return clusters, neurons, cl_labels




