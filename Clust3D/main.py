"""
Authors: {
            Orestis D. Papagiannopoulos :   orepap_a@hotmail.com
            Vasileios Pezoulas          :   bpezoulas@gmail.com
            Costas Papaloukas           :   papalouk@uoi.gr
            Dimitrios I. Fotiadis       :   fotiadis@uoi.gr
         }


Institution                             :    University of Ioannina, Ioannina, Greece.
                                             Unit of Medical Technology and Intelligent Information Systems,
                                             Department of Materials Science and Engineering,
                                             University of Ioannina, Ioannina GR45110, Greece.

Contact                                 :    orepap@uoi.gr
GitHub                                  :    https://github.com/Orepap/Clust3D

"""

import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Clust3D.imputation import impute
from Clust3D.dim_red import apply_dim_red
from Clust3D.inputs import inputs
from Clust3D.auto_neuron_number_selection import get_number_of_neurons
from Clust3D.neuron_init import neurons_initialization
from Clust3D.training import train_Clust3D


def masked_norm(a, b, ord=2):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if not np.any(mask):
        return np.inf
    return np.linalg.norm((a - b)[mask], ord=ord)


def Clust3D(data_file,
            correlation_file,
            n_neurons,
            distance="euclidean",
            imputation="zeros",
            max_n_neurons=8,
            dim_red="pca_auto",
            scaling="minmax",
            scaling_per_dimension=False,
            neighbors=True,
            epochs=5000,
            lr=0.3,
            neuron_init="points",
            t1=1,
            t2=1,
            random_state=random.randint(1, 10000),
            depth=100000):

    rng = np.random.RandomState(random_state)
    t0 = time.time()
    print()

    # Load data file
    try:
        if data_file.endswith(".txt"):
            df = pd.read_csv(data_file, delimiter="\t", index_col=0)
            print("Data file loaded")
        elif data_file.endswith(".csv"):
            df = pd.read_csv(data_file, index_col=0)
            print("Csv file loaded")
        else:
            raise ValueError("Unsupported file extension for data file.")
    except FileNotFoundError:
        print("\nEnter a valid data file")
        input("Press enter to close program")
        exit()

    df = impute(df, imputation)
    df_transposed = df.transpose()

    # Load correlation file
    try:
        if correlation_file.endswith(".txt"):
            df_cor = pd.read_csv(correlation_file, header=None, delimiter=" ")
        elif correlation_file.endswith(".csv"):
            df_cor = pd.read_csv(correlation_file, header=None, delimiter=",")
        else:
            raise ValueError("Unsupported file extension for correlation file.")
        print("Correlation file loaded\n")
    except FileNotFoundError:
        print("\nEnter a valid correlation file")
        input("Press enter to close program")
        exit()

    # Convert correlation dataframe to list of lists
    correlation = df_cor.values.tolist()

    # Build raw input data tensor
    data = [[np.array(df_transposed.loc[sample]) for sample in cor[1:]] for cor in correlation]
    data = np.array(data)

    # Set distance metric
    if distance == "euclidean":
        ord = None
    elif distance == "manhattan":
        ord = 1
    else:
        print("ERROR: Enter 'euclidean' or 'manhattan' for the 'distance' parameter value")
        exit()

    # Handle default t1 and t2 if not set
    if t1 == 1 and t2 == 1:
        t1 = int(epochs / 2)
        t2 = int(epochs)

    # Scale input data
    if scaling == "minmax":
        scaler_class = MinMaxScaler
    elif scaling == "standard":
        scaler_class = StandardScaler
    elif scaling == "none":
        scaler_class = None
    else:
        print("ERROR: Enter 'minmax', 'standard' or 'none' for the 'scaling' parameter value")
        exit()

    if scaling != "none":
        data = np.array(data, dtype=float)

        if np.isnan(data).any():
            print("NaN values detected in the data. Masked Frobenius distance will be used.")

        if scaling_per_dimension:
            for i in range(data.shape[1]):
                scaler = scaler_class()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data[:, i, :] = scaler.fit_transform(data[:, i, :])
        else:
            original_shape = data.shape
            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
            scaler = scaler_class()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = scaler.fit_transform(data)
            data = data.reshape(original_shape)

    # Dimensionality reduction
    if dim_red not in ["pca_elbow", "none", "pca_auto", "t-sne", "ica"]:
        print("ERROR: Refer to the parameters docx for the available dimensionality reduction options")
        exit()

    input_data, data_max, data_min = apply_dim_red(dim_red, data, correlation)

    # Validate and format all inputs
    MDC_data, neighbors, n_neurons, epochs, lr_0, t1, t2, depth, max_n_neurons = inputs(
        input_data, neighbors, n_neurons, epochs, lr, t1, t2, depth, max_n_neurons, dim_red, neuron_init
    )

    # If neuron count unknown, find optimal number with elbow method
    if n_neurons == -1:
        print("The neural network is being trained...")
        print("Finding the optimal no. of neurons based on the elbow rule...")
        print("This may take a few minutes\n")

        best = get_number_of_neurons(
            max_n_neurons, neuron_init, lr_0, MDC_data, neighbors,
            correlation, data_min, data_max, depth, rng, ord, random_state
        )
        n_neurons = best

        print(f"Best no. of neurons: {n_neurons}")
        print(f"The neural network is being trained with {n_neurons} neurons...\n")

        neurons = neurons_initialization(
            neuron_init, correlation, MDC_data, n_neurons,
            data_min, data_max, depth, rng, ord
        )

        std = [
            np.mean([
                masked_norm(neuron, n, ord=ord)
                for n in neurons
                if masked_norm(neuron, n, ord=ord) not in (0, np.inf)
            ])
            for neuron in neurons
        ]
        std_mean_all = np.mean(std)

        epochs = n_neurons * 1000
        cl_labels, neurons, MDC_data, clusters_data, clusters = train_Clust3D(
            epochs, lr_0, t1, t2, neurons, n_neurons, MDC_data,
            neighbors, std_mean_all, correlation, ord, random_state
        )

    else:
        print("The neural network is being trained...\n")

        neurons = neurons_initialization(
            neuron_init, correlation, MDC_data, n_neurons,
            data_min, data_max, depth, rng, ord
        )

        std = [
            np.mean([
                masked_norm(neuron, n, ord=ord)
                for n in neurons
                if masked_norm(neuron, n, ord=ord) not in (0, np.inf)
            ])
            for neuron in neurons
        ]
        std_mean_all = np.mean(std)

        cl_labels, neurons, MDC_data, clusters_data, clusters = train_Clust3D(
            epochs, lr_0, t1, t2, neurons, n_neurons, MDC_data,
            neighbors, std_mean_all, correlation, ord, random_state
        )

    print(f"The neural network trained in {np.round(time.time() - t0, 0)} seconds\n")
    return clusters, neurons, cl_labels
