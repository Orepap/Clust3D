import numpy as np
import copy
from random import shuffle



def train_Clust3D(epochs, lr_0, t1, t2, neurons, n_neurons, MDC_data, neighbors, std_mean_all, correlation, ord):

    t1 = int(epochs / 2)
    t2 = int(epochs)

    for j in range(epochs):

        lr = lr_0 * np.exp(-j / t1)

        MDC_data_copy = copy.copy(MDC_data)
        MDC_data_copy_list = list(MDC_data_copy)
        shuffle(MDC_data_copy_list)
        MDC_data_copy_nparray = np.array(MDC_data_copy_list)


        for matrix in MDC_data_copy_nparray:

            # Distances between data point and neurons
            dists = [np.linalg.norm(matrix - neurons[i], ord=ord) for i in range(n_neurons)]


            # Finding the best matching unit (BMU)
            index = np.argmin(dists)
            q = copy.copy(neurons[index])

            # No use of neighbors
            if not neighbors:
                # BMU weights update
                neurons[index] = q + lr * (matrix - q)


            # Use of neighbors (SOM)
            if neighbors:
                # For every neuron
                for nn in range(n_neurons):

                    m = copy.copy(neurons[nn])

                    h = np.exp((-(np.linalg.norm(q - m, ord=ord) ** 2)) / (2 * (std_mean_all * np.exp(-j * np.log(std_mean_all) / t2)) ** 2))

                    neurons[nn] = m + lr * h * (matrix - q)



    clusters = dict([(str(i), []) for i in range(1, n_neurons + 1)])

    clusters_data = dict([(str(i), []) for i in range(1, n_neurons + 1)])


    cl_labels = []
    d = []
    for num, matrix in enumerate(MDC_data):

        ds = [np.linalg.norm(matrix - neurons[i], ord=ord) for i in range(n_neurons)]

        # Find neuron with the smallest distance for the data point to be assigned to
        k = np.argmin(ds)
        kk = k + 1

        clusters[str(kk)].append(correlation[num][0])

        clusters_data[str(kk)].append(np.array(matrix))

        cl_labels.append(kk)
        d.append(matrix)

    return cl_labels, neurons, MDC_data, clusters_data, clusters

