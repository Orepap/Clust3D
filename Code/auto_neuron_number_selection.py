import numpy as np
from TMDC.training import train_MDC
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from TMDC.neuron_init import neurons_initialization


def get_number_of_neurons(nn, neuron_init, lr_0, MDC_data, neighbors, correlation, data_min, data_max, depth, rng, ord):

    nn = nn + 1
    ord = ord

    sse_list = []
    for i in range(2, nn):

        epochs = i * 500
        t1 = int(epochs / 2)
        t2 = int(epochs)


        neurons = neurons_initialization(neuron_init, correlation, MDC_data, i, data_min, data_max, depth, rng, ord)


        # Standart deviation of neuron distances
        std = []
        for neuron in neurons:
            std.append(
                np.mean(np.array([np.linalg.norm(neuron - n, ord=ord) for n in neurons if np.linalg.norm(neuron - n, ord=ord) != 0])))
        std_mean_all = np.mean(std)


        cl_labels, neurons_data, MDC_data, clusters_data, clusters = train_MDC(epochs, lr_0, t1, t2, neurons, i, MDC_data, neighbors, std_mean_all, correlation, ord)

        sse = 0
        for key, n in zip(clusters_data.keys(), range(len(neurons_data))):

            d = np.array(clusters_data[key])

            errors_in_cluster = [((np.linalg.norm(matrix - neurons_data[n], ord=ord)) ** 2) for matrix in d]
            sum_errors_in_cluster = np.sum(errors_in_cluster)

            sse += sum_errors_in_cluster

        sse_list.append(sse)

    # SSE - no. of neurons plot
    plt.plot(range(2, nn), sse_list)
    plt.show()

    scaler = MinMaxScaler(feature_range=(2, nn-1))
    a_scaled = scaler.fit_transform(np.array(sse_list).reshape(-1, 1))
    a_scaled = np.array(a_scaled).reshape(len(a_scaled), )

    c = np.diff(a_scaled)
    c = abs(np.array(c))


    for n, value in enumerate(c):
        if value <= 0.90:
            best = n + 2
            break

    return best

