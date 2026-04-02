import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Clust3D.training import train_Clust3D
from Clust3D.neuron_init import neurons_initialization


def masked_norm(a, b, ord=2):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if not np.any(mask):
        return np.inf
    return np.linalg.norm((a - b)[mask], ord=ord)


def get_number_of_neurons(nn, neuron_init, lr_0, MDC_data, neighbors, correlation, data_min, data_max, depth, rng, ord, random_state):

    nn = nn + 1
    sse_list = []

    for i in range(2, nn):
        epochs = i * 1000
        t1 = int(epochs / 2)
        t2 = int(epochs)

        neurons = neurons_initialization(
            neuron_init, correlation, MDC_data, i, data_min, data_max, depth, rng, ord
        )

        std = []
        for neuron in neurons:
            dists = [
                masked_norm(neuron, n, ord=ord)
                for n in neurons
                if masked_norm(neuron, n, ord=ord) not in (0, np.inf)
            ]
            std.append(np.mean(dists))
        std_mean_all = np.mean(std)

        cl_labels, neurons_data, MDC_data, clusters_data, clusters = train_Clust3D(
            epochs, lr_0, t1, t2, neurons, i, MDC_data, neighbors, std_mean_all, correlation, ord, random_state
        )

        sse = 0
        for key, n in zip(clusters_data.keys(), range(len(neurons_data))):
            d = np.array(clusters_data[key])
            cluster_errors = [
                masked_norm(matrix, neurons_data[n], ord=ord) ** 2
                for matrix in d
                if not np.isinf(masked_norm(matrix, neurons_data[n], ord=ord))
            ]
            sse += np.sum(cluster_errors)

        sse_list.append(sse)

    scaler = MinMaxScaler(feature_range=(2, nn - 1))
    a_scaled = scaler.fit_transform(np.array(sse_list).reshape(-1, 1)).flatten()
    c = np.abs(np.diff(a_scaled))

    for n, value in enumerate(c):
        if value <= 0.90:
            best = n + 2
            break

    '''
    x_vals = list(range(2, 2 + len(sse_list)))

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, sse_list, marker='o', label='SSE')
    plt.axvline(x=best, color='red', linestyle='--', label=f'Elbow @ {best}')
    plt.scatter([best], [sse_list[best - 2]], color='red', s=80, zorder=5)

    plt.title('SSE vs Number of Clusters/Neurons')
    plt.xlabel('Number of Clusters / Neurons (nn)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''

    return best
