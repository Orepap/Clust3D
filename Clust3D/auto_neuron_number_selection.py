import numpy as np
from Clust3D.training import train_Clust3D
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Clust3D.neuron_init import neurons_initialization


def _norm(a, b, ord=None, nan_mask=False):
    """
    Compute the distance between two arrays a and b.

    If nan_mask=True, restricts the computation to non-NaN overlapping
    elements (masked Frobenius norm). Returns np.inf if no valid overlap.
    If nan_mask=False, uses the standard numpy norm.
    """
    if nan_mask:
        mask = ~np.isnan(a) & ~np.isnan(b)
        if not np.any(mask):
            return np.inf
        return np.linalg.norm((a - b)[mask], ord=ord)
    else:
        return float(np.linalg.norm(a - b, ord=ord))


def get_number_of_neurons(nn, neuron_init, lr_0, MDC_data, neighbors,
                           correlation, data_min, data_max, depth,
                           rng, ord,
                           nan_mask=False):            # <-- NEW PARAMETER

    nn       = nn + 1
    sse_list = []

    for i in range(2, nn):

        epochs = i * 1000
        t1     = int(epochs / 2)
        t2     = int(epochs)

        neurons = neurons_initialization(
            neuron_init, correlation, MDC_data, i,
            data_min, data_max, depth, rng, ord, nan_mask  # <-- pass nan_mask
        )

        std = []
        for neuron in neurons:
            std.append(np.mean([
                _norm(neuron, n, ord=ord, nan_mask=nan_mask)
                for n in neurons
                if _norm(neuron, n, ord=ord, nan_mask=nan_mask) != 0
            ]))
        std_mean_all = np.mean(std)

        cl_labels, neurons_data, MDC_data, clusters_data, clusters = train_Clust3D(
            epochs, lr_0, t1, t2, neurons, i, MDC_data,
            neighbors, std_mean_all, correlation, ord, nan_mask  # <-- pass nan_mask
        )

        sse = 0
        for key, n in zip(clusters_data.keys(), range(len(neurons_data))):
            d      = np.array(clusters_data[key])
            errors = [
                _norm(matrix, neurons_data[n], ord=ord, nan_mask=nan_mask) ** 2
                for matrix in d
            ]
            sse += np.sum(errors)

        sse_list.append(sse)

    scaler   = MinMaxScaler(feature_range=(2, nn - 1))
    a_scaled = scaler.fit_transform(np.array(sse_list).reshape(-1, 1))
    a_scaled = np.array(a_scaled).reshape(len(a_scaled),)
    c        = abs(np.diff(a_scaled))

    for n, value in enumerate(c):
        if value <= 0.90:
            best = n + 2
            break

    return best
