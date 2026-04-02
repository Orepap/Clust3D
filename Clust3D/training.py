import numpy as np
import copy
from random import shuffle


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


def train_Clust3D(epochs, lr_0, t1, t2, neurons, n_neurons, MDC_data,
                  neighbors, std_mean_all, correlation, ord,
                  nan_mask=False):

    for j in range(epochs):

        lr = lr_0 * np.exp(-j / t1)

        MDC_data_copy = copy.copy(MDC_data)
        MDC_data_copy_list = list(MDC_data_copy)
        shuffle(MDC_data_copy_list)
        MDC_data_copy_nparray = np.array(MDC_data_copy_list)

        for matrix in MDC_data_copy_nparray:

            dists = [_norm(matrix, neurons[i], ord=ord, nan_mask=nan_mask)
                     for i in range(n_neurons)]

            index = np.argmin(dists)
            q = copy.copy(neurons[index])

            if not neighbors:
                neurons[index] = q + lr * np.nan_to_num(matrix - q)

            if neighbors:
                for nn in range(n_neurons):
                    m = copy.copy(neurons[nn])
                    dist_qm = _norm(q, m, ord=ord, nan_mask=nan_mask)

                    if np.isinf(dist_qm):
                        continue

                    h = np.exp(
                        (-(dist_qm ** 2)) /
                        (2 * (std_mean_all * np.exp(-j * np.log(std_mean_all) / t2)) ** 2)
                    )
                    neurons[nn] = m + lr * h * np.nan_to_num(matrix - q)

    # Final cluster assignment
    clusters      = {str(i): [] for i in range(1, n_neurons + 1)}
    clusters_data = {str(i): [] for i in range(1, n_neurons + 1)}
    cl_labels = []
    d = []

    for num, matrix in enumerate(MDC_data):
        ds = [_norm(matrix, neurons[i], ord=ord, nan_mask=nan_mask)
              for i in range(n_neurons)]
        k  = np.argmin(ds)
        kk = k + 1
        clusters[str(kk)].append(correlation[num][0])
        clusters_data[str(kk)].append(np.array(matrix))
        cl_labels.append(kk)
        d.append(matrix)

    return cl_labels, neurons, MDC_data, clusters_data, clusters
