import numpy as np
import copy
from random import shuffle
from tqdm import trange


def train_Clust3D(epochs, lr_0, t1, t2, neurons, n_neurons, MDC_data, neighbors, std_mean_all, correlation, ord, random_state):

    def masked_norm(a, b, ord=2):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if not np.any(mask):
            return np.inf
        return np.linalg.norm((a - b)[mask], ord=ord)

    qe_history = []

    with trange(epochs, desc="Training Clust3D", ncols=100, colour="green") as pbar:
        for j in pbar:
            lr = lr_0 * np.exp(-j / t1)

            MDC_data_copy = copy.copy(MDC_data)
            shuffle(MDC_data_copy)
            MDC_data_copy = np.array(MDC_data_copy)

            epoch_qe = []

            for matrix in MDC_data_copy:

                dists = [masked_norm(matrix, neurons[i], ord=ord) for i in range(n_neurons)]

                index = np.argmin(dists)
                bmu = neurons[index].copy()
                qe = dists[index]
                epoch_qe.append(qe)

                # Skip update if no valid overlap
                if np.isinf(qe):
                    continue

                if not neighbors:
                    neurons[index] = bmu + lr * np.nan_to_num(matrix - bmu)
                else:
                    for nn in range(n_neurons):
                        m = neurons[nn].copy()
                        radius = std_mean_all * np.exp(-j * np.log(std_mean_all) / t2)
                        dist = masked_norm(bmu, m, ord=ord)

                        if np.isinf(dist):
                            continue

                        h = np.exp(-dist ** 2 / (2 * radius ** 2))
                        update = lr * h * np.nan_to_num(matrix - bmu)
                        neurons[nn] = m + update

            avg_qe = np.mean(epoch_qe)
            qe_history.append(avg_qe)

            cluster_labels = []
            for matrix in MDC_data_copy:
                dists = [masked_norm(matrix, neuron, ord=ord) for neuron in neurons]
                cluster_labels.append(np.argmin(dists))

            pbar.set_description(f"Epoch {j + 1}/{epochs} | ")

    # Final cluster assignment
    clusters = {str(i): [] for i in range(1, n_neurons + 1)}
    clusters_data = {str(i): [] for i in range(1, n_neurons + 1)}
    cl_labels = []

    for num, matrix in enumerate(MDC_data):
        dists = [masked_norm(matrix, neurons[i], ord=ord) for i in range(n_neurons)]
        k = np.argmin(dists)
        kk = k + 1
        clusters[str(kk)].append(correlation[num][0])
        clusters_data[str(kk)].append(np.array(matrix))
        cl_labels.append(kk)

    return cl_labels, neurons, MDC_data, clusters_data, clusters
