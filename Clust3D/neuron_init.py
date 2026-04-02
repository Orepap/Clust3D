import numpy as np
import copy
import itertools


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


def neurons_initialization(neuron_init, correlation, MDC_data, n_neurons,
                            data_min, data_max, depth, rng, ord,
                            nan_mask=False):

    n_phases = len(correlation[0]) - 1

    if neuron_init == "random":

        data_copy = np.array(copy.copy(MDC_data))
        neurons = []

        for i in range(n_neurons):
            w = []
            for j in range(n_phases):
                ww = rng.uniform(low=data_min, high=data_max,
                                 size=(np.array(data_copy).shape[2],))
                w.append(ww)
            neurons.append(w)

        return np.array(neurons)

    elif neuron_init == "points":

        if depth == "auto":

            dict_neuron_choose = {
                str(k): list(combo)
                for k, combo in enumerate(
                    itertools.combinations(range(len(MDC_data)), n_neurons)
                )
            }

            all_ress = []
            for value in dict_neuron_choose.values():
                data_copy = np.array(copy.copy(MDC_data))
                neurons   = np.array([data_copy[r] for r in value])

                di = [
                    _norm(neurons[0], nn, ord=ord, nan_mask=nan_mask)
                    for nn in neurons
                    if not np.array_equal(neurons[0], nn)
                ]
                all_ress.append(np.mean(di))

        else:

            choose_times       = depth
            dict_neuron_choose = {str(k): [] for k in range(choose_times)}
            all_ress           = []

            for dd in range(choose_times):
                data_copy = np.array(copy.copy(MDC_data))
                fr        = list(range(len(MDC_data)))
                neurons   = []

                for i in range(n_neurons):
                    r = rng.choice(fr)
                    fr.remove(r)
                    dict_neuron_choose[str(dd)].append(r)
                    neurons.append(data_copy[r])

                neurons = np.array(neurons)
                di = [
                    _norm(neurons[0], nn, ord=ord, nan_mask=nan_mask)
                    for nn in neurons
                    if not np.array_equal(neurons[0], nn)
                ]
                all_ress.append(np.mean(di))

        index_max         = np.argmax(all_ress)
        sorted_index_list = -np.sort(-np.array(dict_neuron_choose[str(index_max)]))

        neurons   = []
        data_copy = np.array(copy.copy(MDC_data))
        for ind in sorted_index_list:
            w         = data_copy[ind, :, :]
            data_copy = np.delete(data_copy, ind, axis=0)
            neurons.append(w)

        return np.array(neurons)

    else:
        print("Enter 'random' or 'points' as the neuron_init parameter value")
        exit()
