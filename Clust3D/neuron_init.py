import numpy as np
import copy
import itertools


def neurons_initialization(neuron_init, correlation, MDC_data, n_neurons, data_min, data_max, depth, rng, ord):

    def masked_norm(a, b, ord=2):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if not np.any(mask):
            return np.inf
        return np.linalg.norm((a - b)[mask], ord=ord)

    n_phases = len(correlation[0]) - 1

    if neuron_init == "random":
        data_copy = np.array(copy.copy(MDC_data))
        neurons = []

        for _ in range(n_neurons):
            w = [
                rng.uniform(low=data_min, high=data_max, size=(data_copy.shape[2],))
                for _ in range(n_phases)
            ]
            neurons.append(w)

        return np.array(neurons)

    elif neuron_init == "points":
        if depth == "auto":
            all_combos = list(itertools.combinations(range(len(MDC_data)), n_neurons))
            dict_neuron_choose = {str(k): list(c) for k, c in enumerate(all_combos)}
        else:
            dict_neuron_choose = {}
            for k in range(depth):
                selected = rng.choice(len(MDC_data), size=n_neurons, replace=False)
                dict_neuron_choose[str(k)] = list(selected)

        all_ress = []

        for selected_indices in dict_neuron_choose.values():
            data_copy = np.array(copy.copy(MDC_data))
            neurons = np.array([data_copy[idx] for idx in selected_indices])

            for nnn in neurons:
                di = [masked_norm(nnn, nn, ord=ord) for nn in neurons if not np.array_equal(nnn, nn)]
                break  # Score using just the first neuron
            all_ress.append(np.mean(di))

        index_max = np.argmax(all_ress)
        selected_indices = dict_neuron_choose[str(index_max)]
        sorted_indices = sorted(selected_indices, reverse=True)

        data_copy = np.array(copy.copy(MDC_data))
        neurons = []

        for idx in sorted_indices:
            w = data_copy[idx]
            data_copy = np.delete(data_copy, idx, axis=0)
            neurons.append(w)

        return np.array(neurons)

    else:
        print("ERROR: Enter 'random' or 'points' for the 'neuron_init' parameter")
        exit()
