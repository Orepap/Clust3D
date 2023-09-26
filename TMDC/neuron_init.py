import numpy as np
import copy
import itertools


def neurons_initialization(neuron_init, correlation, MDC_data, n_neurons, data_min, data_max, depth, rng, ord):


    n_phases = len(correlation[0]) - 1

    if neuron_init == "random":

        data_copy = np.array(copy.copy(MDC_data))
        neurons = []

        for i in range(n_neurons):

            w = []
            for j in range(n_phases):

                ww = rng.uniform(low=data_min, high=data_max, size=(np.array(data_copy).shape[2],))

                w.append(ww)

            neurons.append(w)

        neurons = np.array(neurons)

        return neurons



    elif neuron_init == "points":

        if depth == "auto":

            # Number of different combinations
            # print(sum(1 for _ in itertools.combinations([g for g in range(len(MDC_data))], n_neurons)))

            dict_neuron_choose = {str(k): [] for k in range(sum(1 for _ in itertools.combinations([g for g in range(len(MDC_data))], n_neurons)))}
            all_ress = []

            for n, combo in enumerate(itertools.combinations([g for g in range(len(MDC_data))], n_neurons)):
                dict_neuron_choose[str(n)] = list(combo)

            for value in dict_neuron_choose.values():

                data_copy = np.array(copy.copy(MDC_data))
                neurons = [data_copy[r, :, :] for r in value]
                neurons = np.array(neurons)

                for nnn in neurons:
                    di = [np.linalg.norm(nnn - nn, ord=ord) for nn in neurons if np.linalg.norm(nnn - nn, ord=ord) != 0]
                    break

                ress = np.mean(di)
                all_ress.append(ress)

            index_max = np.argmax(all_ress)
            sorted_index_list = -np.sort(-np.array(dict_neuron_choose[str(index_max)]))

            neurons = []
            data_copy = np.array(copy.copy(MDC_data))
            for ind in sorted_index_list:
                w = data_copy[ind, :, :]
                data_copy = np.delete(data_copy, ind, axis=0)

                neurons.append(w)

            neurons = np.array(neurons)

            return neurons



        else:

            choose_times = depth

            dict_neuron_choose = {str(k): [] for k in range(choose_times)}
            all_ress = []

            for dd in range(choose_times):

                data_copy = np.array(copy.copy(MDC_data))

                fr = [g for g in range(len(MDC_data))]

                neurons = []

                for i in range(n_neurons):

                    r = rng.choice(fr)
                    fr.remove(r)


                    dict_neuron_choose[str(dd)].append(r)

                    w = data_copy[r, :, :]


                    neurons.append(w)

                neurons = np.array(neurons)

                for nnn in neurons:

                    di = [np.linalg.norm(nnn - nn, ord=ord) for nn in neurons if np.linalg.norm(nnn - nn, ord=ord) != 0]
                    break

                ress = np.mean(di)
                all_ress.append(ress)

            index_max = np.argmax(all_ress)

            sorted_index_list = -np.sort(-np.array(dict_neuron_choose[str(index_max)]))


            neurons = []

            data_copy = np.array(copy.copy(MDC_data))


            for ind in sorted_index_list:

                w = data_copy[ind, :, :]
                data_copy = np.delete(data_copy, ind, axis=0)

                neurons.append(w)

            neurons = np.array(neurons)

            return neurons

    else:
        print("Enter 'random' or 'points' as the neuron_init parameter value")
        exit()


