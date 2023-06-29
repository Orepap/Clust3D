from TMDC.main import TMDC

# Specify your paths
data_file = "C:\\Users\\Medlab\\Desktop\\MDC Github\\Code\\example 1\\data file.txt"
correlation_file = "C:\\Users\\Medlab\\Desktop\\MDC Github\\Code\\example 1\\correlation file.txt"

clusters, neurons, cl_labels = TMDC(data_file=data_file, correlation_file=correlation_file, n_neurons=-1)
print(clusters)
print()
print(neurons)
print()
print(cl_labels)