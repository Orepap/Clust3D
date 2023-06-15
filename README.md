# TMDC
Timeseries Multi-Dimensional Clustering tool.

A clustering tool to efficiently associate high-dimensional and time-dependent data. It utilizes competitive learning in order to create a self-organizing neural network to adjust neuron positions in a time-dependent, high dimensionality feature space and assign them as clustering centers.


# INSTALL

Run the following command in the terminal to install /
pip install git+https://github.com/Orepap/TMDC.git


# PREREQUISITES
```python
scikit-learn 1.0.2
numpy 1.21.6
pandas 1.4.0
matplotlib 3.5.1
```

# USAGE
```python
from TMDC.main import TMDC

data_file = "..." # path to the data file
correlation_file = "..." path to the correlation file

clusters, neurons, cl_labels = TMDC(data_file=data_file, correlation_file=correlation_file, n_neurons=-1)
```
