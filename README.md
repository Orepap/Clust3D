# TMDC
Timeseries Multi-Dimensional Clustering tool.

A clustering tool to efficiently associate high-dimensional and time-dependent data. It utilizes competitive learning in order to create a self-organizing neural network to adjust neuron positions in a time-dependent, high dimensionality feature space and assign them as clustering centers.


# INSTALL

Run the following command in the terminal  
_pip install git+https://github.com/Orepap/TMDC.git_


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

# INPUT FILES
TMDC requires two files as input. The first one is the data file which contains a table with the features and all the samples of the different time intervals and can be either a txt or a csv file. 

The second one, is an UTF-8 or ANSI format txt file, in which the correlation between the sample class labels, for which the clustering will take place, along with their corresponding samples in the different time intervals have to be specified.

Specifically for Data Matrix Files from the Gene Expression Omnibus [], the following steps are required in preparation of those two files.

For the data file, the user needs to:
•	Download the Series Matrix File txt file of the desired GSE Series from the Gene Expression Omnibus.
•	Delete everything from within the txt file up until (and) the line ”!series_matrix_table_begin”.
•	Delete the very last line “!series_matrix_table_end”.
•	Save the file and exit


For the correlation file, the user needs to create a txt file in UTF-8 or ANSI format, like the example below.
 
 ![εικόνα](https://github.com/Orepap/TMDC/assets/93657525/176e62ee-9449-4b0f-86cb-eec64ccf15ca)

In this particular example, KD1 is a sample class label and GSM1, GSM2 and GSM3 correspond to that label’s samples at (e.g.) three time intervals. The naming of the sample class labels is arbitrary.
The following have to be true for the correlation file:
```python
1) One space per word
2) No space at the end of each line
3) No space at the end of the file
4) No number as the first label letter
```

# HOW IT WORKS
The Euclidean distance is first computed between an input sample and all the neurons. Then, the neuron that has the smallest distance to the sample is declared as the best matching unit (BMU) and its weights along with its nearest neighbor neurons (self-organizing) are re-adjusted to closer mimic the input sample. The novelty is the introduction of matrix norms as distance concepts. Conventional distance metrics like the Euclidean, are typically calculated between vectors. In TMDC, where the data points are matrices, the distance between two data points is defined as the mathematical norm of the matrix of their differences. As such, TMDC introduces the capability to train the neural network given the input samples and the neurons as matrices and not just as vectors, containing both the temporal and the spatial information. Thus, the clustering can be implemented directly on the patients, given the different timepoints altogether.

The update function for the neurons is defined as:
    W_j (i+1)= W_j (i)+U(W_j,W_q,i)  y(i)  [X-W_q ],

where W_j (i+1) is the matrix of neuron with index j at time (i+1), with i being the current iteration, X is the matrix of the input sample, W_q is the BMU, and y is the learning rate, which decreases exponentially:
              y= y_o  exp⁡〖((-i)/t_1 ),〗	(2)
              
where y_o is the initial learning rate, t_1 is a user defined constant which controls the exponential decrease of the learning rate, and U(W_j,W_q,i) is the neighborhood function, which dictates the cooperation between neurons. It decreases exponentially and includes a reducing Gaussian distance function [7]:

         U = exp((-〖〖 d〗_jq〗^2)/(2 〖(σ_(0 ) exp((-i log⁡(σ_0 ))/t_2 ))〗^2 )),	(3)
where σ_0 is the standard deviation of the initial Euclidean distances of the randomly initiated neurons, t_2  is a user defined constant which controls the exponential decrease of the neighborhood function and lastly, d_jq is the Euclidean distance between a neighbor neuron and the BMU, which is calculated using the Frobenius norm of the neuron matrices difference:
        ‖W_j-W_q ‖_F= ‖D‖_F=√(∑_(k=1)^m▒∑_(l=1)^n▒|D_kl |^2 ) 	(4)


