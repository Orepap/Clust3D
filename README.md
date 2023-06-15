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
The software requires two files as input. The first one is the data file which contains a table with the features and all the samples of the different time intervals and can be either a txt or a csv file. 

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
1) One space per word, 2) No space at the end of each line, 3) No space at the end of the file, 4) No number as the first label letter.
