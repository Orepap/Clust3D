# Timeseries Multi-Dimensional Clustering tool

<p align="justify">
TMDC is a novel method designed specifically for clustering timeseries autoinflammatory gene expression data. TMDC enables the clustering of patients based on their gene expression profiles at different time points. By grouping patients according to the similarity of their gene profile changes over time, TMDC provides a robust foundation for exploratory analysis. Notably, to the best of our knowledge, there is currently no publicly available tool or clustering algorithm specifically tailored for the simultaneous clustering of SAID patients across multiple time points. TMDC fills this gap by employing a self-adjusting neural network approach to effectively cluster SAID patients with time-related gene expressions. The distinguishing feature of TMDC lies in its capability for efficient multi-timepoint and multi-dimensional clustering. TMDC’s neural network training design makes it also applicable for clustering complex data structures with temporal dimensions, other than gene expressions. </p>

 
• Orestis D. Papagiannopoulos  
• Vasileios C. Pezoulas  
• Costas Papaloukas  
• Dimitrios I. Fotiadis  

Unit of Medical Technology and Intelligent Information Systems  
Dept. of Materials Science and Engineering
University of Ioannina   
Ioannina, Greece

Contact: orepap@uoi.gr

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
correlation_file = "..." # path to the correlation file
n_neurons = # Specify the number of neurons

clusters, neurons, cl_labels = TMDC(data_file=data_file, correlation_file=correlation_file, n_neurons=n_neurons)
```
The code returs a dictionary with the cluster memberships (clusters),    
the clustering centers (neurons)  
and a list with the clustering labels (cl_labels).  

Folder "Examples" provides three usage examples based on timeseries gene expression data from systemic autoinflammatory diseases.


# INPUT FILES
<p align="justify">
TMDC requires two files as input. The first one is the data file which contains a table with the features and all the samples of the different time intervals and can be either a txt or a csv file. The second one, is an UTF-8 or ANSI format txt file, in which the correlation between the sample class labels, for which the clustering will take place, along with their corresponding samples in the different time intervals have to be specified. </p>

Specifically for Data Matrix Files from the Gene Expression Omnibus, the following steps are required in preparation of those two files.

For the data file, the user needs to:  
•	Download the Series Matrix File txt file of the desired GSE Series from the Gene Expression Omnibus  
•	Delete everything from within the txt file up until (and) the line ”!series_matrix_table_begin”  
•	Delete the very last line “!series_matrix_table_end”  
•	Save the file and exit  


For the correlation file, the user needs to create a txt file in UTF-8 or ANSI format, like the example below:

![εικόνα](https://github.com/Orepap/TMDC/assets/93657525/80b3de60-8e8e-481e-8466-0033ddc2d5b6)

In this particular example, P1 is a patient (arbitrary label) and GSM1, GSM2 and GSM3 correspond to that patients’s samples at (e.g.) three time intervals. The following have to be true for the correlation file:

1) One space per word  
2) No space at the end of each line  
3) No space at the end of the file  
4) No number as the first label letter

# PARAMETERS
"TMDC Parameters.md" details all the tunable parameters of TMDC.

# HOW IT WORKS
<p align="justify">
The Euclidean distance is first computed between an input sample and all the neurons. Then, the neuron that has the smallest distance to the sample is declared as the best matching unit (BMU) and its weights along with its nearest neighbor neurons (self-organizing) are re-adjusted to closer mimic the input sample. The novelty is the introduction of matrix norms as distance concepts. Conventional distance metrics like the Euclidean, are typically calculated between vectors. In TMDC, where the data points are matrices, the distance between two data points is defined as the mathematical norm of the matrix of their differences. As such, TMDC introduces the capability to train the neural network given the input samples and the neurons as matrices and not just as vectors, containing both the temporal and the spatial information. Thus, the clustering can be implemented directly on the patients, given the different timepoints altogether. </p>

The selected norm is the Frobenius norm:  
![εικόνα](https://github.com/Orepap/TMDC/assets/93657525/2de1dec0-3b0c-46e7-88fa-b8a4dc960f15)
 
# ADVANCED
The  user can copy and paste the .py files included in the "Code" folder in a directory and change/experiment with different parameters.  
  
# STANDALONE TRAINING
The user can feed their own preprocessed data file in order to be trained by TMDC. This is feasible by changing the "preprocess" parameter to False, while providing the preprocessed data file and the correlation file.
