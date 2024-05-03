# Clust3D - 3D Clustering tool

<p align="justify">
Clust3D is a clustering tool designed for clustering 3D data, such as timeseries, using a self-adjusting neural network.
Clust3D provides the capability to directly cluster 3D data, exploiting the entire data structure, without the need for flattening or decomposition of one of the dimensions. </p>

 
Authors:  

• *Orestis D. Papagiannopoulos*  
• *Vasileios C. Pezoulas*  
• *Costas Papaloukas*  
• *Dimitrios I. Fotiadis*  

Unit of Medical Technology and Intelligent Information Systems  
Dept. of Materials Science and Engineering
University of Ioannina   
Ioannina, Greece

Contact: *orepap@uoi.gr*


# INSTALL

_pip install git+https://github.com/Orepap/Clust3D.git_


# PREREQUISITES
```python
scikit-learn 1.0.2
numpy 1.21.6
pandas 1.4.0
matplotlib 3.5.1
```

# USAGE
```python
from Clust3D.main import Clust3D

data_file = "..." # path to the data file
correlation_file = "..." # path to the correlation file
n_neurons = # Specify the number of neurons

clusters, neurons, cl_labels = Clust3D(data_file=data_file, correlation_file=correlation_file, n_neurons=n_neurons)
```
The code returns a dictionary with the cluster memberships (clusters),    
a numpy array of the clustering centers (neurons)  
and a list with the clustering labels (cl_labels).  

Folder "Examples" provides three usage examples based on timeseries gene expression data from systemic autoinflammatory diseases.


# INPUT FILES
Clust3D requires two files as input.  
The first one is the data file (txt or csv) which contains a table with the features (rows) and all the samples of the different time intervals (columns).  
The second one, is a UTF-8 or ANSI format txt file, in which the correlation between the class labels, for which the clustering will take place, along with their corresponding samples in the different time intervals has to be specified.

Specifically for Data Matrix Files from the Gene Expression Omnibus, the following steps are required in preparation of those two files.

For the **data file**, the user needs to:  
•	Download the Series Matrix File txt file of the desired GSE Series from the Gene Expression Omnibus  
•	Delete everything from within the txt file up until (and) the line ”!series_matrix_table_begin”  
•	Delete the very last line “!series_matrix_table_end”  
•	Save the file and exit  

![sdvfrb](https://github.com/Orepap/Clust3D/assets/93657525/fb7bb192-d8b0-4241-b48c-2976556c9f48)  
Snapshot example of a data file.  


For the **correlation file**, the user needs to create a txt file in UTF-8 or ANSI format, like the example below:

![εικόνα](https://github.com/Orepap/Clust3D/assets/93657525/80b3de60-8e8e-481e-8466-0033ddc2d5b6)

In this particular example, P1 is a patient (arbitrary naming for the first class label) and GSM1, GSM2 and GSM3 correspond to that patients’s samples at (e.g.) three time intervals. The following have to be true for the correlation file:

1) One space per word  
2) No space at the end of each line  
3) No space at the end of the file  
4) No number as the first label letter

# PARAMETERS
"Clust3D Parameters.md" details all the tunable parameters of Clust3D.

# HOW IT WORKS
<p align="justify">
The Euclidean distance is first computed between an input sample and all the neurons. Then, the neuron that has the smallest distance to the sample is declared as the best matching unit (BMU) and its weights along with its nearest neighbor neurons (self-organizing) are re-adjusted to closer mimic the input sample. The novelty is the introduction of matrix norms as distance concepts. Conventional distance metrics like the Euclidean, are typically calculated between vectors. In Clust3D, where the data points are matrices, the distance between two data points is defined as the mathematical norm of the matrix of their differences. As such, Clust3D introduces the capability to train the neural network given the input samples and the neurons as matrices and not just as vectors, containing both the temporal and the spatial information. Thus, the clustering can be implemented directly on the patients, given the different timepoints altogether. </p>

The selected norm is the Frobenius norm:  
![εικόνα](https://github.com/Orepap/TMDC/assets/93657525/2de1dec0-3b0c-46e7-88fa-b8a4dc960f15)  

The file "Equation.docx" details all the equations used for the neural network training.  
 
# ADVANCED
The user can input their own preprocessed data file to be trained by Clust3D, by setting the preprocessing parameters to "none".  
```python
from Clust3D.main import Clust3D

data_file = "..." # path to the data file
correlation_file = "..." # path to the correlation file
n_neurons = # Specify the number of neurons

clusters, neurons, cl_labels = Clust3D(data_file=data_file, correlation_file=correlation_file, n_neurons=n_neurons, dim_red="none", imputation="none", scaling="none")
```
**Caution!** The data file should be in the same format as a GEO Series Matrix File (features as rows, samples as columns).  
.  
.  
The user can change/experiment with different parameters on the source code found in the "Clust3D" folder.  
  


# STANDALONE VERSION  
  
You can find a standalone version (.exe) of Clust3D [here](https://drive.google.com/drive/folders/13GMeJf4_lBE9GbTf__8FlC8FEaXmFcO1).  
  
The user has the ability to select to run Clust3D with all default parameters or tune each one.  
Running Clust3D with default parameters is not always viable or the most robust approach.

**Viability**: Example 2 in Examples folder requires a lower "max_n_neurons" value than the default due to sample size (see parameters)  
**Robustness**: Examples 2 and 3 is best used with the "depth" parameter set to "auto". This provides the best consistency (see parameters) 

**Caution!** The standalone version is a beta version. The user should expect to encounter minor bugs and/or inconsistencies.


![image](https://github.com/Orepap/Clust3D/assets/93657525/0d8fda6f-0dcb-4eb4-9758-5f6e51ffff4d)











