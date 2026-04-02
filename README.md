# Clust3D - 3D Clustering tool

<p align="justify">
Clust3D is a clustering tool designed for clustering 3D data, such as timeseries and multi-omic datasets, using a self-adjusting neural network.
Clust3D provides the capability to directly cluster 3D data, exploiting the entire data structure, without the need for flattening or decomposition of one of the dimensions (layers). </p>


**PAPER**:   
[Papagiannopoulos, Orestis D., et al. "3D Clustering of Gene Expression Data from Systemic Autoinflammatory Diseases Using Self-Organizing Maps (Clust3D)." Computational and Structural Biotechnology Journal (2024).](https://www.sciencedirect.com/science/article/pii/S2001037024001491)

 
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
```
Python    >= 3.10
numpy     >= 1.26.4
pandas    >= 2.2.3
scikit-learn >= 1.5.2
matplotlib >= 3.10.0
scipy     >= 1.14.1
fancyimpute >= 0.7.0
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


# UPDATES
<p align="justify">
 
**v1.1** — Added `scaling_per_dimension` parameter enabling independent per-dimension (layer) scaling.
When `scaling_per_dimension=True`, a separate scaler is fitted and applied to each
dimension (layer) independently, preserving the relative structure within each dimension (layer). This is
the recommended setting when the dimensions of the input tensor have different value
ranges or distributions. When `scaling_per_dimension=False` (default), all dimensions (layers)
are scaled jointly. This update is available in the Python package only and is not
included in the standalone version.

<p align="justify">
 
**v1.1** — Added `nan_mask` parameter enabling masked Frobenius distance computation.
When `nan_mask=True`, distance computations between sample matrices and neuron matrices
are restricted to non-NaN overlapping entries, allowing layers with structured
missingness to contribute to cluster geometry without being confounded by
missingness-driven artifacts. This update is available in the Python package only
and is not included in the standalone version.


# USAGE WITH NaN MASKING
<p align="justify">
For datasets containing missing values that should be excluded from distance
computations rather than imputed, set `nan_mask=True`. This enables the masked
Frobenius distance, which restricts distance computations to non-NaN
overlapping entries between each sample-neuron pair.

```python
from Clust3D.main import Clust3D

data_file = "..." # path to the data file
correlation_file = "..." # path to the correlation file
n_neurons = # Specify the number of neurons

clusters, neurons, cl_labels = Clust3D(
    data_file=data_file,
    correlation_file=correlation_file,
    n_neurons=n_neurons,
    imputation="none",             # no imputation — missing values are handled by nan_mask
    nan_mask=True,                 # enable masked Frobenius distance for data with missing values
    scaling_per_dimension=True     # recommended when dimensions differ in dynamic range
)
```


# INPUT FILES
<p align="justify">
 
Clust3D requires two files as input.  

<p align="justify">
The first one is the data file (txt or csv) which contains a table with the features (rows) and all the samples of the different time intervals (columns).  
The second one, is a UTF-8 or ANSI format txt file, in which the correlation between the class labels, for which the clustering will take place, along with their corresponding samples in the different time intervals has to be specified.

Specifically for Data Matrix Files from the Gene Expression Omnibus, the following steps are required in preparation of those two files.

<p align="justify">
 
For the **data file**, the user needs to:  
•	Download the Series Matrix File txt file of the desired GSE Series from the Gene Expression Omnibus  
•	Delete everything from within the txt file up until (and) the line "!series_matrix_table_begin"  
•	Delete the very last line "!series_matrix_table_end"  
•	Save the file and exit  

![sdvfrb](https://github.com/Orepap/Clust3D/assets/93657525/fb7bb192-d8b0-4241-b48c-2976556c9f48)  
Snapshot example of a data file.  


For the **correlation file**, the user needs to create a txt file in UTF-8 or ANSI format, like the example below:

![εικόνα](https://github.com/Orepap/Clust3D/assets/93657525/80b3de60-8e8e-481e-8466-0033ddc2d5b6)

In this particular example, P1 is a patient (arbitrary naming for the first class label) and GSM1, GSM2 and GSM3 correspond to that patient's samples at (e.g.) three time intervals. The following have to be true for the correlation file:

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

# STOCHASTICITY
<p align="justify">
Clust3D involves stochasticity at two stages. First, during neuron initialization,
when `neuron_init="points"` and `depth` is a finite integer, candidate data-point
combinations are sampled randomly, meaning different runs may initialize neurons from
different starting points. This can be fully eliminated by setting `depth="auto"`, which
exhaustively evaluates every possible combination and makes initialization deterministic.
Setting `random_state` to a fixed integer also controls the initialization seed, though
it does not affect the second source of stochasticity. During training, the order in
which samples are presented to the network is shuffled at every epoch using Python's
built-in random module, which is independent of `random_state`. This is intentional —
randomizing sample presentation order is a well-established practice in neural network
training that helps avoid local minima and improves convergence. Fully seeding this
behaviour would require fixing Python's built-in random seed separately, which may
interfere with convergence and is therefore not recommended. As a result, repeated runs
may still produce different clustering solutions. Repeating the training a
number of times and taking a consensus assignment across runs is always recommended as
good practice to verify the stability of the resulting partition.

# ADVANCED
<p align="justify">
 
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
<p align="justify">
 
You can find a standalone version (.exe) of Clust3D [here](https://drive.google.com/drive/folders/13GMeJf4_lBE9GbTf__8FlC8FEaXmFcO1).  
  
The user has the ability to select to run Clust3D with all default parameters or tune each one.  
Running Clust3D with default parameters is not always viable or the most robust approach.

**Viability**: Example 2 in Examples folder requires a lower "max_n_neurons" value than the default due to sample size (see parameters)  
**Robustness**: Examples 2 and 3 is best used with the "depth" parameter set to "auto". This provides the best consistency (see parameters) 

**Caution!** The standalone version is a beta version. The user should expect to encounter minor bugs and/or inconsistencies. The `nan_mask` and `scaling_per_dimension` parameters introduced in v1.1 are not available in the standalone version.


![image](https://github.com/Orepap/Clust3D/assets/93657525/0d8fda6f-0dcb-4eb4-9758-5f6e51ffff4d)
