Folder “Example 3” contains the processed and ready to be used data file and correlation file.   
They correspond to the Gene Expression Omnibus (GEO) Series Matrix File for the GSE9863 series.   
GSE9863 provides gene expression data of Kawasaki patients, and it contains three different time points corresponding to different stages of the disease.  

Usage:  
```python
from TMDC.main import TMDC

data_file = "..." # path to the data file
correlation_file = "..." # path to the correlation file

clusters, neurons, cl_labels = TTMDC(data_file=data_file, correlation_file=correlation_file, n_neurons=-1)

```  

The code returs a dictionary with the cluster memberships (clusters),  the clustering centers (neurons) and a list with the clustering labels (cl_labels).


