Folder “Example 1” contains the processed and ready to be used data file and correlation file.  
They correspond to the Gene Expression Omnibus (GEO) Series Matrix File for the GSE80060 series.  
GSE80060 provides gene expression data of whole blood of systemic juvenile idiopathic arthritis (SJIA) patients treated with canakinumab.

Usage:  
The user has to first download the corresponding files.

```python
from TMDC.main import TMDC

data_file = "..." # path to the data file
correlation_file = "..." # path to the correlation file

clusters, neurons, cl_labels = MDC(data_file=data_file, correlation_file=correlation_file, n_neurons=-1)
```  

The code returs a dictionary with the cluster memberships (clusters),  the clustering centers (neurons) and a list with the clustering labels (cl_labels).
