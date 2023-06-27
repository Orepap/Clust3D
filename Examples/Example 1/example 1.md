The user has to first download the Series Matrix File from GEO (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80060)  
and follow the preprocessing steps explained in the readme.md file.  
The correlation file is provided in Folder “Example 1”.  

GSE80060 provides gene expression data of whole blood of systemic juvenile idiopathic arthritis (SJIA) patients treated with canakinumab.

Usage:  
```python
from TMDC.main import TMDC

data_file = "..." # path to the data file
correlation_file = "..." # path to the correlation file

clusters, neurons, cl_labels = TMDC(data_file=data_file, correlation_file=correlation_file, n_neurons=-1)
```  

The code returs a dictionary with the cluster memberships (clusters),  the clustering centers (neurons) and a list with the clustering labels (cl_labels).
