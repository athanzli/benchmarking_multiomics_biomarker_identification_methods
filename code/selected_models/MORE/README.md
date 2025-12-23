# MORE：a multi-omics data-driven hypergraph integration network for biomedical data classification and biomarker identification
![figure.png](https://github.com/Wangyuhanxx/MORE/blob/main/model.png)
## Overview of MORE
MORE is mainly composed of two modules: MOHE for omics-specific knowledge learning and MOSA for multi-omics integration. Preprocessing is first performed on each omics modality to remove noise and redundant features. A comprehensive hyperedge group is constructed by extensively exploring the correlations within and across modalities. The hypergraph is generated from the comprehensive hyperedge group. Subsequently, the hypergraph, along with the features from each omics modality, is inputted into the MOHE module to extract the omics-specific features. Eventually, the MOSA module is employed to adaptatively aggregate valuable information across modalities for final prediction. 
## Code
main_MORE.py: MORE for classification tasks\
main_biomarker.py: MORE for identifying biomarkers\
models.py: MORE model\
train_test.py: Training and testing functions\
param.py: Parameter setting functions\
feat_importance.py: Feature importance functions\
utils.py: Supporting functions
## Requirement
```console
pip install torch=1.9.1
pip install python=3.9
```
