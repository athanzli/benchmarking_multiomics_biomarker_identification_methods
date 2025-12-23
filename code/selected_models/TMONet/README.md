# TCGA PanCancer Training Data

## Table of Contents

Mutation Data: 0/1 Coding, 4539 Features; <br>
Expression Data: numeric, 6016 Features; <br>
Methylation Data: numeric, 6617 Features; <br>
CNA Data: numeric, 7460 Features; <br>
<br>

Clinical Data: 8217/8238 Samples; <br>
--Data_Available: Samples with clinical information or not; <br>
--Cancer_Type: The cancer type of samples; <br>
--OS, OS.time: Overall survival status & time (days), OS is the period from the date of diagnosis until the date of death from any cause; <br>
--DSS, DSS.time: Death from the disease status & time (days), DSS is the death from the disease; <br>
--DFI, DFI.time: Disease-free status after their initial diagnosis and treatment status & time (days); <br>
--PFI, PFI.time: Progression-Free Interval status & time (days), period from the date of diagnosis until the date of the first occurrence of a new tumor event; <br>

## Downstream tasks
OS and PFI could be preferred for survival analysis. <br>
Cancer_Type can be used to training TCGA cancer classifier.<br>

## Load Data

```python
mutData = pd.read_csv('./TCGA_PanCancerAtlas_Mutation_230109.csv',header=0,index_col=0)
methData = pd.read_csv('./TCGA_PanCancerAtlas_Methylation_230109.csv',header=0,index_col=0)
expData = pd.read_csv('./TCGA_PanCancerAtlas_Expression_230109.csv',header=0,index_col=0)
cnaData = pd.read_csv('./TCGA_PanCancerAtlas_CNA_230109.csv',header=0,index_col=0)
clinData = pd.read_csv('./TCGA_PanCaner_Cinical_Info_0109.csv',header=0,index_col=0)
