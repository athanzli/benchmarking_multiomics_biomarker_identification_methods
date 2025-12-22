# Benchmarking computational methods for multi-omics biomarker discovery in cancer

This repository provides a benchmark pipeline to evaluate the biomarker identification performance of multi-omics integration methods. Five real-world TCGA datasets with systematically curated gold-standard biomarker sets are provided for comprehensive evaluation. The user can conveniently run their method and compare its performance with 20 other baselines.

For more information, see our manuscript *Benchmarking computational methods for multi-omics biomarker discovery in cancer*.

---

## Installation

### Download the benchmark data

Download `data.zip` from `https://zenodo.org/records/17860662`, unzip it, and place the extracted `data/` folder at the repository root.

At minimum, the following files must exist:

- `data/TCGA/TCGA_cpg2gene_mapping.csv`
- `data/TCGA/TCGA_miRNA2gene_mapping.csv`
- `data/bk_set/processed/survival_task_bks.csv`
- `data/bk_set/processed/drug_response_task_bks.csv`
- `data/TCGA/<task_name>/*_CNV+DNAm+SNV+mRNA+miRNA.pkl` (for each task you run)

### Dependencies

- Python 3.10+ recommended

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn rbo
```


## Usage

This section guides you through running your method on our benchmark pipeline. The process consists of two main steps:

1. Implement a wrapper function for your method
2. Run the benchmark using the provided pipeline

---

### Step 1: Implement your method's wrapper function

Create a function that wraps your method and follows the required signature. This function will be called by the benchmark pipeline for each dataset, omics combination, and cross-validation fold.

#### Function signature

```python
def run_method_custom(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    mode: int = 0,  # Required parameter - see "mode" section below
):
    """
    A custom function to run your method for benchmarking.
    Note that our pipeline will input its train, val, and test data/labels to this 
    wrapper function, but you can freely choose from the input data/labels of the 
    train/val/test splits to run your method, based on your experimental design.

    Args:
        X_train (pd.DataFrame): Training features. 
            - Index: sample IDs
            - Columns: feature names in "MOD@feature" format (e.g., "mRNA@TP53", "DNAm@cg00000029")
        y_train (pd.DataFrame): Training labels.
            - Index: sample IDs  
            - Column: 'label' (for binary classification) or 'T', 'E' (for survival analysis)
        X_val (pd.DataFrame): Validation features (same format as X_train)
        y_val (pd.DataFrame): Validation labels (same format as y_train)
        X_test (pd.DataFrame): Test features (same format as X_train)
        y_test (pd.DataFrame): Test labels (same format as y_train)
        mode (int): Specifies the format of your output feature scores (see below)

    Returns:
        ft_score (pd.DataFrame): Feature importance scores.
            - Index: feature names (see formats below)
            - Single column: importance scores (higher = more important)
    """
    # Your method implementation here
    ...
    return ft_score
```

#### Input data format

| Omics Type | Feature Level | Example Feature Names |
|------------|---------------|----------------------|
| mRNA | Gene-level | `mRNA@TP53`, `mRNA@KRAS`, `mRNA@EGFR` |
| CNV | Gene-level | `CNV@APOC1`, `CNV@MYC`, `CNV@ERBB2` |
| SNV | Gene-level | `SNV@TP53`, `SNV@BRAF`, `SNV@PIK3CA` |
| DNAm | CpG-level | `DNAm@cg00000029`, `DNAm@cg22832044` |
| miRNA | miRNA-level | `miRNA@hsa-miR-100-5p`, `miRNA@hsa-let-7a-5p` |

#### Output format: `ft_score`

Your function must return a pandas DataFrame with:
- **Index**: Feature names in **`MOD@molecule_name`** format (e.g., `mRNA@TP53`, `DNAm@cg00000029`, `miRNA@hsa-miR-100-5p`)
- **Single column**: Importance scores where **higher values indicate greater importance**

> ⚠️ The index of `ft_score` **must** use the `MOD@molecule_name` format, where `MOD` is one of `mRNA`, `CNV`, `SNV`, `DNAm`, or `miRNA`. This format is required for the benchmark to correctly process and evaluate your results. The only exception is when using `mode=2`, where gene names without modality prefix are accepted.

> ⚠️ If your method produces scores where sign indicates directionality (not importance), convert to absolute values before returning.

#### The `mode` parameter

The `mode` parameter tells the benchmark how to interpret your feature names during evaluation. All evaluations are performed at the gene level (the benchmark automatically maps CpG and miRNA features to genes using provided mapping files), so the benchmark needs to know how to convert your scores.

| Mode | Description | Feature Name Format | Example |
|------|-------------|---------------------|---------|
| `0` | Molecule-centric (default for most methods) | Original molecule-level names with modality prefix | `DNAm@cg00000029`, `miRNA@hsa-miR-100-5p`, `mRNA@TP53` |
| `1` | Gene-centric with modality | Gene names with modality prefix | `DNAm@TP53`, `miRNA@KRAS`, `mRNA@EGFR` |
| `2` | Gene-centric without modality | Gene names only (no modality prefix) | `TP53`, `KRAS`, `EGFR` |

**Choose based on your method's output:**
- **Mode 0**: Your method operates on original features and outputs scores for CpGs, miRNAs, genes, etc.
- **Mode 1**: Your method maps features to genes but retains modality information (e.g., distinguishes `DNAm@TP53` from `mRNA@TP53`)
- **Mode 2**: Your method outputs gene-level scores without distinguishing which omics type the score came from

#### Example implementation

```python
def run_method_custom(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    mode: int = 0,  # Mode 0: molecule-level output
):
    """Example wrapper for a random forest-based feature selection method."""
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np
    
    y_trn = y_train['label'].values
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.values, y_trn)
    
    importances = model.feature_importances_
    
    ft_score = pd.DataFrame(
        index=X_train.columns,
        data={'score': importances}
    )
    
    return ft_score
```

---

### Step 2: Run the benchmark

Pass your wrapper function to `run_benchmark()`. The benchmark uses 5-fold cross-validation with pre-defined splits to ensure reproducibility and fair comparison across methods.

```python
from benchmark_pipeline import run_benchmark

# Run benchmark with default settings (all datasets, all omics combinations, all folds)
acc_res, sta_res = run_benchmark(
    run_method_custom_func=run_method_custom,
)
```

> 💡 **Recommendation**: We recommend using the **default settings** (all five datasets, all six omics combinations, all five folds) for comprehensive evaluation. You are free to customize these settings if you are only interested in certain task datasets (e.g., survival only), your method is specifically designed for certain omics combinations, or if you need to save time during evaluation.

The benchmark automatically handles train/validation/test splitting, label preparation, and data scaling. Your method receives pre-processed data ready for model training/running.

#### Available datasets

You can specify which dataset(s) to run using either integer codes or string names:

| Code | Dataset Name | Task Type | Cancer Type |
|------|--------------|-----------|-------------|
| `0` | `survival_BRCA` | Survival prediction | Breast Cancer |
| `1` | `survival_LUAD` | Survival prediction | Lung Adenocarcinoma |
| `2` | `survival_COADREAD` | Survival prediction | Colorectal Cancer |
| `3` | `drug_response_Cisplatin-BLCA` | Drug response prediction | Bladder Cancer (Cisplatin) |
| `4` | `drug_response_Temozolomide-LGG` | Drug response prediction | Low-Grade Glioma (Temozolomide) |

#### Configuration options

```python
acc_res, sta_res = run_benchmark(
    run_method_custom_func=run_method_custom,
    
    # Select specific dataset(s) - default: all datasets [0,1,2,3,4]
    datasets_to_run=[0, 1],  # Run on BRCA and LUAD survival tasks
    # Or use string names:
    # datasets_to_run=['survival_BRCA', 'survival_LUAD'],
    
    # Select specific omics combination(s) - default: all 6 tri-omics combinations (with mRNA included)
    omics_types=['DNAm', 'mRNA', 'miRNA'],  # Single combination as a list
    # Or multiple combinations as a list of lists:
    # omics_types=[['DNAm', 'mRNA', 'miRNA'], ['CNV', 'mRNA', 'miRNA']],
    
    # Select specific fold(s) - default: all 5 folds (0-4)
    fold_to_run=[0, 1, 2],  # Run only folds 0, 1, 2
    # Or single fold:
    # fold_to_run=0,
    
    # Survival label handling - default: 'binary'
    # By default, survival times are converted to binary labels (long/short based on median).
    # Set to 'continuous' if your method handles survival analysis directly (with censoring info).
    surv_op='binary',
    
    # Data scaling method - default: 'standard'
    scaling='standard',  # Z-score normalization
    # scaling='minmax',  # Min-max normalization (0-1)
    # scaling=None,  # No scaling
    
    # Output path - default: './result/'
    res_save_path='./result/my_method/',
)
```

#### Default omics combinations

When `omics_types=None` (default), the benchmark runs on all 6 tri-omics combinations that include mRNA:

1. `['DNAm', 'mRNA', 'miRNA']`
2. `['CNV', 'mRNA', 'miRNA']`
3. `['SNV', 'mRNA', 'miRNA']`
4. `['DNAm', 'CNV', 'mRNA']`
5. `['DNAm', 'SNV', 'mRNA']`
6. `['CNV', 'SNV', 'mRNA']`

#### Custom omics combinations

You are free to set `omics_types` to other combinations beyond the default tri-omics sets. For example:
- **Two omics**: `['mRNA', 'DNAm']`
- **Four omics**: `['mRNA', 'DNAm', 'CNV', 'miRNA']`
- **Five omics**: `['mRNA', 'DNAm', 'CNV', 'SNV', 'miRNA']`
- **Tri-omics without mRNA**: `['DNAm', 'CNV', 'miRNA']`

In such cases, the generated comparison plots will display your method's results for the specified omics combinations alongside baseline results averaged across all default omics combinations and folds. While you can still get a sense of your method's relative performance, note that these comparisons are not strictly direct since the baselines use different omics combinations.

#### Understanding the generated plots

The benchmark automatically generates comparison plots saved to `./figures/`. The baseline results shown in these plots depend on your configuration:

- **Default settings** (`omics_types=None`, `fold_to_run=None`): Plots show baseline results averaged across all 6 tri-omics combinations and all 5 folds
- **Specific omics/folds specified**: Plots filter baseline results to match only the omics combinations and folds you specified
- **Custom omics combinations** (not in the default 6): Plots show baseline results averaged across all default combinations, providing a reference comparison (though not strictly equivalent)

---

### Understanding the output

> 📊 **Start here**: After running the benchmark, check the **generated comparison plots** saved in `./figures/`. These visualizations provide an immediate overview of how your method performs compared to baseline methods across all evaluation metrics.

#### Returned results

The benchmark returns two dictionaries containing evaluation metrics:

**`acc_res` - Accuracy Metrics** (per dataset × omics combination × fold):
```python
{
    'NDCG': {(dataset_name, omics_comb, fold): score, ...},  # Normalized DCG
    'RR': {(dataset_name, omics_comb, fold): score, ...},    # Reciprocal Rank
    'AR': {(dataset_name, omics_comb, fold): score, ...},    # Average Recall
    'MW_pval': {(dataset_name, omics_comb, fold): pval, ...} # Mann-Whitney p-value
}
```

**`sta_res` - Stability Metrics** (per dataset × omics combination):
```python
{
    'Kendall_tau': {(dataset_name, omics_comb): score, ...},  # Kendall's tau correlation
    'RBO': {(dataset_name, omics_comb): score, ...},          # Rank-Biased Overlap
    'PSD': {(dataset_name, omics_comb): score, ...}           # Percentile Standard Deviation
}
```

#### Saved files

```
result/                                     # Or your specified res_save_path
├── survival_BRCA/
│   └── ft_score_fold{0-4}.csv              # Feature scores for each fold
├── survival_LUAD/
│   └── ...
├── your_method_accuracy_results.pkl        # Accuracy metrics
└── your_method_stability_results.pkl       # Stability metrics

figures/                                    # Comparison plots
├── fig_overall_results.pdf                 # Overall results averaged across tasks
├── fig_task_Survival_BRCA.pdf              # Per-task results
├── fig_task_Survival_LUAD.pdf
├── fig_task_Survival_COADREAD.pdf
├── fig_task_Drug_Response_Cisplatin_BLCA.pdf
├── fig_task_Drug_Response_Temozolomide_LGG.pdf
└── fig_mw_pval_boxplots.pdf                # Mann-Whitney p-value distribution
```

#### Evaluation metrics

**Accuracy Metrics** (how well your method identifies known biomarkers):
- **AR** (Average Recall): Average recall rates of biomarkers across ranking
- **NDCG** (Normalized Discounted Cumulative Gain): Measures ranking quality, giving more weight to top-ranked features
- **RR** (Reciprocal Rank): 1/rank of the first correctly identified biomarker
- **MW_pval** (Mann-Whitney p-value): Statistical test for whether biomarker scores are significantly higher

**Stability Metrics** (how consistent are rankings across folds):
- **Kendall's tau**: Rank correlation between fold rankings
- **RBO** (Rank-Biased Overlap): Top-weighted similarity measure
- **PSD** (Percentile Standard Deviation): Variation in biomarker positions across folds (lower = more stable)

---
