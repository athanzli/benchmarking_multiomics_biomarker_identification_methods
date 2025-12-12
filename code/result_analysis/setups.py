import pickle
from itertools import product
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import sys

sys.path.append("..")
from utils import *
from metrics import *
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
PALETTE_OMICS = {
    'CNV'   : "#ffd8b1",
    'DNAm'  : "#95D378",
    'SNV'   : "#EB8888",
    'mRNA'  : '#5C90E4',
    'miRNA' : "#A0DBD5",
}
MULTI_OMICS_DATA_TYPE_PALETTE = {
    'CNV+DNAm+mRNA':      '#A5BEAF',
    'CNV+SNV+mRNA':       '#C2A5B4',
    'CNV+mRNA+miRNA':     '#A9C1CE',
    'DNAm+SNV+mRNA':      '#9FA4A1',
    'DNAm+mRNA+miRNA':    '#86BFBB',
    'SNV+mRNA+miRNA':     '#A2A6C0',
    'DNAm+mRNA'         : '#78B2AE',   # (149,211,120) ⟶ (92,144,228)
    'CNV+DNAm+SNV+mRNA' : '#B7B1A5',   # (255,216,177)+(149,211,120)+(235,136,136)+(92,144,228)
}

# args = argparse.Namespace()
# args.task = 'BRCA_Subtype_PAM50'
# args.task = 'drug_response_Cisplatin'
# task = args.task

DATA_PATH = "/home/athan.li/eval_bk/data/"
# DATA_PATH = "D:/Projects/eval_bk/data/"

MODELS_KEEPING_NEG_SCORES = [ 
    'MOGONET', 'MORE', 'MoAGLSA',
    'ReliefF', 
]
MODELS_OUTPUT_RANKS = ['RF_RFE','SVM_RFE']
MODELS_RES_STORED_AS_RANKS = ['RF_RFE','SVM_RFE','DIABLO','asmPLSDA']
MODELS_GENE_CENTRIC = [
    'PNet', 'DeePathNet', 'Pathformer', 'GENIUS', 'GNNSubNet','GDF','DPM'
]
MODELS_WITH_BARE_GENES_FT_SCORES = [
    'GNNSubNet','DPM'
]
MODELS_OUTPUT_P_VALUES = ['DPM'] # different from ranks. DPM outputs p vlaues
MODELS_OUTPUT_ONE_MINUS_P_VALUES = ['ttest', 'mannwhitneyu', 'onewayanova', 'kruskalwallis']

DL_MODELS = [
    'DeePathNet', 'Pathformer', 'TMONet', 'CustOmics', 'DeepKEGG',
       'GENIUS', 'PNet', 'MOGLAM', 'GNNSubNet', 'MOGONET', 'MoAGLSA',
       'MORE'
]
NONDL_MODELS = ['Stabl',
            'DIABLO',
            'DPM',
            'GDF',
            'asmPLSDA',
            'MOFA',
            'GAUDI',
            'MCIA',]
CLASSICAL_MODELS = ['SVM_ONE',
     'PLSDA',
     'RF_VI',
     'SVM_RFE',
     'RF_RFE',
     'mannwhitneyu',
     'ReliefF',
     'IG',
     'ttest']

# NOTE ORDERING
categories = {
    # dl
    'cat1': ['DeePathNet',  'Pathformer'], # transformer
    'cat2': ['TMONet', 'CustOmics'], # autoencoder
    'cat3': ['DeepKEGG', 'GENIUS', 'PNet'], #FNN 
    'cat4': ['MOGLAM', 'GNNSubNet', 'MOGONET', 'MoAGLSA', 'MORE'], # graph
    'cat5': [ # non dl
            'Stabl',
            'DIABLO',
            'DPM',
            'GDF',
            'asmPLSDA',
            'MOFA',
            'GAUDI',
            'MCIA',
            ],
    'cat6': [ # classical
        'SVM_ONE',
        'RF_VI',
        'PLSDA',
        'SVM_RFE',
        'RF_RFE',
        'IG',
        'mannwhitneyu',
        'ReliefF',
        'ttest'],
}
MODEL2COLOR = {}
palette_cat1 = sns.dark_palette("#e6a0a0", n_colors=len(categories['cat1']))
for model, color in zip(categories['cat1'], palette_cat1):
    MODEL2COLOR[model] = '#e6a0a0'
palette_cat2 = sns.dark_palette("#e8ad76", n_colors=len(categories['cat2']))
for model, color in zip(categories['cat2'], palette_cat2):
    MODEL2COLOR[model] = '#e8ad76'
palette_cat3 = sns.dark_palette("#abc6f1", n_colors=len(categories['cat3']))
for model, color in zip(categories['cat3'], palette_cat3):
    MODEL2COLOR[model] = "#abc6f1"
palette_cat4 = sns.dark_palette("#d4b9eb", n_colors=len(categories['cat4']))
for model, color in zip(categories['cat4'], palette_cat4):
    MODEL2COLOR[model] = "#d4b9eb"
palette_cat5 = sns.dark_palette("#badea9", n_colors=len(categories['cat5']))
for model, color in zip(categories['cat5'], palette_cat5):
    MODEL2COLOR[model] = '#badea9'
palette_cat6 = sns.dark_palette("#90e5e8", n_colors=len(categories['cat6']))
for model, color in zip(categories['cat6'], palette_cat6):
    MODEL2COLOR[model] = '#90e5e8'
# palette_cat7 = sns.dark_palette("purple", n_colors=len(categories['cat7']))
# for model, color in zip(categories['cat7'], palette_cat7):
#     MODEL2COLOR[model] = 'purple'
# palette_cat8 = sns.dark_palette("grey", n_colors=len(categories['cat8']))
# for model, color in zip(categories['cat8'], palette_cat8):
#     MODEL2COLOR[model] = "grey"

# Define the overall model order: grouped by the defined categories
MODEL_ORDER = np.array((categories['cat1'] + categories['cat2'] + 
            categories['cat3'] + categories['cat4'] + 
            categories['cat5'] + categories['cat6'] # + 
            ))

# NOTE
MODELS = MODEL_ORDER
NON_CLASSICAL_MODELS = [ # ordered by performance
 'DeePathNet',
 'DeepKEGG',
 'MOGLAM',
 'TMONet',
 'CustOmics',
 'GENIUS',
 'Pathformer',
 'GNNSubNet',
 'PNet',
 'MOGONET',
 'MORE',
 'MoAGLSA',
 'DIABLO',
 'GAUDI',
 'GDF',
 'Stabl',
 'asmPLSDA',
 'MOFA',
 'DPM',
 'MCIA']

TASK_NAME_CORRECTION_MAP = {
 'survival_BRCA' : 'BRCA Survival',
 'survival_LUAD' : 'LUAD Survival',
 'survival_COADREAD' : 'COADREAD Survival',
 'drug_response_Cisplatin-BLCA' : 'Cisplatin drug response (BLCA)',
 'drug_response_Temozolomide-LGG' : 'Temozolomide drug response (LGG)'
}

CLASSICAL_MODELS = [        'SVM_ONE',
        'RF_VI',
        'PLSDA',
        'SVM_RFE',
        'RF_RFE',
        'IG',
        'mannwhitneyu',
        'ReliefF',
        'ttest']

MODEL2COLOR['P-Net'] = '#e6a0a0'
MODEL2COLOR['TMO-Net'] = '#e8ad76'
MODEL2COLOR['MoAGL-SA'] = "#d4b9eb"
MODEL2COLOR['GNN-SubNet'] = "#d4b9eb"

MODEL2COLOR['asmbPLS-DA'] = '#badea9'

MODEL2COLOR['PLS-DA'] = '#90e5e8'
MODEL2COLOR['SVM-Coef'] = '#90e5e8'
MODEL2COLOR['RF-Gini'] = '#90e5e8'
MODEL2COLOR['RF-RFE'] = '#90e5e8'
MODEL2COLOR['SVM-RFE'] = '#90e5e8'
MODEL2COLOR['T-test'] = '#90e5e8'
MODEL2COLOR['MW-test'] = '#90e5e8'

MODELS_WO_PERF  = [
    'ttest',
    'mannwhitneyu',
    'onewayanova',
    'kruskalwallis',
    'IG',
    'CHI2',
    'SU',
    'GR',
    'ReliefF',
    'MOFA',
    'MCIA',
    'GAUDI',
    'DPM'
]
PERF_MODEL_ORDER = MODEL_ORDER[~(np.isin(MODEL_ORDER, MODELS_WO_PERF))]

MODS_COMBS = [
['CNV', 'DNAm', 'mRNA'],
['CNV', 'SNV', 'mRNA'],
['CNV', 'mRNA', 'miRNA'],
['DNAm', 'SNV', 'mRNA'],
['DNAm', 'mRNA', 'miRNA'],
['SNV', 'mRNA', 'miRNA'],
]
MODS = ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA']
MODS_COMBS_STR = ['+'.join(mods_comb) for mods_comb in MODS_COMBS]

FOLDS = np.arange(5)

TASKS = [
    "survival_BRCA",
    "survival_LUAD",
    "survival_COADREAD",
    "drug_response_Cisplatin-BLCA",
    "drug_response_Temozolomide-LGG",
]

BROAD_TASKS = ['survival', 'drug_response']
MODS_COMBS_STR_WO_DNAM = [
 'CNV+SNV+mRNA',
 'CNV+mRNA+miRNA',
 'SNV+mRNA+miRNA']

MODEL_NAME_CORRECTION_MAP={
    'PNet':'P-Net',
    'TMONet':'TMO-Net',
    'MoAGLSA':'MoAGL-SA',
    'GNNSubNet':'GNN-SubNet',
    'CustOmics' : 'CustOmics',
    'DeePathNet' : 'DeePathNet',
    'DeepKEGG' : 'DeepKEGG',
    'GENIUS' : 'GENIUS',
    'MOGLAM' : 'MOGLAM',
    'MOGONET' : 'MOGONET',
    'MORE' : 'MORE',
    'Pathformer' : 'Pathformer',

    'SVM_ONE':'SVM-Coef',
    'SVM_RFE':'SVM-RFE',
    'RF_VI':'RF-Gini',
    'RF_RFE':'RF-RFE',
    'ttest':'T-test',
    'mannwhitneyu':'MW-test',

    'MOFA':'MOFA',
    'Stabl':'Stabl',
    'MCIA':'MCIA',
    'GAUDI':'GAUDI',
    'DPM':'DPM',
    'GDF':'GDF',
    'asmPLSDA':'asmbPLS-DA',
    'DIABLO':'DIABLO',
}

def plot_models_by_dataset(df, df_folds=None, df_std=None, savefig=None, title=None):
    r"""
    
    Args:
        df_folds: dataframe with (num_folds*df.index, df.columns)
    """
    if df_folds is not None:
        assert set(df.index.unique())==set(df_folds.index)
        assert (df.columns==df_folds.columns).all()
    if df_std is not None:
        assert set(df.index.unique())==set(df_std.index)
        assert (df.columns==df_std.columns).all()
    
    num_cols = df.shape[1]
    models = df.index.tolist()
    y_pos = np.arange(len(models))
    figsize = (num_cols * 3, len(models) * 0.25 + 2)
    fig, axes = plt.subplots(ncols=num_cols, figsize=figsize, sharey=False)
    for ax, col in zip(axes, df.columns):
        if df_std is not None:
            ax.barh(y_pos,
                    df[col].values,
                    
                    # TODO
                    xerr=df_std[col].values, # NOTE
                    capsize=3,
                    error_kw={'elinewidth':1, 'alpha':0.8},

                    edgecolor='black',
                    color=[MODEL2COLOR[m] for m in models],
                    linewidth=0.0)
        else:
            ax.barh(y_pos,
                    df[col].values,
                    edgecolor='black',
                    color=[MODEL2COLOR[m] for m in models],
                    linewidth=0.0)

        if df_folds is not None: # dots
            for i, m in enumerate(models):
                xs = df_folds.loc[m, col].values
                ys = np.random.normal(loc=y_pos[i], scale=0.1, size=xs.shape) # jitter
                ax.scatter(xs, ys, color='black', s=20, alpha=0.7, zorder=5)

        ax.axvline(0.5, color='red', linestyle='-', linewidth=1) # TODO
        ax.set_title(col, rotation=0)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle=':', linewidth=0.5)
        if ax is axes[0]:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.set_ylabel("")
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()
    if savefig is not None: plt.savefig(savefig, dpi=300)
    plt.show()

MODEL_BK_IDEN_CAT = {
    'MOGONET':"Perturbation",
    'MORE':"Perturbation",
    'MoAGLSA':"Perturbation",
    'MOGLAM':"Matrix factorization",
    'MoGCN':"NN weights",
    'CustOmics':"SHAP",
    'TMONet':"IG",
    'DeepKEGG':"DeepLIFT",
    'DeePathNet':"SHAP",
    'PNet':"DeepLIFT",
    'Pathformer':"SHAP",
    'GENIUS':"IG",
    'GNNSubNet':"GNNExplainer",
    'AGCN':"LRP",
    ####################
    'ttest':"test",
    'IG':"Entropy",
    'CHI2':"test",
    'SU':"Entropy",
    'GR':"Entropy",
    'OneR':"Univariate",
    'ReliefF':"Multivariate",
    'SVM_ONE':"Multivariate",
    'SVM_RFE':"RFE",
    'RF_VI':"Multivariate",
    'RF_RFE':"RFE"
    ####################
}

MULTIOMICS_DATA_TYPES = [
"CNV+DNAm+mRNA",
"CNV+SNV+mRNA",
"CNV+mRNA+miRNA",
"DNAm+SNV+mRNA",
"DNAm+mRNA+miRNA",
"SNV+mRNA+miRNA"
]