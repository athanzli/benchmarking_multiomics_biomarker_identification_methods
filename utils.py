# using violin plots.

import warnings
import numpy as np
import math
from itertools import combinations
from scipy.stats import kendalltau
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from typing import List, Union
import pickle as pkl

SPLITTER = '@'
# TCGA
try:
    C2G = pd.read_csv("./data/TCGA/TCGA_cpg2gene_mapping.csv", index_col=0)
    R2G = pd.read_csv("./data/TCGA/TCGA_miRNA2gene_mapping.csv", index_col=0)
except:
    raise FileNotFoundError("Please make sure the mapping files for DNAm, miRNA to gene are in the ./data/TCGA/ folder.")

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

def factorize_label(y):
    r"""
    Factorize the label.
    """
    y_uni = np.unique(y)
    value_to_index = {value: idx for idx, value in enumerate(y_uni)}
    y_fac = np.array([value_to_index[item] for item in y]).astype(np.int64)
    return y_fac, value_to_index

def mod_mol_dict(mol_ids):
    r"""
    
    Args:
        mol_ids (np.array): an array of molecule ids. e.g., array(['DNAm@cg22832044', 'DNAm@cg19580810', 'DNAm@cg14217534',
            'mRNA@CDH1', 'mRNA@RAB25', 'mRNA@TUBA1B', 'protein@E.Cadherin',
            'protein@Rab.25', 'protein@Acetyl.a.Tubulin.Lys40'], dtype=object)
    Returns:
        dict: a dictionary of mods, mol names, and mods uni.
    """
    mods = np.array([mod_id.split(SPLITTER)[0] for mod_id in mol_ids])
    mods_uni = np.unique(mods) # auto sorting in ascending order
    mols = np.array([mod_id.split(SPLITTER)[1] for mod_id in mol_ids])
    return {'mods': mods, 'mols': mols, 'mods_uni': mods_uni}

def auroc_perm(ft, bk, n_perm):
    r"""
    auroc for bk identification accuracy when ground truth is known.
    """
    if n_perm == 0:
        return auroc(ft, bk)
    else:
        ress = []
        for _ in range(n_perm):
            ft_cur = ft.copy()
            mask0 = (ft_cur['score']==0).values.flatten()
            idx_vals = ft_cur.index.values.copy()
            zeros    = idx_vals[mask0]
            shuffled = np.random.permutation(zeros)
            idx_vals[mask0] = shuffled
            ft_cur.index = idx_vals
            ress.append(auroc(ft_cur, bk))
        return np.mean(ress)

def auroc(ft, bk):
    y_true   = [1 if feature in bk else 0  for feature in ft.index.values.astype(str)]
    y_scores = ft.values.flatten().astype(np.float64)
    return roc_auc_score(y_true=y_true, y_score=y_scores)

def rr_perm(ft, bk, n_perm):
    if n_perm == 0:
        return rr(ft, bk)
    else:
        ress = []
        for _ in range(n_perm):
            ft_cur = ft.copy()
            mask0 = (ft_cur['score']==0).values.flatten()
            idx_vals = ft_cur.index.values.copy()
            zeros    = idx_vals[mask0]
            shuffled = np.random.permutation(zeros)
            idx_vals[mask0] = shuffled
            ft_cur.index = idx_vals
            ress.append(rr(ft_cur, bk))
        return np.mean(ress)

def rr(ft, bk):
    sorted_genes = ft.sort_values('score', ascending=False, inplace=False).copy().index.to_numpy()
    return 1 / (np.where(np.isin(sorted_genes, bk))[0][0] + 1)

def accuracy_top_k_perm(ft, bk, k, n_perm): 
    if n_perm == 0:
        return accuracy_top_k(ft, bk, k)
    else:
        ress = []
        for _ in range(n_perm):
            ft_cur = ft.copy()
            mask0 = (ft_cur['score']==0).values.flatten()
            idx_vals = ft_cur.index.values.copy()
            zeros    = idx_vals[mask0]
            shuffled = np.random.permutation(zeros)
            idx_vals[mask0] = shuffled
            ft_cur.index = idx_vals
            ress.append(accuracy_top_k(ft_cur, bk, k))
        return np.mean(ress)

def accuracy_top_k(ft, bk, k):
    if k is None:
        k = len(bk)
    sorted_genes = ft.sort_values('score', ascending=False, inplace=False).copy().index.to_numpy()
    return np.isin(sorted_genes[:k], bk).sum() / k

def ndcg_perm(relevances, ft, n_perm=0):
    if n_perm == 0:
        return ndcg(relevances)
    else:
        ress = []
        for _ in range(n_perm):
            mask0 = (ft['score']==0).values.flatten()
            tmp = relevances[mask0].copy()
            shuffled = np.random.permutation(tmp)
            relevances[mask0] = shuffled
            ress.append(ndcg(relevances))
        return np.mean(ress)

def ndcg(relevances, alpha=1.0):
    """
    Normalized Discounted Cumulative Gain (NDCG).
    Args:
        relevance scores in the order of ranked ele.
    """
    def gain_func(rel):
        # NOTE choose gain function
        # return 2**rel - 1
        return rel # identity function
    def dcg(relevances):
        return sum(gain_func(rel) / (math.log2(idx + 1 + 1))**alpha for idx, rel in enumerate(relevances))
    actual_dcg = dcg(relevances)
    ideal_dcg = dcg(sorted(relevances, reverse=True)) # descending sorting
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

def mw_test(scores, is_target, alternative="greater",
            exact_if_possible=True):
    """Mann-Whitney test.
    """

    scores   = np.asarray(scores, dtype=float)
    is_target = np.asarray(is_target, dtype=bool)
    x = scores[is_target]           # targets, size n1
    y = scores[~is_target]          # non-targets, size n2

    method = "exact" if (exact_if_possible and
                         len(set(scores)) == len(scores) and
                         min(len(x), len(y)) <= 50) else "asymptotic"
    U, p = mannwhitneyu(x, y, alternative=alternative, method=method)
    # auc = U / (len(x) * len(y))

    return p

def avg_recall_top_k_perm(ft, bk, K, n_perm=0):
    if n_perm == 0:
        return avg_recall_top_k(ft, bk, K)
    else:
        ress = []
        for _ in range(n_perm):
            ft_cur = ft.copy()
            mask0 = (ft_cur['score']==0).values.flatten()

            idx_vals = ft_cur.index.values.copy()
            zeros    = idx_vals[mask0]
            shuffled = np.random.permutation(zeros)
            idx_vals[mask0] = shuffled
            ft_cur.index = idx_vals
        
            ress.append(avg_recall_top_k(ft_cur, bk, K))
        return np.mean(ress)

def avg_recall_top_k(ft, bk, K):
    """
    Compute average recall over the top-K ranked features
    """
    # sort descending by score
    sorted_genes = ft.sort_values('score', ascending=False, inplace=False).copy().index.to_numpy()
    n = len(sorted_genes)
    m = len(bk)
    if m == 0 or K <= 0:
        return 0.0

    # only look at the top K (but not more than we have)
    K = min(K, n)
    top_k = sorted_genes[:K]
    # 1 if gene is in bk, else 0
    indicator = np.isin(top_k, bk).astype(float)
    # cumulative hits up to each i
    cumsum_hits = np.cumsum(indicator)
    # recall@i = (hits up to i) / m, for i = 1..K
    recall = cumsum_hits / m
    return recall.mean()

def kendalltau_score(rank1, rank2):
    # NOTE focus on intersected elements
    rank1 = rank1.astype(str)
    rank2 = rank2.astype(str)
    inter_ele = np.intersect1d(rank1, rank2)
    assert len(inter_ele) > 0
    if not ((len(inter_ele) == len(rank1)) and (len(inter_ele) == len(rank2))):
        rank1 = rank1[pd.Index(rank1).isin(inter_ele)]
        rank2 = rank2[pd.Index(rank2).isin(inter_ele)]

    pos_in_rank2 = {obj: i for i, obj in enumerate(rank2)}
    rank2_positions = [pos_in_rank2[obj] for obj in rank1]
    rank1_positions = list(range(len(rank1)))
    return np.float32(kendalltau(rank1_positions, rank2_positions)[0])

def average_kendall_tau(ranking_lists):
    num_pairs = 0
    tau_total = 0.0
    for rank1, rank2 in combinations(ranking_lists, 2):
        tau = kendalltau_score(rank1, rank2)
        tau_total += tau
        num_pairs += 1
    return tau_total / num_pairs if num_pairs else None

def rbo_score(S, T, p=0.95):
    """
    """
    import rbo
    # NOTE focus on intersected elements
    S = S.astype(str)
    T = T.astype(str)
    inter_ele = np.intersect1d(S, T)
    assert len(inter_ele) > 0
    if not ((len(inter_ele) == len(S)) and (len(inter_ele) == len(T))):
        S = S[pd.Index(S).isin(inter_ele)]
        T = T[pd.Index(T).isin(inter_ele)]
    assert 0<p<1
    k = len(S) # NOTE can change k
    return float(rbo.RankingSimilarity(S, T).rbo(p=p))

def average_rbo(ranking_lists, p=0.95):
    total_rbo = 0.0
    num_pairs = 0
    for S, T in combinations(ranking_lists, 2):
        total_rbo += rbo_score(S, T, p)
        num_pairs += 1
    
    return total_rbo / num_pairs if num_pairs else None

def percentile_standard_deviation(ranking_lists, bks):
    res = []
    bks = bks[np.isin(bks, ranking_lists[0])]
    if len(bks) == 0: return np.nan
    for bk in bks:
        perts = []
        for rl in ranking_lists:
            assert bk in rl
            pert = np.where(np.array(rl)==bk)[0][0] / (len(rl) - 1)
            perts.append(pert)
        res.append(np.sqrt(
                np.mean([(pert - (np.mean(perts)) )**2 for pert in perts])
        ))
    return np.mean(res)


def evaluate_stability(
    rankings: List[np.ndarray],
    bk: np.ndarray,
):
    r"""
    Args:
        rankings (List[np.ndarray]): list of gene rankings from different folds.
            Each ranking is an array of gene names.
        bk (np.ndarray): array of gold standard biomarkers.
    """
    if len(rankings) < 2:
        raise ValueError("At least two rankings are required for stability evaluation.")
    score_kendall = average_kendall_tau(rankings)
    score_rbo = average_rbo(rankings, p=0.98)
    score_psd = percentile_standard_deviation(rankings, bk)
    return score_kendall, score_rbo, score_psd

def evaluate_accuracy(
    ft_score: pd.DataFrame,
    bk: np.ndarray,
):
    r"""
    Args:
        ft_score (pd.DataFrame): feature importance scores with index as gene names
            and a single column as scores.
        bk (np.ndarray): array of ground truth gene names.
    """
    ft_gset = ft_score.index.values # update current gene set
    ft = ft_score.copy()
    mask = np.isin(bk, ft_gset)
    if len(set(bk) - set(ft_gset)) == len(bk): # if ft gset contains no bk
        raise ValueError("The feature set contains no biomarkers. Evaluation aborted.")
    bk = bk[mask]

    print("Gene ranking size:", len(ft))
    print("Biomarker set size:", len(bk))

    # NDCG
    relevances = np.full(len(ft), 0)
    relevances[np.isin(ft.index, bk)] = 1
    score_ndcg = ndcg_perm(relevances=relevances, ft=ft, n_perm=1)
    # RR
    score_rr = rr_perm(ft=ft, bk=bk, n_perm=1)
    # AR
    score_ar = avg_recall_top_k_perm(ft, bk, K=int(1*len(ft_gset)), n_perm=1)
    # mw test pval
    ft_cur = ft.copy()
    # shuffle 0-scored fts
    mask0 = (ft_cur['score']==0).values.flatten()
    idx_vals = ft_cur.index.values.copy()
    zeros    = idx_vals[mask0]
    shuffled = np.random.permutation(zeros)
    idx_vals[mask0] = shuffled
    ft_cur.index = idx_vals
    scores = len(ft_cur) - np.arange(len(ft_cur))
    pval_mw = mw_test(
        scores=scores,
        is_target=ft_cur.index.isin(bk).astype(bool),
        exact_if_possible=True)
    return score_ndcg, score_rr, score_ar, pval_mw


###############################################################################
############################### figures ################################
###############################################################################
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


TRI_OMICS_COMBS = [ # tri-omics combinations with mRNA included
    ['DNAm', 'mRNA', 'miRNA'],
    ['CNV', 'mRNA', 'miRNA'],
    ['SNV', 'mRNA', 'miRNA'],
    ['DNAm', 'CNV', 'mRNA'],
    ['DNAm', 'SNV', 'mRNA'],
    ['CNV', 'SNV', 'mRNA'],
]

TRI_OMICS_COMBS_STR = [
    '+'.join(np.sort(comb)) for comb in TRI_OMICS_COMBS
]


MODELS_UNCORRECTED_NAMES = [
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


BASELINES = ['DeePathNet',
 'DeepKEGG',
 'MOGLAM',
 'TMO-Net',
 'CustOmics',
 'GENIUS',
 'Pathformer',
 'GNN-SubNet',
 'P-Net',
 'MOGONET',
 'MORE',
 'MoAGL-SA',
 'DIABLO',
 'GAUDI',
 'GDF',
 'Stabl',
 'asmbPLS-DA',
 'MOFA',
 'DPM',
 'MCIA']

# Add Your Method to the list
ALL_MODELS = BASELINES + ['Your Method']

METRIC_CMAPS = {
    'AR': 'Reds_r',
    'NDCG': 'Oranges_r',
    'RR': 'Wistia_r',
    'RBO': 'Purples_r',
    'RPSD': 'Blues',
    'KT': None,  # Custom colors for Kendall's Tau
}

KT_COLORS = [
    "#158197",
    "#20A2B9",
    "#42CCE5",
    "#68DBEF",
    "#AAECF7",
    "#C9F0F7",
]

TASK_SHORT = {
    'survival_BRCA': 'Survival BRCA',
    'survival_LUAD': 'Survival LUAD',
    'survival_COADREAD': 'Survival COADREAD',
    'drug_response_Cisplatin-BLCA': 'Drug Response Cisplatin (BLCA)',
    'drug_response_Temozolomide-LGG': 'Drug Response Temozolomide (LGG)',
}

# load baseline results - results for each omics comb and fold
bl_acc = {} # baseline accuracy results
for task in list(TASK_SHORT.keys()):
    for metric in ['AR', 'NDCG', 'RR', 'mwtestpval_exact']:
        with open(f"./result/baseline_results/bkacc_res_{metric}_TCGA_{task}.pkl", 'rb') as f:
            bl_acc[(task, metric)] = pkl.load(f)
        for fold in [0,1,2,3,4]:
            assert bl_acc[(task, metric)][fold].columns.isin(TRI_OMICS_COMBS_STR).all()
bl_sta = {} # baseline stability results
for task in list(TASK_SHORT.keys()):
    for metric in ['KendallTau', 'RBO', 'RPSD']:
        with open(f"./result/baseline_results/stability_res_{metric}_TCGA_{task}.pkl", 'rb') as f:
            bl_sta[(task, metric)] = pkl.load(f)
        assert bl_sta[(task, metric)].columns.isin(TRI_OMICS_COMBS_STR).all()

"""
bl_acc format:
{('survival_BRCA', 'AR'): {0: DataFrame, 1: DataFrame, 2: DataFrame, 3: DataFrame, 4: DataFrame},
 ('survival_BRCA', 'NDCG'): {0: DataFrame, 1: DataFrame, 2: DataFrame, 3: DataFrame, 4: DataFrame},
 ...
} # where each DataFrame has models (names from BASELINES) as index and omics combs (from TRI_OMICS_COMBS_STR) as columns.

bl_sta format:
{('survival_BRCA', 'KendallTau'): DataFrame,
 ('survival_BRCA', 'RBO'): DataFrame,
    ('survival_BRCA', 'RPSD'): DataFrame,
    ...
} # where each DataFrame has models (names from BASELINES) as index and omics combs (from TRI_OMICS_COMBS_STR) as columns.
"""

#########
def get_metric_colors(metric, n_colors):
    if metric == 'KT':
        if n_colors <= len(KT_COLORS):
            return KT_COLORS[:n_colors]
        else:
            cmap = LinearSegmentedColormap.from_list('kt_cmap', KT_COLORS, N=n_colors)
            return [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    else:
        cmap_name = METRIC_CMAPS.get(metric, 'viridis')
        cmap = plt.colormaps.get_cmap(cmap_name)
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]

def plot_benchmark_results(
    acc_res,
    sta_res,
    omics_types : List[str],
    fold_to_run: Union[str, int, List[Union[str, int]]],
):
    r"""A function to plot benchmark results.

    Args:
        acc_res: accuracy results dictionary for Your Method.
        sta_res: stability results dictionary for Your Method.
        omics_types: list of omics types to plot. e.g., ['DNAm', 'mRNA', 'miRNA']
        fold_to_run: which fold(s) to plot..

    Returns:
        None. Saves the plots to ./result/benchmark_plots/

    Notes:
        format of acc_res:
        -------------------------------------------------------------------------
        {'NDCG': {('survival_BRCA', 'DNAm+mRNA+miRNA', 1): 0.20411774917314454,
        ('survival_BRCA', 'DNAm+mRNA+miRNA', 3): 0.20411774917314454,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 1): 0.18157483706343258,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 3): 0.18157483706343258,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 1): 0.1744697467882827,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 3): 0.1744697467882827,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA', 1): 0.07680220713549075,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA', 3): 0.07680220713549075,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA',
        1): 0.09109841238936288,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA',
        3): 0.09109841238936288},
        'RR': {('survival_BRCA', 'DNAm+mRNA+miRNA', 1): 0.0003763643206624012,
        ('survival_BRCA', 'DNAm+mRNA+miRNA', 3): 0.017543859649122806,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 1): 0.003937007874015748,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 3): 0.000445632798573975,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 1): 0.0004127115146512588,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 3): 0.000980392156862745,
        ('drug_response_Cisplatin-BLCA',
        'DNAm+mRNA+miRNA',
        1): 0.0002881014116969173,
        ('drug_response_Cisplatin-BLCA',
        'DNAm+mRNA+miRNA',
        3): 0.0002139495079161318,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA',
        1): 5.64334085778781e-05,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA',
        3): 6.086427267194157e-05},
        'AR': {('survival_BRCA', 'DNAm+mRNA+miRNA', 1): 0.5971907993966817,
        ('survival_BRCA', 'DNAm+mRNA+miRNA', 3): 0.5151960784313725,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 1): 0.6291630520876551,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 3): 0.5654315133815756,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 1): 0.5807572783774941,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 3): 0.5490614227120606,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA', 1): 0.8683811257775755,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA', 3): 0.8227507206797148,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA',
        1): 0.26612110674327183,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA',
        3): 0.34842757786513456},
        'MW_pval': {('survival_BRCA', 'DNAm+mRNA+miRNA', 1): 0.5159606097514122,
        ('survival_BRCA', 'DNAm+mRNA+miRNA', 3): 0.5159606097514122,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 1): 0.7754760726470906,
        ('survival_LUAD', 'DNAm+mRNA+miRNA', 3): 0.7754760726470906,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 1): 0.4467144754108136,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA', 3): 0.4467144754108136,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA', 1): 0.3151266879077809,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA', 3): 0.3151266879077809,
        ('drug_response_Temozolomide-LGG', 'DNAm+mRNA+miRNA', 1): 0.3680878695998139,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA',
        3): 0.3680878695998139}}

        -------------------------------------------------------------------------
        format of sta_res:
        -------------------------------------------------------------------------
        {'Kendall_tau': {('survival_BRCA', 'DNAm+mRNA+miRNA'): 0.08035887777805328,
        ('survival_LUAD', 'DNAm+mRNA+miRNA'): 0.08939094841480255,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA'): 0.08749798685312271,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA'): 0.07893016934394836,
        ('drug_response_Temozolomide-LGG', 'DNAm+mRNA+miRNA'): 0.0846269428730011},
        'RBO': {('survival_BRCA', 'DNAm+mRNA+miRNA'): 0.0001453493544688719,
        ('survival_LUAD', 'DNAm+mRNA+miRNA'): 0.008324292002570984,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA'): 0.0002433679079226249,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA'): 7.216842441357716e-05,
        ('drug_response_Temozolomide-LGG',
        'DNAm+mRNA+miRNA'): 0.0006876153541768236},
        'PSD': {('survival_BRCA', 'DNAm+mRNA+miRNA'): 0.08404620945629279,
        ('survival_LUAD', 'DNAm+mRNA+miRNA'): 0.09486270368135183,
        ('survival_COADREAD', 'DNAm+mRNA+miRNA'): 0.12524220234275718,
        ('drug_response_Cisplatin-BLCA', 'DNAm+mRNA+miRNA'): 0.02281606797405454,
        ('drug_response_Temozolomide-LGG', 'DNAm+mRNA+miRNA'): 0.04735399735399734}}
    """
    # ============================================================================
    # Determine omics_combs_str and folds to use
    # ============================================================================
    def _canonical_omics_comb_str(omics_comb: str) -> str:
        parts = [p.strip() for p in str(omics_comb).split('+') if p.strip()]
        return '+'.join(sorted(parts))

    def _normalize_omics_combs(omics_types_input):
        """Return a list of canonical omics combination strings.

        Accepted inputs:
        - None -> all TRI_OMICS_COMBS_STR
        - ['DNAm','mRNA','miRNA'] -> single combination
        - [['DNAm','mRNA','miRNA'], ['CNV','SNV','mRNA']] -> multiple combinations
        - ['DNAm+mRNA+miRNA', 'CNV+SNV+mRNA'] -> multiple combinations
        """
        if omics_types_input is None:
            combs = list(TRI_OMICS_COMBS_STR)
        else:
            if not isinstance(omics_types_input, (list, tuple)):
                raise TypeError(
                    "omics_types must be None, a list of omics types, a list of omics-combination strings, or a list of lists of omics types"
                )
            if len(omics_types_input) == 0:
                raise ValueError("omics_types must be non-empty when provided")

            first = omics_types_input[0]
            if isinstance(first, (list, tuple, set)):
                combs = []
                for comb in omics_types_input:
                    if not isinstance(comb, (list, tuple, set)):
                        raise TypeError("omics_types mixes nested and flat formats")
                    combs.append('+'.join(sorted([str(x).strip() for x in comb])))
            else:
                if not all(isinstance(x, str) for x in omics_types_input):
                    raise TypeError("omics_types must contain strings")
                if any('+' in x for x in omics_types_input):
                    combs = [_canonical_omics_comb_str(x) for x in omics_types_input]
                else:
                    combs = ['+'.join(sorted([x.strip() for x in omics_types_input]))]

        combs = [_canonical_omics_comb_str(c) for c in combs]
        invalid = [c for c in combs if c not in TRI_OMICS_COMBS_STR]
        if invalid:
            raise ValueError(
                f"Requested omics combination(s) not found in baseline results: {invalid}. "
                f"Allowed combinations: {TRI_OMICS_COMBS_STR}"
            )
        return combs

    def _normalize_folds(fold_to_run_input):
        if fold_to_run_input is None:
            folds_out = [0, 1, 2, 3, 4]
        elif isinstance(fold_to_run_input, (int, np.integer, str)):
            folds_out = [int(fold_to_run_input)]
        else:
            folds_out = [int(f) for f in fold_to_run_input]
        folds_out = list(dict.fromkeys(folds_out))
        if not folds_out:
            raise ValueError("fold_to_run resulted in an empty fold list")
        sample_task = next(iter(TASK_SHORT.keys()))
        sample_metric = 'AR'
        available = sorted({int(k) for k in bl_acc[(sample_task, sample_metric)].keys()})
        missing = [f for f in folds_out if f not in available]
        if missing:
            raise ValueError(f"Requested fold(s) {missing} not available in baseline results. Available folds: {available}")
        return folds_out

    omics_combs_str = _normalize_omics_combs(omics_types)
    folds = _normalize_folds(fold_to_run)
    
    # ============================================================================
    # Build baseline DataFrames from bl_acc and bl_sta filtered by omics_combs and folds
    # ============================================================================
    acc_metric_map = {'AR': 'AR', 'NDCG': 'NDCG', 'RR': 'RR', 'MW_pval': 'mwtestpval_exact'}
    sta_metric_map = {'Kendall_tau': 'KendallTau', 'RBO': 'RBO', 'PSD': 'RPSD'}
    
    def build_baseline_acc_df(metric_key):
        bl_metric_key = acc_metric_map[metric_key]
        tasks = list(TASK_SHORT.keys())
        data = {}
        models = list(BASELINES)
        for task in tasks:
            per_fold = []
            for fold in folds:
                if fold not in bl_acc[(task, bl_metric_key)]:
                    raise KeyError(f"Missing fold {fold} for baseline metric={bl_metric_key}, task={task}")
                df_fold = bl_acc[(task, bl_metric_key)][fold]

                missing_models = [m for m in BASELINES if m not in df_fold.index]
                if missing_models:
                    raise ValueError(
                        f"Baseline results missing model(s) {missing_models} for metric={bl_metric_key}, task={task}, fold={fold}"
                    )

                missing_cols = [c for c in omics_combs_str if c not in df_fold.columns]
                if missing_cols:
                    raise ValueError(
                        f"Baseline results missing requested omics combination(s) {missing_cols} for metric={bl_metric_key}, task={task}, fold={fold}"
                    )

                df_sel = df_fold.reindex(index=BASELINES).loc[:, omics_combs_str]
                per_fold.append(df_sel.to_numpy(dtype=float))  # (n_models, n_omics)

            # (n_models, n_omics, n_folds) -> flatten to (n_models, n_omics*n_folds)
            stacked = np.stack(per_fold, axis=2).reshape(len(BASELINES), -1)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                with np.errstate(all='ignore'):
                    means = np.nanmean(stacked, axis=1)
            means = np.where(np.isfinite(means), means, np.nan)
            data[task] = means
        return pd.DataFrame(data, index=models)
    
    def build_baseline_sta_df(metric_key):
        bl_metric_key = sta_metric_map[metric_key]
        tasks = list(TASK_SHORT.keys())
        data = {}
        for task in tasks:
            df_sta = bl_sta[(task, bl_metric_key)]
            missing_models = [m for m in BASELINES if m not in df_sta.index]
            if missing_models:
                raise ValueError(
                    f"Baseline stability results missing model(s) {missing_models} for metric={bl_metric_key}, task={task}"
                )
            missing_cols = [c for c in omics_combs_str if c not in df_sta.columns]
            if missing_cols:
                raise ValueError(
                    f"Baseline stability results missing requested omics combination(s) {missing_cols} for metric={bl_metric_key}, task={task}"
                )
            df_sel = df_sta.reindex(index=BASELINES).loc[:, omics_combs_str]
            data[task] = df_sel.mean(axis=1).to_numpy(dtype=float)
        return pd.DataFrame(data, index=list(BASELINES))
    
    acc_ar = build_baseline_acc_df('AR')
    acc_ndcg = build_baseline_acc_df('NDCG')
    acc_rr = build_baseline_acc_df('RR')
    sta_kt = build_baseline_sta_df('Kendall_tau')
    sta_rbo = build_baseline_sta_df('RBO')
    sta_psd = build_baseline_sta_df('PSD')
    
    # ============================================================================
    # prepare Your Method results
    # ============================================================================
    # convert acc_res to DataFrame format matching baselines
    # acc_res keys are (task, omics_comb, fold), filter by omics_combs_str and folds
    your_method_acc = {}
    for metric_key, metric_data in acc_res.items():
        your_method_acc[metric_key] = {}
        for key, val in metric_data.items():
            task, omics_comb, fold = key
            omics_comb = _canonical_omics_comb_str(omics_comb)
            fold = int(fold)
            if omics_comb in omics_combs_str and fold in folds:
                if task not in your_method_acc[metric_key]:
                    your_method_acc[metric_key][task] = []
                your_method_acc[metric_key][task].append(val)
        for task in your_method_acc[metric_key]:
            your_method_acc[metric_key][task] = float(np.nanmean(np.asarray(your_method_acc[metric_key][task], dtype=float)))

    your_method_sta = {}
    for metric_key, metric_data in sta_res.items():
        your_method_sta[metric_key] = {}
        for key, val in metric_data.items():
            task, omics_comb = key
            omics_comb = _canonical_omics_comb_str(omics_comb)
            if omics_comb in omics_combs_str:
                if task not in your_method_sta[metric_key]:
                    your_method_sta[metric_key][task] = []
                your_method_sta[metric_key][task].append(val)
        for task in your_method_sta[metric_key]:
            your_method_sta[metric_key][task] = float(np.nanmean(np.asarray(your_method_sta[metric_key][task], dtype=float)))

    def add_your_method_to_df(baseline_df, your_data, task_mapping=None):
        """Add your method row to baseline DataFrame."""
        df = baseline_df.copy()
        your_row = {col: your_data.get(col, np.nan) for col in df.columns}
        df.loc['Your Method'] = pd.Series(your_row)
        return df

    def filter_to_baselines(df):
        """Filter DataFrame to only include models in BASELINES + Your Method."""
        valid_models = [m for m in ALL_MODELS if m in df.index]
        return df.loc[valid_models]

    acc_ar_combined = filter_to_baselines(add_your_method_to_df(acc_ar, your_method_acc.get('AR', {})))
    acc_ndcg_combined = filter_to_baselines(add_your_method_to_df(acc_ndcg, your_method_acc.get('NDCG', {})))
    acc_rr_combined = filter_to_baselines(add_your_method_to_df(acc_rr, your_method_acc.get('RR', {})))
    sta_kt_combined = filter_to_baselines(add_your_method_to_df(sta_kt, your_method_sta.get('Kendall_tau', {})))
    sta_rbo_combined = filter_to_baselines(add_your_method_to_df(sta_rbo, your_method_sta.get('RBO', {})))
    sta_psd_combined = filter_to_baselines(add_your_method_to_df(sta_psd, your_method_sta.get('PSD', {})))

    # ============================================================================
    # figure - overall results (averaged across tasks)
    # ============================================================================
    def plot_overall_ranked_dotplot_panel():
        """Create a 3x2 panel of Cleveland dot plots showing ranked model performance for all metrics."""
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        
        metrics = [
            (acc_ar_combined, 'Average Recall (AR)', 'AR', 0, 0),      # row 0, col 0
            (sta_kt_combined, "Kendall's τ", 'KT', 0, 1),              # row 0, col 1
            (acc_ndcg_combined, 'NDCG', 'NDCG', 1, 0),                 # row 1, col 0
            (sta_rbo_combined, 'RBO', 'RBO', 1, 1),                    # row 1, col 1
            (acc_rr_combined, 'Reciprocal Rank (RR)', 'RR', 2, 0),     # row 2, col 0
            (sta_psd_combined, 'RPSD', 'RPSD', 2, 1),                  # row 2, col 1
        ]
        
        for idx, (df, metric_name, metric_key, row_idx, col_idx) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            df = df.copy()
            df.columns = [TASK_SHORT.get(c, c) for c in df.columns]
            
            models = [m for m in df.index if m in ALL_MODELS]
            df = df.loc[models]
            
            tasks = [c for c in df.columns if c in TASK_SHORT.values()]
            df['mean'] = df[tasks].astype(float).mean(axis=1)
            df['std'] = df[tasks].astype(float).std(axis=1)
            df = df.sort_values('mean', ascending=True)
            
            n_models = len(df)
            y_pos = np.arange(n_models)
            
            colors = get_metric_colors(metric_key, n_models)[::-1]
            
            ax.errorbar(df['mean'], y_pos, xerr=df['std'], fmt='none',
                        ecolor='#cccccc', elinewidth=1, capsize=2, capthick=1, zorder=1)
            
            for i, (model, row) in enumerate(df.iterrows()):
                ax.hlines(y=i, xmin=0, xmax=row['mean'], colors='#eeeeee', lw=0.8, zorder=0)
            
            for i, (model, row) in enumerate(df.iterrows()):
                marker = '*' if model == 'Your Method' else 'o'
                size = 120 if model == 'Your Method' else 70
                edgecolor = 'black' if model == 'Your Method' else 'white'
                edgewidth = 2 if model == 'Your Method' else 1
                ax.scatter(row['mean'], i, c=[colors[i]], s=size, zorder=5,
                        edgecolors=edgecolor, linewidths=edgewidth, marker=marker)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df.index, fontsize=7)
            ax.set_xlabel(metric_name, fontweight='bold', fontsize=9)
            ax.set_xlim(0, df['mean'].max() * 1.15)
            
            ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
            ax.set_axisbelow(True)
            
            # Panel label
            ax.text(-0.10, 1.02, chr(97 + idx), transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top')
        
        plt.suptitle('Overall Benchmark Results (Averaged Across Tasks)', fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('./figures/fig_overall_results.pdf', dpi=300, facecolor='white')
        plt.show()

    plot_overall_ranked_dotplot_panel()


    # ============================================================================
    # Build DataFrames with individual values for violin plots
    # ============================================================================
    def build_baseline_acc_individual(metric_key, task_name):
        """Build DataFrame with individual values per fold/omics_comb for a specific task."""
        bl_metric_key = acc_metric_map[metric_key]
        models = None
        records = []
        for fold in folds:
            df_fold = bl_acc[(task_name, bl_metric_key)][fold]
            if models is None:
                models = df_fold.index.tolist()
            valid_cols = [c for c in omics_combs_str if c in df_fold.columns]
            for col in valid_cols:
                for model, val in zip(df_fold.index, df_fold[col].values):
                    records.append({'model': model, 'value': float(val), 'fold': fold, 'omics': col})
        return pd.DataFrame(records)
    
    def build_baseline_sta_individual(metric_key, task_name):
        """Build DataFrame with individual values per omics_comb for a specific task."""
        bl_metric_key = sta_metric_map[metric_key]
        df_sta = bl_sta[(task_name, bl_metric_key)]
        valid_cols = [c for c in omics_combs_str if c in df_sta.columns]
        records = []
        for col in valid_cols:
            for model, val in zip(df_sta.index, df_sta[col].values):
                records.append({'model': model, 'value': float(val), 'omics': col})
        return pd.DataFrame(records)
    
    def build_your_method_acc_individual(metric_key, task_name):
        """Build DataFrame with individual values for Your Method."""
        metric_data = acc_res.get(metric_key, {})
        records = []
        for key, val in metric_data.items():
            task, omics_comb, fold = key
            if task == task_name and omics_comb in omics_combs_str and fold in folds:
                records.append({'model': 'Your Method', 'value': float(val), 'fold': fold, 'omics': omics_comb})
        return pd.DataFrame(records)
    
    def build_your_method_sta_individual(metric_key, task_name):
        """Build DataFrame with individual values for Your Method."""
        metric_data = sta_res.get(metric_key, {})
        records = []
        for key, val in metric_data.items():
            task, omics_comb = key
            if task == task_name and omics_comb in omics_combs_str:
                records.append({'model': 'Your Method', 'value': float(val), 'omics': omics_comb})
        return pd.DataFrame(records)

    # ============================================================================
    # violin plot, one figure per task 
    # ============================================================================
    def plot_task_violin_comparison(task_name):
        task_col = TASK_SHORT.get(task_name, task_name)
        has_your_method_data = False
        for df in [acc_ar_combined, sta_kt_combined]:
            df_check = df.copy()
            df_check.columns = [TASK_SHORT.get(c, c) for c in df_check.columns]
            if 'Your Method' in df_check.index:
                for col in df_check.columns:
                    if task_name in col or col in task_name or col == task_col:
                        val = df_check.loc['Your Method', col]
                        if pd.notna(val):
                            has_your_method_data = True
                            break
            if has_your_method_data:
                break
        
        if not has_your_method_data:
            print(f"Skipping {task_name}: No results for Your Method")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))
        
        metrics_config = [
            ('AR', 'Average Recall (AR)', 'AR', 0, 0, 'acc'),
            ('Kendall_tau', "Kendall's τ", 'KT', 0, 1, 'sta'),
            ('NDCG', 'NDCG', 'NDCG', 1, 0, 'acc'),
            ('RBO', 'RBO', 'RBO', 1, 1, 'sta'),
            ('RR', 'Reciprocal Rank (RR)', 'RR', 2, 0, 'acc'),
            ('PSD', 'RPSD', 'RPSD', 2, 1, 'sta'),
        ]
        
        for idx, (metric_key, metric_name, color_key, row_idx, col_idx, metric_type) in enumerate(metrics_config):
            ax = axes[row_idx, col_idx]
            
            if metric_type == 'acc':
                df_baseline = build_baseline_acc_individual(metric_key, task_name)
                df_your = build_your_method_acc_individual(metric_key, task_name)
            else:
                df_baseline = build_baseline_sta_individual(metric_key, task_name)
                df_your = build_your_method_sta_individual(metric_key, task_name)
            
            df_all = pd.concat([df_baseline, df_your], ignore_index=True)
            df_all = df_all.dropna(subset=['value'])
            
            if df_all.empty:
                ax.set_visible(False)
                continue
            
            valid_models = [m for m in df_all['model'].unique() if m in ALL_MODELS]
            df_all = df_all[df_all['model'].isin(valid_models)]
            
            model_means = df_all.groupby('model')['value'].mean().sort_values(ascending=False)
            model_order = model_means.index.tolist()
            
            n_models = len(model_order)
            colors = get_metric_colors(color_key, n_models)
            
            box_data = [df_all[df_all['model'] == m]['value'].values for m in model_order]
            bp = ax.boxplot(
                box_data,
                positions=range(n_models),
                widths=0.6,
                patch_artist=True,
                showfliers=False
            )
            
            for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
                box.set_facecolor(colors[i])
                box.set_alpha(0.6)
                box.set_edgecolor('gray')
                median.set_color('black')
                median.set_linewidth(1.5)
            
            for element in ['whiskers', 'caps']:
                for item in bp[element]:
                    item.set_color('gray')
            
            for i, model in enumerate(model_order):
                model_data = df_all[df_all['model'] == model]['value'].values
                jitter = np.random.uniform(-0.15, 0.15, size=len(model_data))
                ax.scatter(
                    np.full_like(model_data, i) + jitter,
                    model_data,
                    c=[colors[i]],
                    s=30,
                    marker='o',
                    edgecolors='white',
                    linewidths=0.5,
                    zorder=5,
                    alpha=0.8
                )
            
            ax.set_ylabel(metric_name, fontweight='bold', fontsize=9)
            ax.set_xticks(range(n_models))
            xlabels = []
            for m in model_order:
                if m == 'Your Method':
                    xlabels.append('* Your Method')
                else:
                    xlabels.append(m)
            ax.set_xticklabels(xlabels, rotation=90, ha='center', fontsize=6)
            for i, label in enumerate(ax.get_xticklabels()):
                if model_order[i] == 'Your Method':
                    label.set_fontweight('bold')
                    label.set_fontsize(7)
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
            ax.set_axisbelow(True)
            
            ax.text(-0.08, 1.02, chr(97 + idx), transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top')
        
        task_display = TASK_SHORT.get(task_name, task_name)
        plt.suptitle(f'Benchmark Results - {task_display}', fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        safe_task_name = task_display.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f'./figures/fig_task_{safe_task_name}.pdf', dpi=300, facecolor='white')
        plt.show()

    TASKS = ['survival_BRCA', 'survival_LUAD', 'survival_COADREAD', 
            'drug_response_Cisplatin-BLCA', 'drug_response_Temozolomide-LGG']
    for task in TASKS:
        plot_task_violin_comparison(task)

    # ============================================================================
    # MW p-value (-log10) box plots with dots colored by omics combination
    # ============================================================================
    # OMICS_PALETTE = {
    #     'CNV+DNAm+mRNA':      '#A5BEAF',
    #     'CNV+SNV+mRNA':       '#C2A5B4',
    #     'CNV+mRNA+miRNA':     '#A9C1CE',
    #     'DNAm+SNV+mRNA':      '#9FA4A1',
    #     'DNAm+mRNA+miRNA':    '#86BFBB',
    #     'SNV+mRNA+miRNA':     '#A2A6C0',
    # }
    OMICS_PALETTE = {
        'CNV+DNAm+mRNA':      '#ffd8b1',
        'CNV+SNV+mRNA':       '#95D378',
        'CNV+mRNA+miRNA':     '#EB8888',
        'DNAm+SNV+mRNA':      '#5C90E4',
        'DNAm+mRNA+miRNA':    '#A0DBD5',
        'SNV+mRNA+miRNA':     '#C4A5CF'
    }
    OMICS_DEFAULT_COLOR = '#B0B0B0'
    
    def get_omics_color(omics_comb):
        return OMICS_PALETTE.get(omics_comb, OMICS_DEFAULT_COLOR)
    
    def build_mw_pval_individual(task_name):
        """Build DataFrame with individual MW p-values per fold/omics_comb for a specific task."""
        bl_metric_key = acc_metric_map['MW_pval']
        models = None
        records = []
        for fold in folds:
            df_fold = bl_acc[(task_name, bl_metric_key)][fold]
            if models is None:
                models = df_fold.index.tolist()
            valid_cols = [c for c in omics_combs_str if c in df_fold.columns]
            for col in valid_cols:
                for model, val in zip(df_fold.index, df_fold[col].values):
                    records.append({'model': model, 'value': float(val), 'fold': fold, 'omics': col})
        return pd.DataFrame(records)
    
    def build_your_method_mw_pval_individual(task_name):
        """Build DataFrame with individual MW p-values for Your Method."""
        metric_data = acc_res.get('MW_pval', {})
        records = []
        for key, val in metric_data.items():
            task, omics_comb, fold = key
            if task == task_name and omics_comb in omics_combs_str and fold in folds:
                records.append({'model': 'Your Method', 'value': float(val), 'fold': fold, 'omics': omics_comb})
        return pd.DataFrame(records)
    
    def plot_mw_pval_boxplots():
        """Create figure with MW p-value (-log10) box plots for tasks with Your Method results."""
        tasks_with_data = []
        for task_name in TASKS:
            df_your = build_your_method_mw_pval_individual(task_name)
            if not df_your.empty and df_your['value'].notna().any():
                tasks_with_data.append(task_name)
        
        if not tasks_with_data:
            print("No tasks with Your Method MW p-value results to plot.")
            return
        
        n_tasks = len(tasks_with_data)
        
        if n_tasks == 1:
            nrows, ncols = 1, 1
        elif n_tasks == 2:
            nrows, ncols = 1, 2
        elif n_tasks == 3:
            nrows, ncols = 1, 3
        elif n_tasks == 4:
            nrows, ncols = 2, 2
        elif n_tasks == 5:
            nrows, ncols = 2, 3
        elif n_tasks == 6:
            nrows, ncols = 2, 3
        else:
            ncols = int(np.ceil(np.sqrt(n_tasks)))
            nrows = int(np.ceil(n_tasks / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), sharey=False)
        
        if n_tasks == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()
        
        for idx in range(n_tasks, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        for task_idx, task_name in enumerate(tasks_with_data):
            ax = axes_flat[task_idx]
            
            df_baseline = build_mw_pval_individual(task_name)
            df_your = build_your_method_mw_pval_individual(task_name)
            
            df_all = pd.concat([df_baseline, df_your], ignore_index=True)
            df_all = df_all.dropna(subset=['value'])
            
            if df_all.empty:
                ax.set_visible(False)
                continue
            
            valid_models = [m for m in df_all['model'].unique() if m in ALL_MODELS]
            df_all = df_all[df_all['model'].isin(valid_models)]
            
            df_all['neg_log10_pval'] = df_all['value'].apply(
                lambda x: -np.log10(max(x, 1e-300)) if x > 0 else 300
            )
            
            model_means = df_all.groupby('model')['neg_log10_pval'].mean().sort_values(ascending=False)
            model_order = model_means.index.tolist()
            
            n_models = len(model_order)
            
            box_data = [df_all[df_all['model'] == m]['neg_log10_pval'].values for m in model_order]
            bp = ax.boxplot(
                box_data,
                positions=range(n_models),
                widths=0.6,
                patch_artist=True,
                showfliers=False
            )
            
            for box, median in zip(bp['boxes'], bp['medians']):
                box.set_facecolor('#E8E8E8')
                box.set_alpha(0.6)
                box.set_edgecolor('gray')
                median.set_color('black')
                median.set_linewidth(1.5)
            
            for element in ['whiskers', 'caps']:
                for item in bp[element]:
                    item.set_color('gray')
            
            for i, model in enumerate(model_order):
                model_df = df_all[df_all['model'] == model]
                model_data = model_df['neg_log10_pval'].values
                omics_combs = model_df['omics'].values
                jitter = np.random.uniform(-0.2, 0.2, size=len(model_data))
                
                for j, (val, omics) in enumerate(zip(model_data, omics_combs)):
                    color = get_omics_color(omics)
                    ax.scatter(
                        i + jitter[j],
                        val,
                        c=[color],
                        s=35,
                        marker='o',
                        edgecolors='white',
                        linewidths=0.5,
                        zorder=5,
                        alpha=0.85
                    )
            
            task_display = TASK_SHORT.get(task_name, task_name)
            ax.set_title(task_display, fontweight='bold', fontsize=11)
            ax.set_ylabel('-log10(MW p-value)', fontweight='bold', fontsize=9)
            ax.set_xticks(range(n_models))
            
            thresh_05 = -np.log10(0.05)
            thresh_01 = -np.log10(0.01)
            ax.axhline(y=thresh_05, color='#888888', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
            ax.axhline(y=thresh_01, color='#888888', linestyle='-.', linewidth=1, alpha=0.6, zorder=1)
            ax.text(n_models - 0.5, thresh_05, 'p=0.05', fontsize=7, color='#666666', 
                    va='bottom', ha='right', alpha=0.8)
            ax.text(n_models - 0.5, thresh_01, 'p=0.01', fontsize=7, color='#666666', 
                    va='bottom', ha='right', alpha=0.8)
            
            xlabels = []
            for m in model_order:
                if m == 'Your Method':
                    xlabels.append('* Your Method')
                else:
                    xlabels.append(m)
            ax.set_xticklabels(xlabels, rotation=90, ha='center', fontsize=7)
            for i, label in enumerate(ax.get_xticklabels()):
                if model_order[i] == 'Your Method':
                    label.set_fontweight('bold')
                    label.set_fontsize(8)
            
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
            ax.set_axisbelow(True)
            
            ax.text(-0.05, 1.02, chr(97 + task_idx), transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top')
        
        # legend
        legend_handles = []
        for omics, color in OMICS_PALETTE.items():
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color, markersize=8, label=omics))
        unlisted_omics = [o for o in omics_combs_str if o not in OMICS_PALETTE]
        if unlisted_omics:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=OMICS_DEFAULT_COLOR, markersize=8, 
                                             label='Other combinations'))
        
        fig.legend(handles=legend_handles, loc='upper center', ncol=min(len(legend_handles), 6),
                   bbox_to_anchor=(0.5, 0.02), fontsize=9, frameon=True, title='Omics Combination')
        
        plt.suptitle('Mann-Whitney U Test p-values (-log10) Across Tasks', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.savefig('./figures/fig_mw_pval_boxplots.pdf', dpi=300, facecolor='white')
        plt.show()
    
    plot_mw_pval_boxplots()
