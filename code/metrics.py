import numpy as np 
import itertools
import math
from itertools import combinations
from scipy.stats import kendalltau
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

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

def ndcg_percentile_perm(relevances, ft, n_perm=0):
    if n_perm == 0:
        return ndcg_percentile(relevances)
    else:
        ress = []
        for _ in range(n_perm):
            mask0 = (ft['score']==0).values.flatten()
            tmp = relevances[mask0].copy()
            shuffled = np.random.permutation(tmp)
            relevances[mask0] = shuffled
            ress.append(ndcg_percentile(relevances))
        return np.mean(ress)

def ndcg_percentile(relevances):
    """
    Normalized Discounted Cumulative Gain (NDCG).
    Args:
        relevance scores in the order of ranked ele.
    """
    assert sum(relevances) <= 100, "B must be less or equal to 100 for this percentile version to work."
    B = sum(relevances)
    idcg = np.sum([(1/np.log2(2)) for i in range(B)])
    # convert relevances (rankings, F ele) to percentile (F=100)
    n_bins = 100
    arr = np.asarray(relevances)
    one_positions = np.flatnonzero(arr)  # e.g. [5, 23, 57, ...]
    bin_indices = np.floor(one_positions * n_bins / len(arr)).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    return np.sum([(1/np.log2(idx+2)) for idx in bin_indices]) / idcg

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

def to_scientific(s: str, ndigits: int = 2) -> str:
    """
    """
    f = float(s)
    sci = f"{f:.{ndigits}e}"
    mantissa, exp_str = sci.split("e")
    exp = int(exp_str)
    return f"${mantissa}\\times10^{{{exp}}}$"

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

import rbo
def rbo_score(S, T, p=0.95):
    """
    """
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

# if __name__ == "__main__":
#     ranking1 = ["a", "b", "c", "d", "e"]
#     ranking2 = ["b", "a", "d", "c", "e"]
#     ranking3 = ["a", "c", "b", "e", "d"]
#     rankings = [ranking1, ranking2, ranking3]

#     avg_tau = average_kendall_tau(rankings)
#     avg_rbo = average_rbo(rankings, p=0.9)

#     print("Average Kendall's tau:", avg_tau)
#     print("Average RBO (p=0.9):", avg_rbo)


def ensure_sized_array(arr):
    """
    Ensures that the input NumPy array is sized (i.e., has at least one dimension).
    If the array is scalar (shape ()), it converts it into a one-element array.
    """
    if arr.shape == ():  # Check if the array is scalar
        return np.array([arr.item()], dtype=object)
    return arr

def jaccard_similarity(list1, list2):
    list1 = ensure_sized_array(list1)
    list2 = ensure_sized_array(list2)
    if len(list1) == 0 or len(list2) == 0:
        return 0
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def jaccard_pairwise_avg(lists):
    n = len(lists)
    pairs = list(itertools.combinations(range(n), 2))
    sim = np.zeros(len(pairs))
    for i, pair in enumerate(pairs):
        sim[i] = jaccard_similarity(lists[pair[0]], lists[pair[1]])
    return np.mean(sim)

