#%%
import numpy as np
from itertools import product
from tqdm import tqdm

import pandas as pd
import sys

sys.path.append("..")
from utils import *
from metrics import *
import pickle

from setups import *

DE_STRENGTHS = ['Weak', 'Moderate', 'Strong']
MOSIM_REFS = [
    f'Ref{i+1}' for i in range(len(TASKS))
]
INTERSIM_REFS = [
    f'Ref{i+1}' for i in range(5)
]

FOLDS = [0] # no folds

#%%
###############################################################################
# InterSIM
###############################################################################
RES_PATH = f"../../result/InterSIM/" 

effects = [0.5, 1, 2, 3, 4, 5]
total_samples=[100]
props = [0.01]

MODS_COMBS = [['DNAm', 'mRNA', 'protein']]
fold = 0
TASKS = [
    "survival_BRCA",
]

summary = pd.DataFrame(
    index = [f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}" for t,n,p,e in product(TASKS,total_samples,props,effects)],
    columns = ['#Class1', '#Class2','#DNAm', '#mRNA', '#Protein','#Biomarker','Effect','p','Reference']
)

for ref_idx, task in enumerate(TASKS):
    ref = f"Ref{ref_idx+1}"
    print("==================================================")
    print(f"Running task {task} ...")
    print("==================================================")

    for i, (total_sample, prop, effect) in enumerate(tqdm(product(total_samples,props,effects))):
        try:
            bk = pd.read_csv(f"../../data/synthetic/InterSIM/InterSIM_ref={task}_n={total_sample}_p.dmp={prop}_p.deg4p=0.1_shift={effect}_bk.csv", index_col=0)
            mask = bk.index.str.split('@').str[0]=='mRNA'
            bk = bk.index[mask][bk.loc[mask].any(axis=1)].str.split('@').str[1].values.astype(str)
            data = pd.read_csv(f"../../data/synthetic/InterSIM/InterSIM_ref={task}_n={total_sample}_p.dmp={prop}_p.deg4p=0.1_shift={effect}_data.csv", index_col=0)
        except FileNotFoundError:
            continue

        mmdic = mod_mol_dict(data.columns)
        ft_num = dict(zip(np.unique(mmdic['mods'],return_counts=True)[0], np.unique(mmdic['mods'],return_counts=True)[1]))

        t = task
        n = total_sample        
        p = prop
        e = effect
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", '#DNAm'] = ft_num['DNAm']
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", '#mRNA'] = ft_num['mRNA']
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", '#Protein'] = ft_num['protein']
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", '#Biomarker'] = len(bk)
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", '#Class1'] = np.int64(n//2)
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", '#Class2'] = np.int64(n//2)
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", 'Effect'] = e
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", 'p'] = p
        summary.loc[f"InterSIM-ref={t}-n={n}-prop={p}-effect={e}", 'Reference'] = t

#%%
summary = summary.loc[summary.isna().sum(axis=1)==0]
summary = summary.drop_duplicates()
summary = summary.reset_index().drop(columns=['index'])
summary = summary.sort_values(
    by=["Reference", "#Class1", "Effect", "p"],
    ascending=[True, True, True, True]
)

summary_intersim = summary.copy()

summary_intersim.to_csv("../../result/intersim_data_statistics_summary.csv")
