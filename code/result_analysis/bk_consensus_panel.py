#%%
import pandas as pd
import pickle
from setups import *
import matplotlib
import matplotlib.pyplot as plt
import cmocean
from mpl_toolkits.axes_grid1 import make_axes_locatable

CMAP_FOR_OMICS = {
    'CNV' : matplotlib.colors.LinearSegmentedColormap.from_list('CNV', ['#FFE6CC','#FFCA95','#FFAE5E','#FF9326','#EE7700','#B75B00','#804000']),
    'DNAm' : matplotlib.colors.LinearSegmentedColormap.from_list('DNAm', ['#E1F2D9','#C0E5AF','#9FD785','#7FCA5C','#61B33B','#4B8A2D','#34601F']),
    'SNV' : matplotlib.colors.LinearSegmentedColormap.from_list('SNV', ['#F8D3D3','#F0A4A4','#E87575','#E04545','#CC2222','#9C1A1A','#6D1212']),
    'mRNA' : matplotlib.colors.LinearSegmentedColormap.from_list('mRNA', ['#D3E1F8','#A4C1F0','#74A1E8','#4580E0','#2263CC','#1A4C9D','#12356D']),
    'miRNA' : matplotlib.colors.LinearSegmentedColormap.from_list('miRNA', ['#DAF1EF','#B2E2DD','#8AD3CB','#62C3BA','#41ADA2','#32857C','#235C57']),
}
CMAP_FOR_OMICS_R = {k: v.reversed(name=f"{k}_r") for k, v in CMAP_FOR_OMICS.items()}

TASK_NAME_CORRECTION_MAP_REVERSE = {
    v : k for k, v in TASK_NAME_CORRECTION_MAP.items()
}
MODEL_NAME_CORRECTION_MAP_REVERSE = {
    v : k for k, v in MODEL_NAME_CORRECTION_MAP.items()
}

with open("/home/athan.li/eval_bk/result/consensus_panel.pkl", 'rb') as f:
    panel = pickle.load(f)

# save to csv files
for task in TASKS:
    for mod in MODS:
        tmp = panel[task][mod].rename(columns={'p-value':'adj. p-value'}).copy()
        tmp.index = tmp.index.str.split('@').str[1]
        tmp.index.rename(mod, inplace=True)
        tmp.columns = tmp.columns.str.replace("size", "ranking size", regex=False)
        tmp.columns = [
            c if ("size" in c or "p-value" in c) else c + " 5-fold RRA ranking"
            for c in tmp.columns
        ]
        tmp.to_csv(f"~/eval_bk/result/RRA_results/{task}({mod}).csv")

import re
for task in TASKS:
    for mod in MODS:
        d = panel[task][mod]
        for s in [c for c in d.columns if c.endswith('-size')]:
            b = s[:-5]  # drop the suffix "-size"
            if b in d.columns:
                # ratio = method / method-size, but NaN where method == -1
                d[f'{b}(percentile)'] = d[b].mask(d[b].eq(-1)) / d[s]
        # store
        panel[task][mod] = d


res = {
    k : {
        kk : None for kk in MODS
    } for k in TASKS
}

# %%
MODELS = NON_CLASSICAL_MODELS
MODEL_MM_ALL = [m+'-'+mm for m in NON_CLASSICAL_MODELS for mm in MULTIOMICS_DATA_TYPES]

CHOSEN_CUTOFFS = dict(zip(
    [task + '_' + mod for task in TASKS for mod in MODS],
    [
        1e-4,
        1e-3,
        1e-6,
        1e-3,
        1e-3,
        1e-3,
        1e-5,
        1e-6,
        1e-6,
        1e-4,
        1e-4,
        1e-4,
        1e-6,
        1e-5,
        1e-4,
        1e-5,
        1e-4,
        1e-5,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4
    ]
))

for task in TASKS:
    if 'drug_response' in task:
        bk = pd.read_csv("../../data/bk_set/processed/drug_response_task_bks.csv", index_col=0)
        bk = bk.loc[(bk['Task'] == task), ['Gene', 'Level']]
    elif task.split('_')[0] == 'survival':
        bk = pd.read_csv("../../data/bk_set/processed/survival_task_bks.csv", index_col=0)
        bk = bk.loc[(bk['Task'] == task), ['Gene', 'Level']]
    bk = bk.loc[bk['Level']<='B'] # NOTE
    bk0 = bk['Gene'].unique().copy()

    """ all candidate cutoffs
    (panel[TASKS[0]][MODS[0]]['p-value'] < 1e-4).sum() # 14 
    (panel[TASKS[0]][MODS[1]]['p-value'] < 1e-3).sum() # 14
    (panel[TASKS[0]][MODS[2]]['p-value'] < 1e-6).sum() # 10. or -5 (30), -7 (6)
    (panel[TASKS[0]][MODS[3]]['p-value'] < 1e-3).sum() # 10
    (panel[TASKS[0]][MODS[4]]['p-value'] < 1e-3).sum() # 10
    (panel[TASKS[1]][MODS[0]]['p-value'] < 1e-3).sum() # 21. or -4 (4) ######
    (panel[TASKS[1]][MODS[1]]['p-value'] < 1e-5).sum() # 23. or -6 (4), -7 (2)
    (panel[TASKS[1]][MODS[2]]['p-value'] < 1e-6).sum() # 12. or -5 (29), -7 (7), -8 (3)
    (panel[TASKS[1]][MODS[3]]['p-value'] < 1e-6).sum() # 5. or -7 (2)
    (panel[TASKS[1]][MODS[4]]['p-value'] < 1e-4).sum() # 7. or -3 (34)
    (panel[TASKS[2]][MODS[0]]['p-value'] < 1e-4).sum() # 18 ######
    (panel[TASKS[2]][MODS[1]]['p-value'] < 1e-4).sum() # 21
    (panel[TASKS[2]][MODS[2]]['p-value'] < 1e-6).sum() # 19. or -7 (5)
    (panel[TASKS[2]][MODS[3]]['p-value'] < 1e-5).sum() # 10. or -6 (6)
    (panel[TASKS[2]][MODS[4]]['p-value'] < 1e-4).sum() # 9
    (panel[TASKS[3]][MODS[0]]['p-value'] < 1e-5).sum() # 4 ######
    (panel[TASKS[3]][MODS[1]]['p-value'] < 1e-4).sum() # 7
    (panel[TASKS[3]][MODS[2]]['p-value'] < 1e-5).sum() # 14. or -6 (6)
    (panel[TASKS[3]][MODS[3]]['p-value'] < 1e-4).sum() # 3
    (panel[TASKS[3]][MODS[4]]['p-value'] < 1e-4).sum() # 17. or -5 (6) 
    (panel[TASKS[4]][MODS[0]]['p-value'] < 1e-4).sum() # 6 ######
    (panel[TASKS[4]][MODS[1]]['p-value'] < 1e-4).sum() # 12
    (panel[TASKS[4]][MODS[2]]['p-value'] < 1e-4).sum() # 5. or -4 (11), -7 (4), -8 (3)
    (panel[TASKS[4]][MODS[3]]['p-value'] < 1e-4).sum() # 8
    (panel[TASKS[4]][MODS[4]]['p-value'] < 1e-4).sum() # 13. or -5 (3)
    """
    for mod in MODS:
        d = panel[task][mod].loc[panel[task][mod]['p-value']<CHOSEN_CUTOFFS[task + '_' + mod]].copy()
        # print(len(d))
        mols = d.index.str.split('@').str[1].values.astype(str)
        res[task][mod] = mols.tolist()
        print("=============================================")
        print(f"Task: {task}, Modality: {mod}\n")
        for mol in mols:
            if mol in bk0:
                print(f"**{mol}**, ", end='')
            else:
                print(f"{mol}, ", end='')
        print("=============================================")
        print()
