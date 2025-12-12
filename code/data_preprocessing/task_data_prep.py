#%%
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import os
import seaborn as sns
from copy import deepcopy
from itertools import combinations
from functools import reduce

TCGA_DATA_PATH = '/data/zhaohong/TCGA_data/data/'
DATA_PATH = '/home/athan.li/eval_bk/data/'

s0 = pd.read_csv(TCGA_DATA_PATH + 'processed/sample_info.tsv', sep='\t', index_col = 0)

MODS = ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA']
MODS_COMB = [
    ['CNV', 'DNAm',        'mRNA'],
    ['CNV',         'SNV', 'mRNA'],
    ['CNV',                'mRNA', 'miRNA'],
    [       'DNAm', 'SNV', 'mRNA'],
    [       'DNAm',        'mRNA', 'miRNA'],
    [               'SNV', 'mRNA', 'miRNA'],
]

import sys
sys.path.append('/home/athan.li/eval_bk/code/')
from utils import modmol_gene_set_tcga, mod_mol_dict, P2G, C2G, SPLITTER


def all_more_than_one_combinations_from(projs):
    return [list(comb) for r in range(2, len(projs)+1) for comb in combinations(projs, r)]

def collapse_tcga_projects_forcoadread(df: pd.DataFrame) -> pd.DataFrame:
    assert np.isin(df['TCGA_project'].unique(), ['TCGA-COAD','TCGA-READ']).all()
    group_cols = [col for col in df.columns if col not in ['TCGA_project', 'Evidence statement']]
    def _collapse(series: pd.Series) -> str:
        projects = set(series)
        if {'TCGA-COAD', 'TCGA-READ'}.issubset(projects):
            return 'TCGA-COADREAD'
        else:
            return series.iloc[0]
    collapsed = (
        df
        .groupby(group_cols, as_index=False)
        .agg({'TCGA_project': _collapse})
    )
    return collapsed

def conti_stratified_split(data, num_bins=10, random_state=42, plot_distr=True):
    """
    Splits a 1D NumPy array into training and testing sets with similar distributions.
    Using binning.

    Args:
        data (np.ndarray): The data to split. Must be 1D.
        train_ratio (float): The ratio of training data to testing data.
        num_bins (int): The number of bins to split the data into.
    """
    data_series = pd.Series(data)
    
    try:
        labels = pd.qcut(data_series, q=num_bins, duplicates='drop').cat.codes
    except ValueError as e:
        raise ValueError()
    
    indices = np.arange(len(data))


    train_idxs = []
    test_idxs = []
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)

    train = data[train_idx]
    test = data[test_idx]

    # # NOTE uncomment for plotting
    # plt.figure(figsize=(12, 6))
    # sns.kdeplot(data, label='Original', shade=True)
    # sns.kdeplot(train, label='Train', shade=True)
    # sns.kdeplot(test, label='Test', shade=True)
    # plt.title('Distribution Comparison (KDE)')
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # comb = np.concatenate([train, test])
    # hue = ['Train'] * len(train) + ['Test'] * len(test)
    # comb = pd.DataFrame({'data': comb, 'hue': hue})
    # sns.histplot(comb, x='data', hue='hue', bins=150, multiple='layer')
    # plt.title('Distribution Comparison (Hist)')
    # plt.show()
    
    return train_idxs, test_idxs

def print_datasets_statistics():
    r"""
    Load all datasets and print their sample size per mods comb and per class.
    """
    NotImplementedError

#%%
###############################################################################
# load TCGA data
###############################################################################
data0 = [pd.read_csv(TCGA_DATA_PATH + f"processed/{mod}_mat.csv", index_col=0) for mod in MODS if mod not in ['DNAm']]
dnam_nonan = pd.read_csv(TCGA_DATA_PATH + 'processed/DNAm_mat_nonan.csv', index_col=0)
data0.insert(1, dnam_nonan)

# remove any features with at least one nan to reduce number of features mainly for DNAm
for i, x in enumerate(data0):
    print("MODS[i]", MODS[i])
    print("Before, #features", x.shape[1])
    x = x.loc[:, x.isna().sum(axis=0)==0]
    data0[i] = x
    print('After, #features', x.shape[1])
    print()

# remove CpGs w/o gene mapping info in TCGA files
for i, x in enumerate(data0):
    print("MODS[i]", MODS[i])
    print("Before, #features", x.shape[1])
    if MODS[i] == 'DNAm':
        x = x.loc[:, x.columns.isin(C2G.index)]
    elif MODS[i] == 'miRNA':
        pass # NOTE
    print('After, #features', x.shape[1])
    data0[i] = x
    print()

s0 = pd.read_csv(TCGA_DATA_PATH + 'processed/sample_info.tsv', sep='\t', index_col=0)
# correct s0 mod presence info, otherwise there are some inconsistencies.
for i, mod in enumerate(MODS):
    s0.loc[:, mod] = False
    s0.loc[data0[i].index, mod] = True

#%%
###############################################################################
# task: drug response
###############################################################################
d = pd.read_csv(TCGA_DATA_PATH + 'processed/drug_response.csv', index_col=0)
d.index.rename('id', inplace=True)
d.rename(columns={'case_submitter_id': 'case_id', 'therapeutic_agents': 'drug'}, inplace=True)
temp = d.groupby(['drug', 'response']).size().unstack().sort_values('response', ascending=False)
temp = temp.loc[(temp['non-response']>=20) & (temp['response']>=20)].astype(int) # NOTE no effect on the 4 filtering criteria
print(temp)

drug_bk = pd.read_csv(DATA_PATH + "/bk_set/processed/drug_bks.csv", index_col=0)
drug_bk.index = drug_bk['Drug']
drug_bk.index = drug_bk.index.rename('drug')
drug_ids = np.intersect1d(temp.index.astype(str), drug_bk.index.astype(str))
drug_bk = drug_bk.loc[drug_ids]

#### aggregate COAD AND READ
mask = d['project_id'].isin(['TCGA-COAD', 'TCGA-READ'])
d.loc[mask, 'project_id'] = 'TCGA-COADREAD'

mask = drug_bk['TCGA_project'].isin(['TCGA-COAD', 'TCGA-READ'])
drug_bk_coadread = collapse_tcga_projects_forcoadread(drug_bk.loc[mask].copy())
drug_bk_non_coadread = drug_bk.loc[~mask]
drug_bk = pd.concat([drug_bk_coadread, drug_bk_non_coadread], axis=0)
####

#### keep only intersected
d = d.loc[d['drug'].isin(drug_ids)]
d1 = d.loc[d['response']=='response']
d2 = d.loc[d['response']=='non-response']

common_projs = np.intersect1d(np.intersect1d(d1['project_id'].unique(), d2['project_id'].unique()), drug_bk['TCGA_project'].unique())
common_drugs = np.intersect1d(np.intersect1d(d1['drug'].unique(), d2['drug'].unique()), drug_bk['Drug'].unique())

bk_num_by_proj = drug_bk.groupby(['Drug', 'TCGA_project']).size().unstack()
bk_num_by_proj = bk_num_by_proj.loc[common_drugs, common_projs]
d = d.loc[d['drug'].isin(common_drugs) & d['project_id'].isin(common_projs)]
####

sp_num_by_proj = d.groupby(['drug', 'project_id']).size().unstack()
sp_num_by_proj = sp_num_by_proj.loc[bk_num_by_proj.index, bk_num_by_proj.columns]
sp_num_by_proj = sp_num_by_proj.T
sp_num_by_proj1 = d1.groupby(['drug', 'project_id']).size().unstack()
sp_num_by_proj1 = sp_num_by_proj1.loc[bk_num_by_proj.index, bk_num_by_proj.columns]
sp_num_by_proj1 = sp_num_by_proj1.T
sp_num_by_proj2 = d2.groupby(['drug', 'project_id']).size().unstack()
sp_num_by_proj2 = sp_num_by_proj2.loc[bk_num_by_proj.index, bk_num_by_proj.columns]
sp_num_by_proj2 = sp_num_by_proj2.T
projs = np.intersect1d(sp_num_by_proj1.index, sp_num_by_proj2.index)
drugs_inter = np.intersect1d(sp_num_by_proj1.columns, sp_num_by_proj2.columns)
bk_num_by_proj = bk_num_by_proj.T
bk_num_by_proj = bk_num_by_proj.loc[projs, drugs_inter]
sp_num_by_proj1 = sp_num_by_proj1.loc[projs, drugs_inter]
sp_num_by_proj2 = sp_num_by_proj2.loc[projs, drugs_inter]
assert (sp_num_by_proj1.columns==bk_num_by_proj.columns).all() and (sp_num_by_proj2.columns==bk_num_by_proj.columns).all()
df = bk_num_by_proj.astype(str) + '@' + sp_num_by_proj1.astype(str) + ' : ' + sp_num_by_proj2.astype(str)
df = df.replace(to_replace=r'.*nan.*', value=np.nan, regex=True)
df = df.replace(to_replace=r'\.0', value='', regex=True)

## identify single-cancer type feasible pairs (>thres_num: >thres_num)
"""
### Level <= 'B'
all those drug-singlecancer pairs >= 20 samples in the lesser class
Cisplatin - BLCA - 1@40 : 23
Fluorouracil - COADREAD - 4@72 : 29
Fluorouracil - STAD - 2@60 : 32
Gemcitabine - PAAD - 3@29 : 42	
Oxaliplatin - COADREAD - 3@52 : 20	
Temozolomide - LGG - 2@21 : 106

### Level <= 'D'
Cisplatin - BLCA - 8@40 : 23
Fluorouracil - COADREAD - 5@72 : 29
Fluorouracil - STAD - 2@60 : 32
Gemcitabine - BLCA - 2@44:30
Gemcitabine - PAAD - 7@29 : 42	
Oxaliplatin - COADREAD - 3@52 : 20	
Temozolomide - LGG - 4@21 : 106

---- if relax to >=15
Cisplatin-STAD 2@23 : 15
"""

## prep sample sheet
s = s0.copy()
pt_id = np.intersect1d(d.index, s['Case ID'].values)
s = s.loc[s['Case ID'].isin(pt_id)]

s = s.loc[s['Sample Type'].isin(['Primary Tumor', 'Metastatic', 'Additional - New Primary', 'Recurrent Tumor'])]

########
data = {MODS[i]: data0[i].copy() for i in range(len(data0))}
for i, x in enumerate(data.values()):
    x = x.loc[np.isin(x.index, s.index)]
    x.index = x.index.str.slice(0, 12) # already verified that each sample id follows the pattern: r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]$'
    # group by index and take mean
    x = x.groupby(x.index).mean()
    data[MODS[i]] = x

s.index = s['Case ID']

s = s[["Case ID", 'CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA']]

s = s.drop_duplicates()
s.index.rename('CaseID', inplace=True)
s = s.groupby('Case ID')[MODS].any()

# update intersected pts
pts = np.intersect1d(s.index, d.index.unique())
s = s.loc[pts]
d = d.loc[d.index.isin(pts)]

#%% 
decided_drug_projs_pairs = [
    "Cisplatin-BLCA",
    "Temozolomide-LGG",
    "Fluorouracil-STAD",
    "Gemcitabine-PAAD",
]
decided_mods = {
    "Cisplatin-BLCA" : ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA'],
    "Temozolomide-LGG" : ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA'],
    "Fluorouracil-STAD" : ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA'],
}

#%%
########### prep bks
bk_drug = drug_bk.loc[drug_bk['Level'] <= 'C'] # NOTE
all_bks = []
for pair in decided_drug_projs_pairs:
    tmp = bk_drug.loc[(bk_drug['Drug'] == pair.split('-')[0]) & (bk_drug['TCGA_project'] == "TCGA-"+pair.split('-')[1])]
    all_bks.append(tmp)
all_bks = pd.concat(all_bks, axis=0)
all_bks = all_bks.reset_index().drop(columns=['index'])
all_bks['Task'] = [
    'drug_response_' + dr + '-' + proj.split('-')[1] for dr, proj  in zip(all_bks['Drug'].values.flatten(), all_bks['TCGA_project'].values.flatten())
]
all_bks.to_csv("../../data/bk_set/processed/drug_response_task_bks.csv")

# all_bks = pd.read_csv("../../data/bk_set/processed/drug_response_task_bks.csv",index_col=0)

"""
tried all levels to look at. 
Gemcitabine-BLCA really does not work. The BKs have no strong evidence.

"""

#%% ################################################
## for each pair, split the data
d_prev = deepcopy(d)
s_prev = deepcopy(s)
for pair in decided_drug_projs_pairs:
    # pair = 'Gemcitabine-LUAD+LUSC+PAAD'
    drug_cur = pair.split('-')[0]
    projs = ['TCGA-' + proj for proj in pair.split('-')[1].split('+')]
    d_cur = d_prev.loc[(d_prev['drug']==drug_cur) & (d_prev['project_id'].isin(projs))].copy()
    s_cur = s_prev.loc[d_cur.index.intersection(s_prev.index)].copy()
    print()
    print("===================================================================")
    print(f"================{drug_cur}-{projs}====================")
    print("===================================================================")
    
    mods_comb = decided_mods[pair]

    s_tmp = s_cur.copy()[mods_comb]
    mods_comb_str = '+'.join(mods_comb)
    print("Current mods:", mods_comb)
    pts = s_tmp.index[s_tmp[mods_comb].sum(axis=1)==len(mods_comb)]
    pts = np.intersect1d(pts, d_cur.index)


    ########## filter pts, keep only those in data mods
    print('Before intersecting with data:')
    temp = d_cur.loc[pts, 'response'].value_counts()
    print(temp.index[0], temp.values[0])
    print(temp.index[1], temp.values[1])    

    for mod in mods_comb:
        pts = data[mod].index.intersection(pts)

    print('After intersecting with data:')
    temp = d_cur.loc[pts, 'response'].value_counts()
    print(temp.index[0], temp.values[0])
    print(temp.index[1], temp.values[1])    

    assert np.unique(pts).shape[0] == pts.shape[0]
    assert (d_cur['drug']==drug_cur).all()

    ###########################################################################
    # feature processing
    ###########################################################################
    Xs = {}
    for mod in mods_comb:
        x = data[mod].loc[pts].copy()
        x = x.loc[:, x.var() > 0]
        x = x.loc[:, x.isna().sum(axis=0)==0]
        x = x.loc[:, ~x.columns.duplicated()] # remove duplicated columns
        x.columns = [f'{mod}@{mol}' for mol in x.columns]
        Xs[mod] = x
        print()
    X = pd.concat(Xs.values(), axis=1)
    ###########################################################################
    ###########################################################################
    data_cur_mods = {
        'X': X,
        'y': d_cur.loc[pts, 'response'].values.flatten(),
    }
    y = d_cur.loc[pts, 'response'].values.flatten()
    X = pts

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_val_index, test_index) in enumerate(skf.split(X, y)):
        X_test, y_test = X[test_index], y[test_index]
        X_train_val, y_train_val = X[train_val_index], y[train_val_index]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1/8, random_state=42)
        train_index_within_trainval, val_index_within_trainval = next(sss.split(X_train_val, y_train_val))
        X_train, y_train = X_train_val[train_index_within_trainval], y_train_val[train_index_within_trainval]
        X_val, y_val = X_train_val[val_index_within_trainval], y_train_val[val_index_within_trainval]
        train_index = np.where(np.isin(X, X_train))[0] # correction
        val_index = np.where(np.isin(X, X_val))[0] # correction

        assert (len(X_train) + len(X_val) + len(X_test)) == len(X)
        splits = np.full(len(pts), 'nan')
        splits[train_index] = 'trn'
        splits[val_index] = 'val'
        splits[test_index] = 'tst'
        assert splits[splits=='nan'].shape[0] == 0
        data_cur_mods[f'fold{fold}'] = splits
    
    # save data_cur_mods
    dir_path = f"../../data/TCGA/drug_response_{pair}"
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, f"drug_response_{pair}_{mods_comb_str}.pkl"), 'wb') as f:
        pickle.dump(data_cur_mods, f)

#%% 
###############################################################################
# task: survival.
###############################################################################
surv = pd.read_csv(TCGA_DATA_PATH + 'processed/survival.csv', index_col=0)
clic = pd.read_csv(TCGA_DATA_PATH + 'processed/clinical.csv', index_col=0)
c = clic[['case_submitter_id', 'project_id']].copy().drop_duplicates()
surv['project_id'] = c.loc[surv.index, 'project_id'].values

## prep sample sheet
d = surv.copy()
d['case_id'] = d.index
s = s0.copy()
pt_id = np.intersect1d(d.index, s['Case ID'].values)
s = s.loc[s['Case ID'].isin(pt_id)]
s = s.loc[s['Sample Type'].isin(['Primary Tumor', 'Metastatic', 'Additional - New Primary', 'Recurrent Tumor', 'Primary Blood Derived Cancer - Peripheral Blood', 'Additional Metastatic'])]
s.index = s['Case ID']
s = s[["Case ID", 'miRNA', 'mRNA', 'CNV', 'DNAm', 'SNV']]
s = s.drop_duplicates()
s.index.rename('CaseID', inplace=True)
s = s.groupby('Case ID')[MODS].any()

# update intersected pts
assert (d.index==d['case_id'].values).all()
assert d['case_id'].nunique()==d.shape[0]
pts = np.intersect1d(s.index, d['case_id'].unique())
s = s.loc[pts]
d = d.loc[pts]
s['Case ID'] = s.index

# ######## choose projs. only consider pts with all mods
mods_comb=MODS
print(mods_comb)
pts = s.loc[s[mods_comb].sum(axis=1)==len(mods_comb), 'Case ID'].unique()
assert set(pts) <= set(surv.index)
tmp = d.loc[pts, ['project_id']]
projs = tmp['project_id'].unique()
print(tmp.loc[tmp['project_id'].isin(projs)].value_counts())
tmp = tmp.loc[tmp['project_id'].isin(projs)].value_counts().to_frame().reset_index()
tmp.index = tmp['project_id']
tmp.drop(columns=['project_id'], inplace=True)

bk = pd.read_csv("/home/athan.li/eval_bk/data/bk_set/processed/prog_bks.csv", index_col=0)
bk = bk.loc[bk['Level']<='B'] #  NOTE
assert (bk.loc[(bk['TCGA_project']=='TCGA-READ'), 'Gene'].unique() == bk.loc[(bk['TCGA_project']=='TCGA-COAD'), 'Gene'].unique()).all()
proj_sample_bk_count = tmp.copy()
proj_sample_bk_count['bk_count'] = np.nan
for proj in proj_sample_bk_count.index:
    proj_sample_bk_count.loc[proj, 'bk_count'] = bk.loc[bk['TCGA_project']==proj, 'Gene'].nunique()
proj_sample_bk_count.sort_values(by='count', ascending=False, inplace=True)

# calculate the left half (for higher risk group, divided by median) censor rate. 
#           the lower this left half censor rate, the better quality the data.
for proj in surv.loc[pts]['project_id'].unique():
    tmp = surv.loc[(surv['project_id']==proj) & (surv.index.isin(pts))]
    tmp = tmp.loc[tmp['T'] <= tmp['T'].median(), 'E'] # short group
    proj_sample_bk_count.loc[proj, 'censor_rate'] = 1 - (tmp.sum() / len(tmp))
proj_sample_bk_count['bk_count'] = proj_sample_bk_count['bk_count'].astype(np.int64)
print(proj_sample_bk_count)
""" all 5 mods
            count  bk_count  censor_rate
project_id                              
TCGA-BRCA     647        14     0.913580
TCGA-LGG      489         6     0.791837
TCGA-HNSC     478        12     0.404167
TCGA-THCA     475         7     0.975000
TCGA-PRAD     471         9     0.978814
TCGA-SKCM     430         8     0.502326
TCGA-LUAD     423        21     0.613208
TCGA-UCEC     381         7     0.817708
TCGA-BLCA     377         6     0.449735
TCGA-LIHC     350         5     0.588571
TCGA-STAD     345         9     0.520231
TCGA-LUSC     341        16     0.555556
TCGA-COAD     281        15     0.716312
TCGA-CESC     274         5     0.751825
TCGA-KIRC     258         9     0.666667
TCGA-KIRP     253         6     0.826772
TCGA-SARC     227         7     0.561404
TCGA-ESCA     179         8     0.544444
TCGA-PCPG     173         3     0.942529
TCGA-PAAD     166         9     0.397590
TCGA-TGCT     124         3     0.967742
TCGA-THYM     118         3     0.915254
TCGA-LAML      95        23     0.229167
TCGA-READ      90        15     0.800000
TCGA-MESO      78         4     0.051282
TCGA-ACC       75         4     0.552632
TCGA-UVM       75         4     0.578947
TCGA-KICH      64         6     0.718750
TCGA-UCS       53         3     0.222222
TCGA-CHOL      35         4     0.333333
TCGA-DLBC      34         5     0.823529
TCGA-OV         9        10     0.600000
"""



"""
delete count < 300

            count  bk_count  censor_rate
project_id                              
TCGA-BRCA     647        14     0.913580
TCGA-LGG      489         6     0.791837
TCGA-HNSC     478        12     0.404167
TCGA-THCA     475         7     0.975000
TCGA-PRAD     471         9     0.978814
TCGA-SKCM     430         8     0.502326
TCGA-LUAD     423        21     0.613208
TCGA-UCEC     381         7     0.817708
TCGA-BLCA     377         6     0.449735
TCGA-LIHC     350         5     0.588571
TCGA-STAD     345         9     0.520231
TCGA-LUSC     341        16     0.555556
TCGA-COADREAD 371        15     0.716312
"""

chosen_projs = [
"TCGA-BRCA",
"TCGA-LUAD",
"TCGA-COADREAD",
]

#%%
########### manual inspection of each bk
bk = bk.loc[bk['Level'] <= 'B'] #  NOTE

survival_task_bks = pd.DataFrame()
bk.loc[bk['TCGA_project']=='TCGA-BRCA']
"""
aftering considering CIVIC rating for each bk, no BK should be removed.

# 
CCND1. ER-pos specific.
ABCB1. HER2 and post drug specific
FCGR2B. HER2 pos and drug specific.
MKI67 ER POS only.
"""
assert all(bk.loc[bk['TCGA_project']=='TCGA-BRCA','Level']<='B')
cr_genes = bk.loc[bk['TCGA_project']=='TCGA-BRCA','Gene'].unique()
cur_proj = 'TCGA-BRCA'
st_i = len(survival_task_bks)
for i, cur_g in enumerate(cr_genes):
    cur_l = np.unique(bk.loc[(bk['TCGA_project']==cur_proj) & (bk['Gene']==cur_g), 'Level'].values)[0]
    survival_task_bks.loc[i+st_i, ['Gene','Task','Level']]=[cur_g,f'survival_{cur_proj.split('-')[1]}',cur_l]

bk.loc[bk['TCGA_project']=='TCGA-LUSC'].sort_values(by='Gene')
"""
after considering CIVIC rating for each bk still above 10 BK

Removal:
remove TP53 https://civicdb.org/links/evidence_items/2999	no prognosis correlation
remove STK11 for LUSC. statement mentions "non-squamous" but disease is marked as non small cell carcinoma which is why both LUSC and LUAD are mapped
remove RB1, low rating.
remove PIM1, low rating.
remove CBLB, low rating.
remove EZH2, result with lung cancer not significant according to paper 

#   XRCC1. therapy and stage specific.
"""
to_rmv = set([
    "TP53",
    "STK11",
    "RB1",
    "PIM1",
    "CBLB",
    "EZH2"])
cr_genes = set(bk.loc[bk['TCGA_project']=='TCGA-LUSC', 'Gene'].unique()) - to_rmv
assert (bk.loc[bk['TCGA_project']=='TCGA-LUSC','Level']<='B').all()
st_i = len(survival_task_bks)
cur_proj = 'TCGA-LUSC'
for i, cur_g in enumerate(cr_genes):
    cur_l = np.unique(bk.loc[(bk['TCGA_project']==cur_proj) & (bk['Gene']==cur_g), 'Level'].values)[0]
    survival_task_bks.loc[i+st_i, ['Gene','Task','Level']]=[cur_g,f'survival_{cur_proj.split('-')[1]}',cur_l]

bk.loc[bk['TCGA_project']=='TCGA-LUAD']
"""
Removal:
ACTA1: remove, low rating.
PIM1: remove, low rating.
RB1: remove, low rating.
CBLB: remove, low rating.
TP53: remove, https://civicdb.org/links/evidence_items/2999	  "no prognostic effects"
EZH2: remove, result with lung cancer not significant according to paper "https://www.oncotarget.com/article/6612/text/". ""
MAP2K1: remove. not studied in terms of OS, by the paper.

#  XRCC1: therapy and stage specific.

"""
to_rmv = set([
    "ACTA1",
    "PIM1",
    "RB1",
    "CBLB",
    "TP53",
    "EZH2",
    "MAP2K1"])
cr_genes = set(bk.loc[bk['TCGA_project']=='TCGA-LUAD', 'Gene'].unique()) - to_rmv
assert (bk.loc[bk['TCGA_project']=='TCGA-LUAD','Level']<='B').all() # TODO
cur_proj = 'TCGA-LUAD'
st_i = len(survival_task_bks)
for i, cur_g in enumerate(cr_genes):
    cur_l = np.unique(bk.loc[(bk['TCGA_project']==cur_proj) & (bk['Gene']==cur_g), 'Level'].values)[0]
    survival_task_bks.loc[i+st_i, ['Gene','Task','Level']]=[cur_g,f'survival_LUAD',cur_l]

bk.loc[bk['TCGA_project']=='TCGA-COAD'].sort_values(by='Gene')
"""
CD274, remove
NOTCH1, remove
MIR218-1, remove #  not in our omics gset
"""
to_rmv = set(["CD274","NOTCH1","MIR218-1"])
cr_genes = set(bk.loc[bk['TCGA_project']=='TCGA-COAD', 'Gene'].unique()) - to_rmv
assert (bk.loc[bk['TCGA_project']=='TCGA-COAD','Level']<='B').all() #  
assert (bk.loc[bk['TCGA_project']=='TCGA-READ','Level']<='B').all() # 
assert all(bk.loc[bk['TCGA_project']=='TCGA-COAD','Gene'].unique() == bk.loc[bk['TCGA_project']=='TCGA-READ','Gene'].unique())
st_i = len(survival_task_bks)
cur_proj = 'TCGA-COAD' # since COAD and READ have the same biomarkers
for i, cur_g in enumerate(cr_genes):
    cur_l = np.unique(bk.loc[(bk['TCGA_project']==cur_proj) & (bk['Gene']==cur_g), 'Level'].values)[0]
    survival_task_bks.loc[i+st_i, ['Gene','Task','Level']]=[cur_g,f'survival_COADREAD',cur_l]

""" STAD excluded due to < 10 bk.
tmp = bk.loc[bk['TCGA_project']=='TCGA-STAD']
tmp = tmp.loc[tmp['Level']<='B']
tmp['Gene'].nunique()
"""

""" PRAD excluded due to < 10 bk.
tmp = bk.loc[bk['TCGA_project']=='TCGA-PRAD']
tmp = tmp.loc[tmp['Level']<='B']
tmp['Gene'].nunique()
"""

""" HNSC excluded due to < 10 bk after inspection for each gene.
tmp = bk.loc[bk['TCGA_project']=='TCGA-HNSC']
tmp = tmp.loc[tmp['Level']<='B']
tmp['Gene'].nunique()
tmp

*CD274 is not significant for HNSC prog, according to the cited pub. (https://chatgpt.com/share/68859287-cf54-8005-9b2b-c93ae9ebb607) "No significant correlation between PD-L1 and OS was found in the other subgroup analyses" as stated by the original paper.
PTPN11 should be removed due to poor rating on CIVIC.
MDM2 should be removed due to poor rating on CIVIC.

HNSC will have 9 BKs after manual filtering.
kept (these 9 have to be kept)
CCND1
EGFR
TP53
FGFR1
EZH2
TGFA
CDKN2A
ALDH1A2
GAS6
"""
to_rmv = set(["CD274","PTPN11","MDM2"])
cr_genes = set(bk.loc[bk['TCGA_project']=='TCGA-HNSC', 'Gene'].unique()) - to_rmv
assert (bk.loc[bk['TCGA_project']=='TCGA-HNSC','Level']<='B').all()
st_i = len(survival_task_bks)
cur_proj = 'TCGA-HNSC'
for i, cur_g in enumerate(cr_genes):
    cur_l = np.unique(bk.loc[(bk['TCGA_project']==cur_proj) & (bk['Gene']==cur_g), 'Level'].values)[0]
    survival_task_bks.loc[i+st_i, ['Gene','Task','Level']]=[cur_g,f'survival_{cur_proj.split('-')[1]}',cur_l]

# for cur_proj in ["TCGA-LGG",
# "TCGA-SKCM",
# "TCGA-UCEC",
# "TCGA-BLCA",
# "TCGA-LIHC",
# "TCGA-STAD",
# "TCGA-CESC",
# "TCGA-KIRC",
# "TCGA-KIRP",
# "TCGA-SARC"]:
#     to_rmv = set() #  NOTE
#     cr_genes = set(bk.loc[bk['TCGA_project']==cur_proj, 'Gene'].unique()) - to_rmv
#     assert (bk.loc[bk['TCGA_project']==cur_proj,'Level']<='B').all()
#     st_i = len(survival_task_bks)
#     for i, cur_g in enumerate(cr_genes):
#         cur_l = np.unique(bk.loc[(bk['TCGA_project']==cur_proj) & (bk['Gene']==cur_g), 'Level'].values)[0]
#         survival_task_bks.loc[i+st_i, ['Gene','Task','Level']]=[cur_g,f'survival_{cur_proj.split('-')[1]}',cur_l]

survival_task_bks = survival_task_bks.drop_duplicates()

print(survival_task_bks['Task'].value_counts())

survival_task_bks.to_csv("/home/athan.li/eval_bk/data/bk_set/processed/survival_task_bks.csv")

#%%
#############
# current projects
s_new = s.loc[(s.index.isin(s0.loc[s0['Project ID'].isin(chosen_projs), 'Case ID'].unique())) & (s.index.isin(pts))] # pts are those with 6 mods
s_new['project_id'] = s_new.index.map(surv['project_id'])
# combine COAD and READ
assert 'TCGA-COAD' in chosen_projs and 'TGCA-READ' in chosen_projs
mask = s_new['project_id'].isin(['TCGA-COAD', 'TCGA-READ'])
s_new.loc[mask, 'project_id'] = 'TCGA-COADREAD'
mask = d['project_id'].isin(['TCGA-COAD', 'TCGA-READ'])
d.loc[mask, 'project_id'] = 'TCGA-COADREAD'
chosen_projs = np.append(chosen_projs, 'TCGA-COADREAD')
chosen_projs = np.setdiff1d(chosen_projs, ['TCGA-COAD', 'TCGA-READ'])

# w/o the subsampling
data = {MODS[i]: data0[i].copy() for i in range(len(data0))}
for i, x in enumerate(data.values()):
    x.index = x.index.str.slice(0, 12) # already verified that each sample id follows the pattern: r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]$'
    # group by index and take mean
    x = x.groupby(x.index).mean()
    data[MODS[i]] = x

#%%
## for each proj, split the data
mods_combs = MODS_COMB.copy()
mods_combs.append(MODS)
for proj in chosen_projs:
    print("===================================================")
    print("Current proj:", proj)
    dd = d.loc[d['project_id']==proj].drop_duplicates()
    dd = dd.loc[dd.index.isin(s_new.index)]
    s = s_new.copy()
    s_tmp = s.copy()
    s_tmp['case_id'] = s_tmp.index
    ss = s_tmp.loc[(s_tmp.index.isin(dd.index)) & (s_tmp['project_id']==proj)]
    ss = ss.loc[dd.index]

    for mods_comb in mods_combs:
        
        if mods_comb != MODS: continue # only focus on all mods pts
        
        s = ss.copy()[mods_comb]

        mods_comb_str = '+'.join(mods_comb)
        print("Current mods:", mods_comb)

        pts = s.index[s[mods_comb].sum(axis=1)==len(mods_comb)]
        pts = np.intersect1d(pts, dd.index)
        s = s.loc[pts]
        dd = dd.loc[pts]
        assert np.unique(pts).shape[0] == pts.shape[0]
        assert (dd['project_id']==proj).all()
        assert (s.sum(axis=1) == s.shape[1]).all()
        assert np.unique(pts).shape[0] == pts.shape[0]

        print("Number of samples after ovlp with data:", len(pts))

        # overlap with data, to remove pts not in data
        for mod in mods_comb:
            pts = data[mod].index.intersection(pts)

        print("Number of samples before ovlp with data:", len(pts))

        ###########################################################################
        # feature processing
        ###########################################################################
        print("Processing data...")
        Xs = {}
        for mod in mods_comb:
            x = data[mod].loc[pts].copy()
            x = x.loc[:, x.var() > 0]
            x = x.loc[:, x.isna().sum(axis=0)==0]
            x = x.loc[:, ~x.columns.duplicated()] # remove duplicated columns
            x.columns = [f'{mod}@{mol}' for mol in x.columns]
            Xs[mod] = x
            print()
        X = pd.concat(Xs.values(), axis=1)
        ###########################################################################
        ###########################################################################
        data_cur_mods = {
            'X': X,
            'y': dd.loc[pts, ['T', 'E']],
        }
        y = dd.loc[pts, ['T', 'E']]
        X = pts

        train_val_index_s, test_index_s = conti_stratified_split(y['T'].values.flatten(), num_bins=20, random_state=42, plot_distr=False)
        for fold, (train_val_index, test_index) in enumerate(zip(train_val_index_s, test_index_s)):
            X_test, y_test = X[test_index], y.iloc[test_index]
            X_train_val, y_train_val = X[train_val_index], y.iloc[train_val_index]
            train_index_within_trainval_s, val_index_within_trainval_s = conti_stratified_split(y_train_val['T'].values.flatten(), num_bins=20, random_state=42, plot_distr=False)
            train_index_within_trainval = train_index_within_trainval_s[0]
            val_index_within_trainval = val_index_within_trainval_s[0]
            X_train, y_train = X_train_val[train_index_within_trainval], y_train_val.iloc[train_index_within_trainval]
            X_val, y_val = X_train_val[val_index_within_trainval], y_train_val.iloc[val_index_within_trainval]
            train_index = np.where(np.isin(X, X_train))[0] # correction
            val_index = np.where(np.isin(X, X_val))[0] # correction

            assert (len(X_train) + len(X_val) + len(X_test)) == len(X)
            splits = np.full(len(pts), 'nan')
            splits[train_index] = 'trn'
            splits[val_index] = 'val'
            splits[test_index] = 'tst'
            assert splits[splits=='nan'].shape[0] == 0
            data_cur_mods[f'fold{fold}'] = splits

        # save data_cur_mods
        dir_path = f"../../data/TCGA/survival_{proj.split('-')[1]}"
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, f"survival_{proj.split('-')[1]}_{mods_comb_str}.pkl"), 'wb') as f:
            pickle.dump(data_cur_mods, f)

