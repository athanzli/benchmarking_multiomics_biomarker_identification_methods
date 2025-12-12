#%%
from setups import *
import pickle as pkl

RES_PATH = "/home/athan.li/eval_bk/result/"

MULTIOMICS_DATA_TYPES = [
"CNV+DNAm+mRNA",
"CNV+SNV+mRNA",
"CNV+mRNA+miRNA",
"DNAm+SNV+mRNA",
"DNAm+mRNA+miRNA",
"SNV+mRNA+miRNA"
]

MODELS = NON_CLASSICAL_MODELS

with open("../../result/res_all_ft_scores_gene.pkl",'rb') as f:
    res = pkl.load(f)    

#%% load bk
surv_bks = pd.read_csv("/home/athan.li/eval_bk/data/bk_set/processed/survival_task_bks.csv", index_col=0)
surv_bks = surv_bks.loc[(surv_bks['Task'].isin(['survival_BRCA','survival_LUAD','survival_COADREAD'])) & (surv_bks['Level']<='B')]
drug_bks = pd.read_csv("/home/athan.li/eval_bk/data/bk_set/processed/drug_response_task_bks.csv", index_col=0)
drug_bks = drug_bks[['Task','Gene','Level']]
drug_bks = drug_bks.loc[(drug_bks['Level']<='B') & (drug_bks['Task'].isin(['drug_response_Cisplatin-BLCA','drug_response_Temozolomide-LGG']))]
bks = pd.concat([surv_bks, drug_bks], axis=0)

# %%
df1 = pd.DataFrame(index=MODELS, columns=TASKS, dtype=np.int32, data=0)
df2 = pd.DataFrame(index=MODELS, columns=TASKS, dtype=np.int32, data=0)
df3 = pd.DataFrame(index=MODELS, columns=TASKS, dtype=np.int32, data=0)
df4 = pd.DataFrame(index=MODELS, columns=TASKS, dtype=np.int32, data=0)

#%%
for it_cnt, (mm, fold, task, model) in enumerate(product(MULTIOMICS_DATA_TYPES, FOLDS, TASKS, MODELS)):
    print(f"\n====================={it_cnt}/2900======================")
    print(mm, fold, task, model)
    bk_set0 = bks.loc[bks['Task']==task, 'Gene'].unique()

    ###############
    ft = res[task][mm][fold][model].copy()
    ft.index = ft.index.str.split('@').str[1]
    ft = ft.groupby(ft.index, sort=False).max().sort_values(by='score', ascending=False)
    ft_gset = ft.index.values # update current gene set
    mask = np.isin(bk_set0, ft_gset)
    bk_set = bk_set0[mask]

    if ft.index[:10].isin(bk_set).any():
        df1.loc[model, task] += 1
    if ft.index[:30].isin(bk_set).any():
        df2.loc[model, task] += 1
    if ft.index[:50].isin(bk_set).any():
        df3.loc[model, task] += 1
    if ft.index[:100].isin(bk_set).any():
        df4.loc[model, task] += 1

#%%     
df1.index = df1.index.map(MODEL_NAME_CORRECTION_MAP)
df1.columns = df1.columns.map(TASK_NAME_CORRECTION_MAP)
df2.index = df2.index.map(MODEL_NAME_CORRECTION_MAP)
df2.columns = df2.columns.map(TASK_NAME_CORRECTION_MAP)
df3.index = df3.index.map(MODEL_NAME_CORRECTION_MAP)
df3.columns = df3.columns.map(TASK_NAME_CORRECTION_MAP)
df4.index = df4.index.map(MODEL_NAME_CORRECTION_MAP)
df4.columns = df4.columns.map(TASK_NAME_CORRECTION_MAP)

# %%
df1['Total'] = df1.iloc[:, :5].sum(axis=1)
df2['Total'] = df2.iloc[:, :5].sum(axis=1)
df3['Total'] = df3.iloc[:, :5].sum(axis=1)
df4['Total'] = df4.iloc[:, :5].sum(axis=1)
dfs = {
    10: df1,
    30: df2,
    50: df3,
    100: df4
}
with open(RES_PATH + "result_count_topN_hits.pkl", 'wb') as f:
    pkl.dump(dfs, f)
