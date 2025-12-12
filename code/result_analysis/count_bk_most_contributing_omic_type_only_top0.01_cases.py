#%%
from setups import *
import pickle as pkl

RES_PATH = "/data/zhaohong/eval_bk/result_09052025/"
METRIC_ABVS = ['AR','NDCG','RR']
MULTIOMICS_DATA_TYPES = [
"CNV+DNAm+mRNA",
"CNV+SNV+mRNA",
"CNV+mRNA+miRNA",
"DNAm+SNV+mRNA",
"DNAm+mRNA+miRNA",
"SNV+mRNA+miRNA"
]

MODELS = list(np.asarray(NON_CLASSICAL_MODELS)[~np.isin(np.asarray(NON_CLASSICAL_MODELS), np.asarray(MODELS_WITH_BARE_GENES_FT_SCORES))])

with open("../../result/res_all_ft_scores_gene.pkl",'rb') as f:
    res = pkl.load(f)    

# %% load per bk rank results
with open(RES_PATH + "dfs_each_bk_rank.pkl", 'rb') as f:
    rank = pkl.load(f)
with open(RES_PATH + "dfs_ft_gset_sizes.pkl", 'rb') as f:
    dfs_ft_gset_size = pkl.load(f)

#%% load bk
surv_bks = pd.read_csv("/home/athan.li/eval_bk/data/bk_set/processed/survival_task_bks.csv", index_col=0)
surv_bks = surv_bks.loc[(surv_bks['Task'].isin(['survival_BRCA','survival_LUAD','survival_COADREAD'])) & (surv_bks['Level']<='B')]
drug_bks = pd.read_csv("/home/athan.li/eval_bk/data/bk_set/processed/drug_response_task_bks.csv", index_col=0)
drug_bks = drug_bks[['Task','Gene','Level']]
drug_bks = drug_bks.loc[(drug_bks['Level']<='B') & (drug_bks['Task'].isin(['drug_response_Cisplatin-BLCA','drug_response_Temozolomide-LGG']))]
bks = pd.concat([surv_bks, drug_bks], axis=0)

dfs = {mm : pd.DataFrame(
    [[{'CNV':0, 'DNAm':0, 'SNV':0, 'mRNA':0, 'miRNA':0} \
      for _ in bks['Task'] + '-' + bks['Gene']] for _ in MODELS],
    index = MODELS,
    columns = bks['Task'] + '-' + bks['Gene']
) for mm in MULTIOMICS_DATA_TYPES}

#%%
for it_cnt, (mm, fold, task, model) in enumerate(product(MULTIOMICS_DATA_TYPES, FOLDS, TASKS, MODELS)):
    print(f"\n====================={it_cnt}/2700======================")
    print(mm, fold, task, model)
    bk_set0 = bks.loc[bks['Task']==task, 'Gene'].unique()

    ###############
    ft = res[task][mm][fold][model].copy()
    ft.index = ft.index.str.split('@').str[1]
    ft = ft.groupby(ft.index, sort=False).max().sort_values(by='score', ascending=False)
    ft_gset = ft.index.values
    mask = np.isin(bk_set0, ft_gset)
    bk_set = bk_set0[mask]
    if len(bk_set) == 0: continue
    RR = rr_perm(ft=ft, bk=bk_set, n_perm=1)
    print("RR:", RR)

    ###############
    ft = res[task][mm][fold][model].copy()
    ft.index.name = 'index'
    df = ft.reset_index().rename(columns={'index':'orig'}).copy()
    df[['mod','gene']] = df['orig'].str.split('@', expand=True)

    ct = (
        df
        .pivot_table(
            index='gene',
            columns='mod',
            values='score',
            aggfunc='mean', 
            fill_value=0
        )
        .reindex(columns=mm.split('+'))
    )

    ct = ct.loc[(ct<=0).sum(axis=1)<len(mm.split('+'))]
    mods = ct.columns
    row_max = ct.max(axis=1)
    is_max = np.isclose(ct, row_max.to_numpy()[:, None], rtol=0, atol=0)
    n_max = is_max.sum(axis=1) 
    max_mods = (
        pd.DataFrame(np.where(is_max, mods.to_numpy(), None), index=ct.index)
        .apply(lambda row: [m for m in row if m is not None], axis=1)
    )
    out = pd.DataFrame({
        'max_value': row_max,
        'max_mods': max_mods,
        'num_max': n_max
    })
    print("num of max: ", (out['num_max']>1).sum())
    bk_set = bk_set[np.isin(bk_set, out.index)]
    out["max_mods"] = out["max_mods"].apply(lambda xs: "+".join(xs))
    out = out.loc[bk_set, ['max_mods']]
    
    assert out.index.isin(bk_set).all()
    # assign
    for bk in out.index:
        cell = dfs[mm].at[model, task+'-'+bk]
        cur_rank = rank.loc[model, task+'-'+bk][mm][fold]
        ft_gset_size = dfs_ft_gset_size.loc[model, task+'-'+bk][mm]
        flag = (cur_rank / ft_gset_size) <= 0.01
        ####
        for singlem in out.loc[bk].values.astype(str)[0].split('+'):
            if flag:
                cell[singlem] += 1
        dfs[mm].at[model, task+'-'+bk] = cell

# %% save dfs
with open(RES_PATH + "bk_most_contributing_omic_type_count_top0.01_cases.pkl", 'wb') as f:
    pkl.dump(dfs, f)

# %%

