"""
gather all feature scores on gene level, by mods

for subsequent metric calculations and analysis

"""

#%%
from setups import *

MODELS = ['RF_VI','SVM_ONE']
TASKS = [
    "survival_BRCA",
    "survival_LUAD",
    "survival_LUSC",
    "survival_COADREAD",
    "survival_HNSC",
    "drug_response_Cisplatin-BLCA",
    "drug_response_Temozolomide-LGG",
    "drug_response_Fluorouracil-STAD",
    "drug_response_Gemcitabine-PAAD",
]

from collections import defaultdict

res_all = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(dict)
    )
)
def to_regular_dict(d):
    if isinstance(d, dict):
        return {k: to_regular_dict(v) for k, v in d.items()}
    return d

MODS_COMBS = [
 ['CNV', 'DNAm', 'mRNA'],
 ['CNV', 'SNV', 'mRNA'],
 ['CNV', 'mRNA', 'miRNA'],
 ['DNAm', 'SNV', 'mRNA'],
 ['DNAm', 'mRNA', 'miRNA'],
 ['SNV', 'mRNA', 'miRNA'],
 ]

#%%
for task in TASKS:
    
    print("==================================================")
    print(f"Running task {task} ...")
    print("==================================================")

    if 'survival' in task:
        RES_PATH = f"/home/athan.li/eval_bk/result/{task}" + '_binary/'
    else:
        RES_PATH = f"/home/athan.li/eval_bk/result/{task}/" 

    cur_task_mods = MODS
    mods_combs = MODS_COMBS

    n_iter = len(FOLDS) * len(MODELS) * len(mods_combs)

    for cnt, (fold, mods_list) in enumerate(tqdm(product(FOLDS, mods_combs))):
        mods = '+'.join(mods_list)

        for model in MODELS:
            try:
                ft = pd.read_csv(RES_PATH + f"ft_{model}_{mods}_fold{fold}.csv", index_col=0)
            except:
                print(f"Missing: {RES_PATH + f'ft_{model}_{mods}_fold{fold}.csv'}")
                continue

            if model in MODELS_RES_STORED_AS_RANKS: # output ranks
                ft = len(ft) - ft + 1 # reverse
            elif model in MODELS_OUTPUT_ONE_MINUS_P_VALUES:
                ft = 1.0 - ft.astype(np.float64)
                ft.loc[ft.values.flatten()==0.0, :] = 1e-100
                assert (ft.shape[1]==1) and ('score' in ft.columns)
                ft['score'] = -np.log10(ft.values.flatten())
            elif model in MODELS_OUTPUT_P_VALUES:
                ft = ft.astype(np.float64)
                ft.loc[ft.values.flatten()==0.0, :] = 1e-100
                assert (ft.shape[1]==1) and ('score' in ft.columns)
                ft['score'] = -np.log10(ft.values.flatten())
            elif model in MODELS_KEEPING_NEG_SCORES:
                ft = ft.mean(axis=1).sort_values(ascending=False).to_frame()
            else:
                ft = ft.abs().mean(axis=1).sort_values(ascending=False).to_frame()
            if 0 in ft.columns: ft = ft.rename(columns={0: 'score'})
            if ('rank' in ft.columns) and (model in ['DIABLO','asmPLSDA']):
                try:
                    ft.drop(columns=['score'], inplace=True)
                    ft = ft.rename(columns={'rank': 'score'})
                except:
                    ft = ft.rename(columns={'rank': 'score'})
            
            ft = ft.loc[~(ft.isna().values.flatten()), :]

            ##
            if model in MODELS_WITH_BARE_GENES_FT_SCORES:
                ft = pd.DataFrame(
                    index = np.concatenate([
                        [f"{mod}@{mol}" for mol in ft.index] for mod in mods_list
                    ]),
                    data = list(ft.values.flatten())*len(mods_list),
                    columns=['score']
                )
            else:
                if model not in MODELS_GENE_CENTRIC: # otherwise pass
                    mmdic = mod_mol_dict(ft.index)
                    fts = []
                    fts.append(ft.loc[~np.isin(mmdic['mods'], ['DNAm', 'miRNA', 'protein'])])
                    if 'miRNA' in mmdic['mods_uni']:
                        # assert ~R2G[['miRNA', 'gene']].duplicated().any()
                        df_mirna = R2G.loc[np.intersect1d(mmdic['mols'][mmdic['mods']=='miRNA'], R2G.index), ['gene']]
                        mirna_ft = ft.loc[mmdic['mods']=='miRNA']
                        mirna_ft.index = mmdic['mols'][mmdic['mods']=='miRNA']
                        df_mirna['score'] = mirna_ft.loc[df_mirna.index].values.flatten()
                        df_mirna = df_mirna.groupby('gene').mean() #NOTE
                        df_mirna.index = 'miRNA@' + df_mirna.index
                        fts.append(df_mirna)
                    if 'DNAm' in mmdic['mods_uni']:
                        # assert ~C2G[['cpg.1','gene']].duplicated().any()
                        df_cpg = C2G.loc[mmdic['mols'][mmdic['mods']=='DNAm'], ['gene']]
                        cpg_ft = ft.loc[mmdic['mods']=='DNAm']
                        cpg_ft.index = mmdic['mols'][mmdic['mods']=='DNAm']
                        df_cpg['score'] = cpg_ft.loc[df_cpg.index].values.flatten()
                        df_cpg = df_cpg.groupby('gene').mean() #NOTE
                        df_cpg.index = 'DNAm@' + df_cpg.index
                        fts.append(df_cpg)
                    if 'protein' in mmdic['mods_uni']:
                        # assert ~P2G[['AGID.1','gene']].duplicated().any()
                        df_prot = P2G.loc[mmdic['mols'][mmdic['mods']=='protein'], ['gene']]
                        prot_ft = ft.loc[mmdic['mods']=='protein']
                        prot_ft.index = mmdic['mols'][mmdic['mods']=='protein']
                        df_prot['score'] = prot_ft.loc[df_prot.index].values.flatten()
                        df_prot = df_prot.groupby('gene').mean() #NOTE
                        df_prot.index = 'protein@' + df_prot.index
                        fts.append(df_prot)
                    ft = pd.concat(fts, axis=0)

            ft = ft.loc[np.sort(ft.index)].astype(float)

            res_all[task][mods][fold][model] = ft

#%%
res_all=to_regular_dict(res_all)
with open(f"../../result/res_all_ft_scores_gene_RF_SVM.pkl", 'wb') as f:
    pickle.dump(res_all, f)
