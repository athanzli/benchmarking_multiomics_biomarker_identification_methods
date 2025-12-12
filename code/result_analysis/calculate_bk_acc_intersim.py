#%%
from setups import *

RES_PATH = f"/home/athan.li/eval_bk/result/InterSIM/" 
DATA_PATH = "/home/athan.li/eval_bk/data/synthetic/InterSIM/"
FOLDS = [0,1,2,3,4] # no folds
MODS_COMBS = [['DNAm', 'mRNA', 'protein']]
MODS_STR = ['DNAm+mRNA+protein']

BASE_NAMES = [
    f"ref=survival_BRCA_n=100_p.dmp=0.01_p.deg4p=0.1_shift=0.5",
    f"ref=survival_BRCA_n=100_p.dmp=0.01_p.deg4p=0.1_shift=1",
    f"ref=survival_BRCA_n=100_p.dmp=0.01_p.deg4p=0.1_shift=2",
    f"ref=survival_BRCA_n=100_p.dmp=0.01_p.deg4p=0.1_shift=3",
    f"ref=survival_BRCA_n=100_p.dmp=0.01_p.deg4p=0.1_shift=4",
    f"ref=survival_BRCA_n=100_p.dmp=0.01_p.deg4p=0.1_shift=5",
]

res_all = pd.DataFrame(
    columns = ['model','base_name','metric','fold','value']
)

MODELS = MODELS[~np.isin(MODELS, CLASSICAL_MODELS)]

#%% 
for i, base_name in enumerate(BASE_NAMES):
    bk = pd.read_csv(f"../../data/synthetic/InterSIM/InterSIM_{base_name}_bk.csv", index_col=0)
    mask = bk.index.str.split('@').str[0]=='mRNA'
    bk0 = bk.index[mask][bk.loc[mask].any(axis=1)].str.split('@').str[1].values.astype(str)

    for model in MODELS:
        for fold in FOLDS:
            try:
                ft = pd.read_csv(RES_PATH + f"ft_{model}_InterSIM_{base_name}_fold{fold}.csv", index_col=0)
            except:
                print('Missing:', f"{model}_{base_name[18:]}")
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
                pass
            else:
                mmdic = mod_mol_dict(ft.index)
                if model in MODELS_GENE_CENTRIC:
                    ft.index = ft.index.str.split('@').str[1]
                    ft = ft.groupby(ft.index).max() # NOTE
                else:
                    fts = []
                    fts.append(ft.loc[~np.isin(mmdic['mods'], ['DNAm', 'miRNA', 'protein'])])
                    if 'DNAm' in mmdic['mods_uni']:
                        c2g = C2G.loc[(C2G['cpg.1'].isin(mmdic['mols'][mmdic['mods']=='DNAm'])) & (C2G['gene'].isin(mmdic['mols'][mmdic['mods']=='mRNA']))].drop_duplicates().copy()
                        df_cpg = c2g.loc[c2g.index.intersection(mmdic['mols'][mmdic['mods']=='DNAm']), ['gene']]
                        cpg_ft = ft.loc[mmdic['mods']=='DNAm']
                        cpg_ft.index = mmdic['mols'][mmdic['mods']=='DNAm']
                        df_cpg['score'] = cpg_ft.loc[df_cpg.index].values.flatten()
                        df_cpg = df_cpg.groupby('gene').mean()
                        df_cpg.index = 'DNAm@' + df_cpg.index
                        fts.append(df_cpg)
                    if 'protein' in mmdic['mods_uni']:
                        p2g = P2G.loc[(P2G['AGID.1'].isin(mmdic['mols'][mmdic['mods']=='protein'])) & (P2G['gene'].isin(mmdic['mols'][mmdic['mods']=='mRNA']))].drop_duplicates().copy()
                        df_prot = p2g.loc[p2g.index.intersection(mmdic['mols'][mmdic['mods']=='protein']), ['gene']]
                        prot_ft = ft.loc[mmdic['mods']=='protein']
                        prot_ft.index = mmdic['mols'][mmdic['mods']=='protein']
                        df_prot['score'] = prot_ft.loc[df_prot.index].values.flatten()
                        df_prot = df_prot.groupby('gene').mean()
                        df_prot.index = 'protein@' + df_prot.index
                        fts.append(df_prot)
                    ft = pd.concat(fts, axis=0)
                    mmdic = mod_mol_dict(ft.index)
                    ft.index = mmdic['mols']
                    ft = ft.groupby(ft.index).max() # TODO NOTE

            ft = ft.loc[np.sort(ft.index)]
            ft_gset = np.unique(ft.index)

            ##############################################################################
            # Metrics
            ##############################################################################
            mask = np.isin(bk0, ft_gset)
            if len(set(bk0) - set(ft_gset)) == len(set(bk0)):
                print(f"+++++ No BK in ft gset! model={model}  +++++")
                continue
            bk = bk0[mask].copy()
            
            ft = ft.sort_values(by='score', ascending=False, inplace=False)
            
            # NDCG
            relevances = np.full(len(ft), 0)
            relevances[np.isin(ft.index, bk)] = 1
            res_all = pd.concat([res_all, pd.DataFrame({
                'model': [model],
                'base_name': [base_name],
                'metric': ['NDCG'],
                'fold': [fold],
                'value': ndcg_perm(relevances=relevances, ft=ft, n_perm=1)
            })], ignore_index=True)

            # accuracy@K
            res_all = pd.concat([res_all, pd.DataFrame({
                'model': [model],
                'base_name': [base_name],
                'metric': ['ACC'],
                'fold': [fold],
                'value': accuracy_top_k_perm(ft, bk, k=len(bk), n_perm=1)
            })], ignore_index=True)

            # AUROC
            res_all = pd.concat([res_all, pd.DataFrame({
                'model': [model],
                'base_name': [base_name],
                'metric': ['AUROC'],
                'fold': [fold],
                'value': auroc_perm(ft, bk, n_perm=1)
            })], ignore_index=True)

#%%
res_all.to_csv("../../result/bkacc_res_InterSIM.csv")
