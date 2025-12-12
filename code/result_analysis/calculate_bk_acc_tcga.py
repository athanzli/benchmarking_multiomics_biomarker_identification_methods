#%%
from setups import *

#%%
with open(f"../../result/res_all_ft_scores_gene.pkl", 'rb') as f:
    res_all = pickle.load(f)

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

    if 'drug_response' in task:
        bk = pd.read_csv(DATA_PATH + "bk_set/processed/drug_response_task_bks.csv", index_col=0)
        bk = bk.loc[(bk['Task'] == task), ['Gene', 'Level']]
    elif task.split('_')[0] == 'survival':
        bk = pd.read_csv(DATA_PATH + "bk_set/processed/survival_task_bks.csv", index_col=0)
        bk = bk.loc[(bk['Task'] == task), ['Gene', 'Level']]
    bk = bk.loc[bk['Level'] <= 'B']
    bk0 = bk['Gene'].unique().copy()
    # print(bk0)

    cur_task_mods = MODS
    mods_combs = MODS_COMBS
    
    res1 = {k : pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    ) for k in FOLDS}
    res2 = {k : pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    ) for k in FOLDS}
    res3 = {k : pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    ) for k in FOLDS}
    res4 = {k : pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    ) for k in FOLDS}
    res5 = {k : pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    ) for k in FOLDS}

    n_iter = len(FOLDS) * len(MODELS) * len(mods_combs)

    for fold, mods_list in tqdm(product(FOLDS, mods_combs)):
        mods = '+'.join(mods_list)

        for model in MODELS:
            try:
                tmp = res_all[task][mods][fold][model]
            except:
                continue
            if tmp is not None:
                ft = tmp.copy()
            else:
                continue
            
            # aggreg by gene
            ft.index = ft.index.str.split('@').str[1]
            ft = ft.groupby(ft.index, sort=False).max() # max scores for genes

            if ('rank' in ft.columns) and (model in ['DIABLO','asmPLSDA']):
                try:
                    ft.drop(columns=['score'], inplace=True)
                    ft = ft.rename(columns={'rank': 'score'})
                except:
                    ft = ft.rename(columns={'rank': 'score'})
            ft = ft.sort_values(by='score', ascending=False)
            
            ##############################################################################
            # metrics
            ##############################################################################
            ft_gset = ft.index.values # update current gene set
            
            bk = bk0.copy()
            mask = np.isin(bk, ft_gset)

            if len(set(bk) - set(ft_gset)) == len(bk): # if ft gset contains no bk
                print(f"No BK present to evaluate. Model is {model}, mods is {mods_list}. Skip.")
                continue
            
            bk = bk[mask]
            
            # NDCG
            relevances = np.full(len(ft), 0)
            relevances[np.isin(ft.index, bk)] = 1
            res1[fold].loc[model, mods] = ndcg_perm(relevances=relevances, ft=ft, n_perm=1)

            # RR
            res5[fold].loc[model, mods] = rr_perm(ft=ft, bk=bk, n_perm=1)

            # AR
            res3[fold].loc[model, mods] = avg_recall_top_k_perm(ft, bk, K=int(1*len(ft_gset)), n_perm=1)

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
            res2[fold].loc[model, mods] = mw_test(
                scores=scores,
                is_target=ft_cur.index.isin(bk).astype(bool),
                exact_if_possible=True)

    # save
    with open(f"../../result/bkacc_res_NDCG_TCGA_{task}.pkl", 'wb') as f:
        pickle.dump(res1, f)
    with open(f"../../result/bkacc_res_mwtestpval_exact_TCGA_{task}.pkl", 'wb') as f:
        pickle.dump(res2, f)
    with open(f"../../result/bkacc_res_AR_TCGA_{task}.pkl", 'wb') as f:
        pickle.dump(res3, f)
    with open(f"../../result/bkacc_res_RR_TCGA_{task}.pkl", 'wb') as f:
        pickle.dump(res5, f)

# %%
