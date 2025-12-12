#%%
from setups import *

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

    if 'drug_response' in task:
        bk = pd.read_csv(DATA_PATH + "bk_set/processed/drug_response_task_bks.csv", index_col=0)
        bk = bk.loc[(bk['Task'] == task), ['Gene', 'Level']]
    elif task.split('_')[0] == 'survival':
        bk = pd.read_csv(DATA_PATH + "bk_set/processed/survival_task_bks.csv", index_col=0)
        bk = bk.loc[(bk['Task'] == task), ['Gene', 'Level']]
    bk = bk.loc[bk['Level'] <= 'B']
    bk = bk['Gene'].unique()
    # print(bk)

    if 'survival' in task:
        RES_PATH = f"/home/athan.li/eval_bk/result/{task}" + '_binary/'
    else:
        RES_PATH = f"/home/athan.li/eval_bk/result/{task}/" 

    cur_task_mods = MODS
    mods_combs = MODS_COMBS
    
    with open(DATA_PATH + f"TCGA/{task}/{task}_{'+'.join(cur_task_mods)}.pkl", 'rb') as f:
        data = pickle.load(f)
    mdic = mod_mol_dict(data['X'].columns)
    
    #
    res1 = pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    )
    res2 = pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    )
    res3 = pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    )
    res4 = pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    )
    res5 = pd.DataFrame(
        index = MODELS,
        columns = ['+'.join(mods) for mods in mods_combs]
    )

    n_iter = len(FOLDS) * len(MODELS) * len(mods_combs)

    for i, (model, mods_list, fold) in enumerate(tqdm(product(MODELS, mods_combs, FOLDS))):
        mods = '+'.join(mods_list)
        if fold == 0:
            rankings = []
        ft = res_all[task][mods][fold][model].copy()

        mmdic = mod_mol_dict(ft.index)
        assert (mmdic['mols']!='').all() and (mmdic['mols'] != np.nan).all()
        ft.index = ft.index.str.split('@').str[1]
        ft = ft.groupby(ft.index).max() # average scores for genes
        ft = ft.sort_values(by='score', ascending=False)

        # permute zero scores
        ranking = ft.index.values.astype(str)
        mask0 = ft.values.flatten()==0
        ranking[mask0] = np.random.permutation(ranking[mask0])

        rankings.append(ranking)

        if fold == len(FOLDS) - 1:
            # assert np.all([(len(rankings[0]) == len(rankings[i])) for i in range(len(rankings))])
            # check bk presence
            gs = np.unique(ft.index.values)
            if set(bk) - set(gs):
                print(f"{len(set(bk)-set(gs))} out of {len(bk)} BKs not in ft gene set.", set(bk)-set(gs), f"model={model},mods={mods_list}")
            ##############################################################################
            # Metrics
            ##############################################################################            
            res1.loc[model, mods] = average_kendall_tau(rankings)
            res4.loc[model, mods] = average_rbo(rankings, p=0.98)
            res5.loc[model, mods] = percentile_standard_deviation(rankings, bk)
    
    # save
    with open(f"../../result/stability_res_KendallTau_TCGA_{task}.pkl", 'wb') as f:
        pickle.dump(res1, f)
    with open(f"../../result/stability_res_RBO_TCGA_{task}.pkl", 'wb') as f: # NOTE 0.98
        pickle.dump(res4, f)
    with open(f"../../result/stability_res_RPSD_TCGA_{task}.pkl", 'wb') as f:
        pickle.dump(res5, f)

# %%
