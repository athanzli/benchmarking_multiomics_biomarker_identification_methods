#%%
from setups import *

with open(f"../../result/res_all_ft_scores_gene.pkl", 'rb') as f:
    res_all = pickle.load(f)

#%%
# similarity res
def make_empty_df():
    return pd.DataFrame(
        index=MODELS,
        columns=MODELS,
        dtype=float
    )
res_sim_all1 = {
    task: {
        fold: {
            mods_comb_str: make_empty_df()
            for mods_comb_str in (
                MODS_COMBS_STR
            )
        }
        for fold in FOLDS
    }
    for task in TASKS
}

res_sim_all2 = {
    task: {
        fold: {
            mods_comb_str: make_empty_df()
            for mods_comb_str in (
                MODS_COMBS_STR 
            )
        }
        for fold in FOLDS
    }
    for task in TASKS
}

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

    for fold, mods_list in tqdm(product(FOLDS, mods_combs)):
        mods = '+'.join(mods_list)

        res_sim_cur_ft = {}

        for model in MODELS:
            ft = res_all[task][mods][fold][model].copy()
            
            mmdic = mod_mol_dict(ft.index)
            assert len(mmdic['mods_uni'])==3
            assert (mmdic['mols']!='').all() and (mmdic['mols'] != np.nan).all()
            
            # aggreg by gene
            ft.index = ft.index.str.split('@').str[1]
            ft = ft.groupby(ft.index).max() # average scores for genes
            ft = ft.sort_values(by='score', ascending=False)

            res_sim_cur_ft[model] = ft
            
        # similarity between 2 models based on gene ranking
        for model1 in MODELS:
            for model2 in MODELS:
                if (model1 not in res_sim_cur_ft.keys()):
                    print("POS4 missing model1:", model1)
                    continue
                if (model2 not in res_sim_cur_ft.keys()):
                    print("POS4 missing model2:", model2)
                    continue
                ranking1 = res_sim_cur_ft[model1].index.values # NOTE make sure already sorted
                ranking2 = res_sim_cur_ft[model2].index.values # NOTE make sure already sorted
                res_sim_all1[task][fold][mods].loc[model1, model2] = kendalltau_score(ranking1, ranking2)
                res_sim_all2[task][fold][mods].loc[model1, model2] = rbo_score(ranking1, ranking2, p=0.98)

with open(f"../../result/res_TCGA_model_similarity_KendallTau.pkl", 'wb') as f:
    pickle.dump(res_sim_all1, f)
with open(f"../../result/res_TCGA_model_similarity_RBO_p=0.98.pkl", 'wb') as f:
    pickle.dump(res_sim_all2, f)


# %%
