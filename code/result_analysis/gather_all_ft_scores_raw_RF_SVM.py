# for TCGA res
#%%
from setups import *
from collections import defaultdict

def to_regular_dict(d):
    if isinstance(d, dict):
        return {k: to_regular_dict(v) for k, v in d.items()}
    return d

res_all = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(dict)
    )
)

MODS_COMBS = [
 ['CNV', 'DNAm', 'mRNA'],
 ['CNV', 'SNV', 'mRNA'],
 ['CNV', 'mRNA', 'miRNA'],
 ['DNAm', 'SNV', 'mRNA'],
 ['DNAm', 'mRNA', 'miRNA'],
 ['SNV', 'mRNA', 'miRNA'],
 ]

MODELS = ['RF_VI','SVM_ONE']
TASKS = [
    "survival_BRCA",
    "survival_LUAD",
    "survival_LUSC",
    "survival_COADREAD",
    "drug_response_Cisplatin-BLCA",
    "drug_response_Temozolomide-LGG",
    "drug_response_Fluorouracil-STAD",
    "drug_response_Gemcitabine-PAAD",
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

    for i, mods_list in enumerate(tqdm(MODS_COMBS)):
        mods = '+'.join(mods_list)
        for model in (MODELS):
            for fold in FOLDS:
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
                ft = ft.loc[np.sort(ft.index)]
                res_all[task][mods][fold][model] = ft

# %%
res_all = to_regular_dict(res_all)
with open('../../result/res_all_ft_scores_raw_RF_SVM.pkl', 'wb') as f:
    pickle.dump(res_all, f)

# %%
