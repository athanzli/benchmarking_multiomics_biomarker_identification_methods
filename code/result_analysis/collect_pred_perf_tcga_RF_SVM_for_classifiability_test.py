#%%
from setups import *

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

MODS = ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA']
MODS_COMBS = [
 ['CNV', 'DNAm', 'mRNA'],
 ['CNV', 'SNV', 'mRNA'],
 ['CNV', 'mRNA', 'miRNA'],
 ['DNAm', 'SNV', 'mRNA'],
 ['DNAm', 'mRNA', 'miRNA'],
 ['SNV', 'mRNA', 'miRNA'],
 ]

res_all = pd.DataFrame(
    columns = ['model','task','mods','metric','fold','value']
)

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

    for fold, mods_list in tqdm(product(FOLDS, mods_combs)):
        mods = '+'.join(mods_list)

        for model in MODELS:
            try:
                with open(RES_PATH + f"perf_{model}_{mods}_fold{fold}.pkl", 'rb') as f:
                    perf = pickle.load(f)
            except:
                print(f"Missing perf res: perf_{model}_{mods}_fold{fold}.pkl")
                continue
        
            res_all.loc[len(res_all), ['model','task','metric','fold','mods','value']] = [model, task, 'AUROC', fold, mods, perf['roc_auc']]
            res_all.loc[len(res_all), ['model','task','metric','fold','mods','value']] = [model, task, 'AUPR', fold, mods, perf['aucpr']]

#%%
res_all.to_csv(f"../../result/pred_res_TCGA_RF_SVM.csv")
