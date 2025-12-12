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


TASKS = ['survival_BRCA']

MODELS = MODELS[~np.isin(MODELS, CLASSICAL_MODELS)]
MODELS = MODELS[~np.isin(MODELS, MODELS_WO_PERF)]

#%% 
res_all = pd.DataFrame(
    columns = ['model','base_name','metric','fold','value']
)

for task in TASKS:
    print("==================================================")
    print(f"Running task {task} ...")
    print("==================================================")
    for i, base_name in enumerate(BASE_NAMES):
        for model in MODELS:
            for fold in FOLDS:
                try:
                    with open(RES_PATH + f"perf_{model}_InterSIM_{base_name}_fold{fold}.pkl", 'rb') as f:
                        perf = pickle.load(f)
                except:
                    print('Missing:', f"{model}_{base_name[18:]}_fold{fold}.pkl")

                res_all = pd.concat([res_all, pd.DataFrame({
                        'model': [model],
                        'base_name': [base_name],
                        'metric': ['AUPR'],
                        'fold' : [fold],
                        'value': perf['aucpr'],
                    })], ignore_index=True)
                res_all = pd.concat([res_all, pd.DataFrame({
                        'model': [model],
                        'base_name': [base_name],
                        'metric': ['AUROC'],
                        'fold' : [fold],
                        'value': perf['roc_auc'],
                    })], ignore_index=True)

    # save
    res_all.to_csv("../../result/pred_res_InterSIM_ref=survival_BRCA.csv")
