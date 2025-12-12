#%%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from setups import * # os.chdir('/home/athan.li/eval_bk/code/result_analysis/')

bk_metric = [f'AR', 'NDCG',  'RR']
bk_metric_abbrev = bk_metric.copy()

mod_combo_order = ['+'.join(mods) for mods in MODS_COMBS]

#%%
##############################################################################
# TCGA
##############################################################################
TASKS = ['survival_BRCA',
 'survival_LUAD',
 'survival_COADREAD',
 'drug_response_Cisplatin-BLCA',
 'drug_response_Temozolomide-LGG']
res_all_metrics = []

for bk_metric_idx in range(len(bk_metric)):
    print(bk_metric_abbrev[bk_metric_idx])
    res_all = pd.DataFrame(index=MODEL_ORDER)
    res_all_ori = pd.DataFrame(index=MODEL_ORDER)
    for task in TASKS:
        with open(f"../../result/bkacc_res_{bk_metric_abbrev[bk_metric_idx]}_TCGA_{task}.pkl", 'rb') as f:
            res = pickle.load(f)
        data_fig = []
        for fold, df in res.items():
            for mod_combo in df.columns:
                for model in df.index:
                    value = df.loc[model, mod_combo]
                    if pd.notnull(value):
                        data_fig.append({
                            "fold": fold,
                            "model": model,
                            "mod_combo": mod_combo,
                            "value": value,
                            "task": task
                        })
        df = pd.DataFrame(data_fig).drop_duplicates()

        df_mean = df.groupby(['model'], as_index=False)['value'].mean()
        assert (df_mean['model'].isin(MODEL_ORDER)).all()
        nan_models = set(MODEL_ORDER) - set(df_mean['model'].unique())
        if nan_models:
            j = df_mean.index.max()+1
            for nan_model in nan_models:
                df_mean.loc[j, :] = [nan_model, np.nan]
                j += 1
        cur_res = df_mean.set_index('model').loc[MODEL_ORDER].rename(columns={'value':'TCGA-'+task})
        assert (res_all.index==cur_res.index).all()
        res_all = pd.concat([res_all, cur_res], axis=1)

    res_all.to_csv(f"../../result/result_TCGA_bk_accuracy_{bk_metric_abbrev[bk_metric_idx]}.csv")

    res_all_metrics.append(res_all)

