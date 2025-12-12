#%%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from setups import *

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

bk_metric = [f'RBO', 'RPSD',  'KendallTau']
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
    res_all = pd.DataFrame(index=MODELS, dtype=np.float64)
    res_all_ori = pd.DataFrame(index=MODELS, dtype=np.float64)
    for task in TASKS:
        with open(f"../../result/stability_res_{bk_metric_abbrev[bk_metric_idx]}_TCGA_{task}.pkl", 'rb') as f:
            res = pickle.load(f)

        df_mean = res.mean(axis=1).to_frame().rename(columns={0:task}).loc[MODELS]
        # for models that do not have BK present in ft gset thus nan perf
        assert (df_mean.index.isin(MODELS)).all()
        res_all = pd.concat([res_all, df_mean], axis=1)

    res_all.to_csv(f"../../result/result_TCGA_bk_stability_{bk_metric_abbrev[bk_metric_idx]}.csv")

    res_all_metrics.append(res_all)

