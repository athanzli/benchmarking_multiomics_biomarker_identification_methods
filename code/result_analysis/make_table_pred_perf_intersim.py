#%%
from setups import *
import pandas as pd

#%%
res_all = pd.read_csv("../../result/pred_res_InterSIM_ref=survival_BRCA.csv", index_col=0)

res_all['strength'] = res_all['base_name'].str.split('=').str[-1].astype(float)

res_all = res_all[['model','strength','metric','value','fold']]

df = res_all

import pandas as pd
import numpy as np

def summarize_perf(df: pd.DataFrame):
    df = df.copy()
    df["strength"] = pd.to_numeric(df["strength"])
    strength_order = sorted(df["strength"].unique())
    metrics_order = ["AUPR", "AUROC"] if {"AUPR","AUROC"}.issubset(df["metric"].unique()) \
                    else sorted(df["metric"].unique())

    counts = df.groupby(["model","strength","metric"])["fold"].nunique()
    bad = counts[counts != 5]
    if not bad.empty:
        print("Warning: groups with != 5 folds:\n", bad)

    means = (
        df.groupby(["model","strength","metric"])["value"]
          .mean()
          .unstack(["strength","metric"])
    )

    stds = (
        df.groupby(["model","strength","metric"])["value"]
          .std(ddof=1)
          .unstack(["strength","metric"])
    )

    desired_cols = pd.MultiIndex.from_product([strength_order, metrics_order],
                                              names=["strength","metric"])
    means = means.reindex(columns=desired_cols)
    stds  = stds.reindex(columns=desired_cols)
    return means, stds

means, stds = summarize_perf(df)

def mean_pm_std(means: pd.DataFrame, stds: pd.DataFrame, fmt=".3f") -> pd.DataFrame:
    out = means.copy()
    for col in means.columns:
        m = means[col]
        s = stds[col]
        out[col] = [f"{mi:{fmt}} ± {si:{fmt}}" if np.isfinite(mi) and np.isfinite(si) else "" 
                    for mi, si in zip(m, s)]
    return out

means_pm = mean_pm_std(means, stds, fmt='.2f')

models = np.array(NON_CLASSICAL_MODELS)[~np.isin(NON_CLASSICAL_MODELS, ['GAUDI', 'MOFA', 'DPM', 'MCIA'])]
means_pm = means_pm.loc[models]
means_pm.index = means_pm.index.map(MODEL_NAME_CORRECTION_MAP)

means_pm.to_csv("../../result/pred_summary_intersim_mean_std.csv")
