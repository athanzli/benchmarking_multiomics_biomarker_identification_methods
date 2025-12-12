#%%
from setups import *
import pandas as pd

#%%
res_all = pd.read_csv("/home/athan.li/eval_bk/result/pred_res_TCGA.csv", index_col=0)

i = 1 # TODO index for TASKS

res_all = res_all.loc[res_all['task'].isin([TASKS[i]])]

res_all = res_all[['model','mods','metric','value','fold']]

df = res_all

def summarize_perf(df: pd.DataFrame):
    df = df.copy()
    strength_order = sorted(df["mods"].unique())
    metrics_order = ["AUPR", "AUROC"] if {"AUPR","AUROC"}.issubset(df["metric"].unique()) \
                    else sorted(df["metric"].unique())
    
    counts = df.groupby(["model","mods","metric"])["fold"].nunique()
    bad = counts[counts != 5]
    if not bad.empty:
        print("Warning: groups with != 5 folds:\n", bad)

    means = (
        df.groupby(["model","mods","metric"])["value"]
          .mean()
          .unstack(["mods","metric"])
    )

    stds = (
        df.groupby(["model","mods","metric"])["value"]
          .std(ddof=1)
          .unstack(["mods","metric"])
    )

    desired_cols = pd.MultiIndex.from_product([strength_order, metrics_order],
                                              names=["mods","metric"])
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

means_pm.to_csv(f"../../result/pred_summary_TCGA_{TASKS[i]}_mean_std.csv")
