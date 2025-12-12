#%%
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from setups import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

TASK_NAME_MAPPING = {
 'survival_BRCA' : 'Survival BRCA',
 'survival_LUAD' : 'Survival LUAD',
 'survival_COADREAD' : 'Survival COADREAD',
 'drug_response_Cisplatin-BLCA' : 'Drug response Cisplatin (BLCA)',
 'drug_response_Temozolomide-LGG' : 'Drug response Temozolomide (LGG)',
}
METRIC_NAME_MAPPING = {
    'RBO' : 'Rank-biased Overlap',
    'RPSD' : 'RPSD',
    'KendallTau' : "Kendall's tau",
}

#%%
##############################################################################
# bk acc, per task
##############################################################################
bk_metric_abbrev = ['RBO','RPSD', 'KendallTau']

MODELS = [
'MOGLAM',
'Pathformer',
'DeePathNet',
'DeepKEGG',
'TMONet',
'CustOmics',
'PNet',
'GENIUS',
'MORE',
'GNNSubNet',
'MoAGLSA',
'MOGONET',
'DIABLO', 'MCIA', 'MOFA', 'Stabl',
'GAUDI', 'asmPLSDA', 'DPM', 'GDF']

def generate_plot(
    task = TASKS[0],
    bk_metric_idx=0
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches
    import pickle as pkl

    metric_name = bk_metric_abbrev[bk_metric_idx]

    with open(f"/home/athan.li/eval_bk/result/stability_res_{bk_metric_abbrev[bk_metric_idx]}_TCGA_{task}.pkl", "rb") as f:
        res = pkl.load(f)

    MULTI_OMICS_DATA_TYPE_PALETTE = {
        'CNV+DNAm+mRNA':  '#A5BEAF',
        'CNV+SNV+mRNA':   '#C2A5B4',
        'CNV+mRNA+miRNA': '#A9C1CE',
        'DNAm+SNV+mRNA':  '#9FA4A1',
        'DNAm+mRNA+miRNA':'#86BFBB',
        'SNV+mRNA+miRNA': '#A2A6C0'
    }
    palette = MULTI_OMICS_DATA_TYPE_PALETTE

    combos_all = [c for c in palette.keys() if c in res.columns]
    if not combos_all:
        combos_all = list(res.columns) 

    num_label = {combo: str(i+1) for i, combo in enumerate(combos_all)}

    res.index = res.index.map(MODEL_NAME_CORRECTION_MAP)
    DESIRED_MODEL_ORDER = [MODEL_NAME_CORRECTION_MAP[m] for m in MODELS]
    models = [m for m in DESIRED_MODEL_ORDER if m in res.index]

    if (bk_metric_idx==1) and ('drug' in task):
        models = list(np.array(models)[np.array(models)!='DeePathNet'])

    n_rows, n_cols = 2, 10
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*3, n_rows*3),
        sharey=True, sharex=False,
        gridspec_kw={'hspace': 0.2, 'wspace': 0.05}
    )
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        s = pd.Series(res.loc[model, combos_all]).astype(float).dropna()

        combos_present = [c for c in combos_all if c in s.index]
        if len(combos_present) == 0:
            ax.set_visible(False)
            continue

        sub_df = pd.DataFrame({'mod_combo': combos_present,
                               'value': s.loc[combos_present].values})

        sns.barplot(
            x='mod_combo', y='value',
            data=sub_df,
            order=combos_present,
            palette=[palette.get(c, '#999999') for c in combos_present],
            ax=ax,
            edgecolor='black',
            linewidth=0.4,
            errorbar=None
        )

        ax.yaxis.grid(True, color='lightgrey', linestyle='--', linewidth=1)
        ax.set_axisbelow(True)

        ax.set_title(model, fontsize=11, x=0.5, y=1.02, pad=0, color='black')
        ax.set_xlabel('')

        ax.set_xticks(range(len(combos_present)))
        ax.set_xticklabels([num_label[c] for c in combos_present])
        ax.set_ylabel(METRIC_NAME_MAPPING[metric_name])

    vals = res.loc[models, combos_all].astype(float).to_numpy()
    ymax = np.nanmax(vals)
    span = ymax if np.isfinite(ymax) else 1.0
    pad  = max(0.02, 0.05 * span) 
    lower = 0.0

    for ax in axes:
        ax.set_ylim(lower, ymax + pad)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.suptitle(f"{METRIC_NAME_MAPPING[metric_name]} - {TASK_NAME_MAPPING[task]}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"../../figures/raw/fig_bk_stability_by_all_mods_{task}_{metric_name}.pdf", dpi=300)
    plt.show()

for task in TASKS:
    for bk_metric_idx in [0,1,2]:
        generate_plot(task=task, bk_metric_idx=bk_metric_idx)
