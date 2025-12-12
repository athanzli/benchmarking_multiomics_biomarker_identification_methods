#%%
import pickle
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
    'AR' : 'Average Recall',
    'NDCG' : 'NDCG',
    'RR' : 'Reciprocal Rank',
    'mwtestpval' : 'Mann-Whitney U rank-sum test'
}

#%%
##############################################################################
# bk acc, per task
##############################################################################
bk_metric_abbrev = ['AR','NDCG', 'RR', 'mwtestpval_exact']

def generate_plot(
    task = TASKS[0],
    bk_metric_idx=0
):
    ##################
    metric_name = bk_metric_abbrev[bk_metric_idx]
    with open(f"../../result/bkacc_res_{bk_metric_abbrev[bk_metric_idx]}_TCGA_{task}.pkl", 'rb') as f:
        res = pickle.load(f)
    if metric_name == 'mwtestpval_exact':
        metric_name = 'mwtestpval'
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
                        "value": -np.log10(value) if metric_name=='mwtestpval' else value,
                    })
    df = pd.DataFrame(data_fig).drop_duplicates()
    assert not df['value'].isna().any()
    ##################

    df['model'] = df['model'].map(MODEL_NAME_CORRECTION_MAP)

    MODELS = [
        'DeePathNet',
        'DeepKEGG',
        'MOGLAM',
        'TMONet',
        'CustOmics',
        'GENIUS',
        'Pathformer',
        'GNNSubNet',
        'PNet',
        'MOGONET',
        'MORE',
        'MoAGLSA',
        'DIABLO',
        'GAUDI',
        'GDF',
        'Stabl',
        'asmPLSDA',
        'MOFA',
        'DPM',
        'MCIA']

    DESIRED_MODEL_ORDER = [MODEL_NAME_CORRECTION_MAP[m] for m in MODELS]

    #
    MULTI_OMICS_DATA_TYPE_PALETTE = {'CNV+DNAm+mRNA': '#A5BEAF',
    'CNV+SNV+mRNA': '#C2A5B4',
    'CNV+mRNA+miRNA': '#A9C1CE',
    'DNAm+SNV+mRNA': '#9FA4A1',
    'DNAm+mRNA+miRNA': '#86BFBB',
    'SNV+mRNA+miRNA': '#A2A6C0'}
    palette = MULTI_OMICS_DATA_TYPE_PALETTE

    combos = list(palette.keys())
    num_label = {combo: str(i+1) for i, combo in enumerate(combos)}

    stats = (
        df
        .groupby(['model', 'mod_combo'])['value']
        .agg(mean='mean', std='std')
        .reset_index()
    )

    models = [m for m in DESIRED_MODEL_ORDER if m in stats['model'].unique()]

    n_rows, n_cols = 2, 10
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*3, n_rows*3),
        sharey=True, sharex=False,
        gridspec_kw={'hspace': 0.2, 'wspace': 0.05}) 
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        sub = stats[stats['model'] == model].reset_index(drop=True)
        cur_multi_omics_types_order = [mm for mm in np.asarray(list(MULTI_OMICS_DATA_TYPE_PALETTE.keys())) if mm in sub['mod_combo'].unique()] # reorder
        sub = (
            sub
            .set_index('mod_combo')
            .loc[cur_multi_omics_types_order] 
            .reset_index()
        )
        x = np.arange(len(sub))
        colors = [palette[m] for m in sub['mod_combo']]

        fold_df = df[df['model']==model].copy()


        cur_multi_omics_types_order = [mm for mm in list(palette) if mm in fold_df['mod_combo'].unique()]

        ### plot
        sns.stripplot(
            x='mod_combo', y='value',
            data=fold_df,
            order=cur_multi_omics_types_order,
            palette=[palette[c] for c in cur_multi_omics_types_order],
            hue='mod_combo',
            dodge=False,
            jitter=0.08, 
            alpha=1.0,
            size=5.3,
            edgecolor='auto',
            linewidth=0.5,
            ax=ax,
            legend=False
        )
        if metric_name == 'mwtestpval':
            ax.axhline(-np.log10(0.05), color='black', linestyle='-.', linewidth=1)
        else:
            sns.barplot(
                x='mod_combo', y='value',
                data=fold_df,
                order=cur_multi_omics_types_order,
                palette=[palette[c] for c in cur_multi_omics_types_order],
                hue='mod_combo', 
                ax=ax,
                legend=False,
                errorbar=None
            )
        ax.yaxis.grid(True, color='lightgrey', linestyle='--', linewidth=1)
        ax.set_axisbelow(True)

        ax.set_title(
            model,
            fontsize=11,
            x=0.5,
            y=1.02,
            pad=0,
            color='black',
        )

        ax.set_xticks(x)
        if metric_name == 'mwtestpval':
            ax.set_ylabel(r'$-\log_{10}(p)$')
        else:
            ax.set_ylabel(metric_name)
        ax.set_xticklabels([num_label[m] for m in sub['mod_combo']])
        ax.set_xlabel('')

    for ax in axes[len(models):]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(color=palette[c], label=f"{num_label[c]}: {c}")
        for c in combos
    ]
    fig.legend(
        handles=handles,
        title='Multi-omics data types',
        loc='lower right',
        bbox_to_anchor=(0.83, 0.12),
        borderaxespad=0.0,
        ncol=1
    )
    fig.suptitle(f"{METRIC_NAME_MAPPING[metric_name]} - {TASK_NAME_MAPPING[task]}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"../../figures/raw/fig_bk_acc_by_all_mods_{task}_{metric_name}.pdf", dpi=300)
    plt.show()

for task in TASKS:
    for bk_metric_idx in [0,1,2,3]:
        generate_plot(task=task, bk_metric_idx=bk_metric_idx)
