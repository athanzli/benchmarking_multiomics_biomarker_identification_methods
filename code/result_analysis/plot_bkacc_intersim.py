#%%
import os
os.chdir("/home/athan.li/eval_bk/code/result_analysis/")
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from setups import *
from adjustText import adjust_text
import cmocean.cm as cmo
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

dl_models = [MODEL_NAME_CORRECTION_MAP[m] for m in DL_MODELS]

model2color = dict(zip(
    [
 'DeePathNet',
 'DeepKEGG',
 'MOGLAM',
 'TMO-Net',
 'CustOmics',
 'GENIUS',
 'Pathformer',
 'GNN-SubNet',
 'P-Net',
 'MOGONET',
 'MORE',
 'MoAGL-SA',
 'DIABLO',
 'GAUDI',
 'GDF',
 'Stabl',
 'asmbPLS-DA',
 'MOFA',
 'DPM',
 'MCIA'
 ], ['#ffffff'] * 20
))
##############################################################################
# prep
##############################################################################
d = pd.read_csv('../../result/bkacc_res_InterSIM.csv', index_col=0)

pattern = (
    r'n=(?P<n>\d+)'
    r'_p\.dmp=(?P<p>[0-9.]+)'
    r'.*_shift=(?P<s>[0-9.]+)'
)

d[['n','p','s']] = d['base_name'].str.extract(pattern)

d = d.astype({'n': int, 'p': float, 's': float, 'value': float, 'fold':int, 'model':str})

d = d.drop(columns=['base_name'])

d['model'] = d['model'].map(MODEL_NAME_CORRECTION_MAP)

def plot_cleveland_dot(d, metric, by, save=None):
    """
    """
    mask = d['metric'] == metric

    if 'pval' in metric:
        d.loc[mask, 'value'] = -np.log10(d.loc[mask, 'value'])

    if by == 'n':
        mask = mask & (d['p']==0.01) & (d['s']==0.05)
    elif by == 'p':
        mask = mask & (d['n']==100) & (d['s']==0.05)
    elif by == 's':
        mask = mask & (d['n']==100) & (d['p']==0.01)
    else:
        raise ValueError

    df = d.loc[mask, ['model', 'fold', 'value', by]]

    assert not df.empty

    # Aggregate mean across folds
    df_agg = df.groupby(['model', by])['value'].mean().reset_index()
    # model order
    full_order = list(model2color.keys())
    present = df_agg['model'].unique().tolist()
    model_order = [m for m in full_order if m in present]
    df_agg['model'] = pd.Categorical(df_agg['model'],
                                     categories=model_order,
                                     ordered=True)

    my_colors  = [

    "#d2d26b", "#fed976", "#feb24c", "#dc6744", "#f02d0f", "#7f0000"

    ]
    levels    = sorted(df_agg[by].unique()) 
    palette = dict(zip(levels, my_colors[:len(my_colors)]))

    plt.figure(figsize=(7, len(model_order) * 0.3))

    ax = sns.pointplot(
        data=df,
        x='value', y='model', hue=by,
        estimator=np.mean, 
        # errorbar=('ci', 95),    # 95% CI; use "sd" or ("se", 1) if preferred
        errorbar='sd', 
        dodge=0.5, join=False, markers='o', scale=0.9,
        palette=palette,
        order=model_order
    )

    for y in np.arange(len(model_order)+1) - 0.5:
        ax.axhline(y=y, color='grey', linewidth=0.7, alpha=0.7, zorder=0)


    ax.xaxis.set_major_locator(MultipleLocator(0.1))

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{metric}")
    ax.legend(title='δ', ncol=1, bbox_to_anchor=(1.25,1.0), loc='upper right')
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300)

    plt.show()
    
# %%
##############################################################################
# 
##############################################################################
res = d.copy()

plot_cleveland_dot(d=res, metric='AUROC', by='s', save='../../figures/raw/bkacc_AUROC_intersim_cleveland_dot_plot.pdf')
plot_cleveland_dot(d=res, metric='ACC', by='s', save='../../figures/raw/bkacc_ACC_intersim_cleveland_dot_plot.pdf')
plot_cleveland_dot(d=res, metric='NDCG', by='s', save='../../figures/raw/bkacc_NDCG_intersim_cleveland_dot_plot.pdf')

# %%
