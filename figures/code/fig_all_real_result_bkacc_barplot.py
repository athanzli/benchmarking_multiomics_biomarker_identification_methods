#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

from setups import *

WIDTH = 0.879370079
HEIGHT = 5.700

# TODO change according to performance ranking
MODEL_ORDER_EMPTY_FILLED = [
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

 'EMPTY0',

 'DIABLO',
 'GAUDI',
 'GDF',
 'Stabl',
 'asmPLSDA',
 'MOFA',
 'DPM',
 'MCIA']

PERF_MODEL_ORDER_EMPTY_FILLED = None # TODO

#%%
def create_bar_plot(data_plot, cmap_name, save_name):
    assert save_name is not None
    #
    data_plot = data_plot.fillna(np.nan)
    for col in data_plot.columns:
        data_plot.loc['EMPTY0', col] = np.nan
        if 'pred' in save_name:
            data_plot.loc['EMPTY0', col] = data_plot[col].min() # TODO

    if 'pred' in save_name:
        data_plot = data_plot.loc[PERF_MODEL_ORDER_EMPTY_FILLED]
    else:
        data_plot = data_plot.loc[MODEL_ORDER_EMPTY_FILLED]

    tasks = data_plot.columns.astype(str)
    models = data_plot.index.astype(str)
    n = len(tasks)
    fig = plt.figure(figsize=(WIDTH * n, HEIGHT))
    axes = []
    for i, task in enumerate(tasks):
        left = i / n
        if i == 0:
            ax = fig.add_axes([left, 0, 1/n, 1])
        else:
            ax = fig.add_axes([left, 0, 1/n, 1], sharey=axes[0])
        axes.append(ax)

    y_pos = np.arange(len(models))

    vmin = np.nanmin(data_plot.values)
    vmax = np.nanmax(data_plot.values)
    norm = mpl.colors.Normalize(vmin=vmin*0.95, vmax=vmax * 1.05)
    cmap_obj = plt.get_cmap(cmap_name)
    ##

    for ax, task in zip(axes, tasks):
        vals = data_plot[task].values
        vmin_cur_col, vmax_cur_col = np.nanmin(vals), np.nanmax(vals)
        colors = cmap_obj(norm(vals))

        ax.barh(
            y_pos,
            vals,
            color=colors,
            edgecolor='k',
            height=0.8,
            linewidth=0.3 # TODO
        )
        
        ax.set_xlim(0, vmax_cur_col*1.1)
        

        ax.set_yticks([])
        ax.set_yticklabels("")
        ax.margins(y=0.007) 

        ax.axvline(
            x=vmax_cur_col,
            color='lightgray',
            linestyle='--',
            linewidth=0.7,
            alpha=0.5,
            ymin=0.0,
            ymax=1.0,
            label=""
        )
        ax.axvline(
            x=vmax_cur_col/2,
            color='lightgray',
            linestyle='--',
            linewidth=0.7,
            alpha=0.5,
            ymin=0.0,
            ymax=1.0,
            label=""
        )
        x_max = vmax_cur_col
        ticks = [x_max/2, x_max]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.4f}" for t in ticks], rotation=25, fontsize=7)
        for idx, spine in enumerate(ax.spines.values()):
            spine.set_visible(False)

    axes[-1].invert_yaxis()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.set_size_inches(WIDTH * len(tasks), HEIGHT)
    plt.show()
    fig.savefig(
        f'../raw/{save_name}.pdf',
        format='pdf',
        dpi=300,
        pad_inches=0,
        bbox_inches=None
    )
    plt.close(fig)

#%%
###############################################################################
# extended mods
###############################################################################
bkacc_metrics = ['AR','NDCG','RR']

################################
data_plots_surv = []
data_plots_drug = []
for i in range(len(bkacc_metrics)):
    data_plot = pd.read_csv(f"../../result/result_TCGA_bk_accuracy_{bkacc_metrics[i]}.csv", index_col=0) 
    data_plot = data_plot.loc[MODELS]
    
    tmp = data_plot.loc[:, data_plot.columns.str.contains('surv')]
    tmp.columns = [bkacc_metrics[i] + '-' + col[5:] for col in tmp.columns]
    data_plots_surv.append(tmp)

    tmp = data_plot.loc[:, data_plot.columns.str.contains('drug')]
    tmp.columns = [bkacc_metrics[i] + '-' + col[5:] for col in tmp.columns]
    data_plots_drug.append(tmp)

data_plot = pd.concat([pd.concat(data_plots_surv, axis=1), pd.concat(data_plots_drug, axis=1)], axis=1)

data_plot_m1 = data_plot.loc[MODELS, data_plot.columns.str.contains('AR-')].copy()
data_plot_m2 = data_plot.loc[MODELS, data_plot.columns.str.contains('NDCG-')].copy()
data_plot_m3 = data_plot.loc[MODELS, data_plot.columns.str.contains('RR-')].copy()

# add overall performance
data_plot_m1.insert(loc=0, column="Overall average", value=data_plot_m1.mean(axis=1))
data_plot_m2.insert(loc=0, column="Overall average", value=data_plot_m2.mean(axis=1))
data_plot_m3.insert(loc=0, column="Overall average", value=data_plot_m3.mean(axis=1))

create_bar_plot(data_plot=data_plot_m1, cmap_name="Reds_r",
                save_name="fig_all_real_bkacc_AR_barplot")
create_bar_plot(data_plot=data_plot_m2, cmap_name="Oranges_r",
                save_name="fig_all_real_bkacc_NDCG_barplot")
create_bar_plot(data_plot=data_plot_m3, cmap_name="Wistia_r",
                save_name="fig_all_real_bkacc_RR_barplot")

# %% overall ranking
avg = (data_plot_m1['Overall average'] + data_plot_m2['Overall average'] + data_plot_m3['Overall average']) / 3
avg = avg.sort_values(ascending=False).to_frame()
dl = avg.loc[DL_MODELS].sort_values(by='Overall average', ascending=False)
nondl = avg.loc[NONDL_MODELS].sort_values(by='Overall average', ascending=False)
print(list(dl.index.values) + list(nondl.index.values))

