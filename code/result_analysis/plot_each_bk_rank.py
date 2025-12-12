#%%
from setups import *
import pickle as pkl

RES_PATH = "/home/athan.li/eval_bk/result_09052025/"
METRIC_ABVS = ['AR','NDCG','RR']

MULTIOMICS_DATA_TYPES = [
"CNV+DNAm+mRNA",
"CNV+SNV+mRNA",
"CNV+mRNA+miRNA",
"DNAm+SNV+mRNA",
"DNAm+mRNA+miRNA",
"SNV+mRNA+miRNA"
]

MODELS = NON_CLASSICAL_MODELS

with open("../../result/res_all_ft_scores_gene.pkl",'rb') as f:
    res = pkl.load(f)    

#%% load bk
surv_bks = pd.read_csv("/home/athan.li/eval_bk/data/bk_set/processed/survival_task_bks.csv", index_col=0)
surv_bks = surv_bks.loc[(surv_bks['Task'].isin(['survival_BRCA','survival_LUAD','survival_COADREAD'])) & (surv_bks['Level']<='B')]
drug_bks = pd.read_csv("/home/athan.li/eval_bk/data/bk_set/processed/drug_response_task_bks.csv", index_col=0)
drug_bks = drug_bks[['Task','Gene','Level']]
drug_bks = drug_bks.loc[(drug_bks['Level']<='B') & (drug_bks['Task'].isin(['drug_response_Cisplatin-BLCA','drug_response_Temozolomide-LGG']))]
bks = pd.concat([surv_bks, drug_bks], axis=0)

#%%
dfs = pd.DataFrame(
    [[{mm:[] for mm in MULTIOMICS_DATA_TYPES} \
      for _ in bks['Task'] + '-' + bks['Gene']] for _ in MODELS],
    index = MODELS,
    columns = bks['Task'] + '-' + bks['Gene']
)
dfs_ft_gset_size = pd.DataFrame(
    [[{mm:np.nan for mm in MULTIOMICS_DATA_TYPES} \
      for _ in bks['Task'] + '-' + bks['Gene']] for _ in MODELS],
    index = MODELS,
    columns = bks['Task'] + '-' + bks['Gene']
)

#%%
for it_cnt, (task, model, mm, fold) in enumerate(product(TASKS, MODELS, MULTIOMICS_DATA_TYPES, FOLDS)):
    print(f"\n====================={it_cnt}/3000======================")
    print(task, model, mm, fold)
    bk_set0 = bks.loc[bks['Task']==task, 'Gene'].unique()

    ###############
    ft = res[task][mm][fold][model].copy()
    ft.index = ft.index.str.split('@').str[1]
    ft = ft.groupby(ft.index, sort=False).max().sort_values(by='score', ascending=False)
    ft_gset = ft.index.values # update current gene set
    mask = np.isin(bk_set0, ft_gset)
    bk_set = bk_set0[mask]
    if len(bk_set) == 0: continue # if no bk present (e.g., DeePathNet for drug task), continue

    ###############
    rank = ft.index.get_indexer(bk_set).astype(int) + 1
    for idx, bk in enumerate(bk_set):
        cell = dfs.at[model, task+'-'+bk]
        cell[mm].append(rank[idx])
        dfs.at[model, task+'-'+bk] = cell
        
        cell = dfs_ft_gset_size.at[model, task+'-'+bk]
        cell[mm] = len(ft_gset)
        dfs_ft_gset_size.at[model, task+'-'+bk] = cell

# %% save dfs
with open(RES_PATH + "dfs_each_bk_rank.pkl", 'wb') as f:
    pkl.dump(dfs, f)
with open(RES_PATH + "dfs_ft_gset_sizes.pkl", 'wb') as f:
    pkl.dump(dfs_ft_gset_size, f)

#%% plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

MULTI_OMICS_DATA_TYPE_PALETTE = {
    'CNV+DNAm+mRNA': '#A5BEAF',
    'CNV+SNV+mRNA':  '#C2A5B4',
    'CNV+mRNA+miRNA':'#A9C1CE',
    'DNAm+SNV+mRNA': '#9FA4A1',
    'DNAm+mRNA+miRNA':'#86BFBB',
    'SNV+mRNA+miRNA':'#A2A6C0',
}

def plot_task_rank_percentiles(dfs, dfs_ft_gset_size, task,
                               palette=MULTI_OMICS_DATA_TYPE_PALETTE,
                               nrows=3,
                               ncols=5,
                               marker_size=30,
                               alpha=0.8,
                               dpi=300,
                               plot_legend=True):
    assert dfs.shape == dfs_ft_gset_size.shape

    task_cols = [c for c in dfs.columns if c.startswith(task + '-')]
    if not task_cols:
        raise ValueError(f"No columns found for task {task}")

    methods = list(dfs.index)
    n_methods = len(methods)
    n_panels = len(task_cols)
    
    total_slots = nrows * ncols
    if n_panels > total_slots:
        nrows = int(math.ceil(n_panels / ncols))

    global_max = 0.0
    for col in task_cols:
        for method in methods:
            cell  = dfs.at[method, col]
            sizes = dfs_ft_gset_size.at[method, col]
            if not isinstance(cell, dict) or not isinstance(sizes, dict):
                continue
            for mtype, ranks in cell.items():
                size = sizes.get(mtype)
                if not size:
                    continue
                for r in ranks:
                    if r is None or r <= 0 or r > size:
                        continue
                    global_max = max(global_max, -math.log10(r / size))
    if global_max == 0:
        global_max = 1.0

    fig_w = 3.4 * ncols
    fig_h = max(2.0, 0.38 * n_methods) * nrows
    fig, axes = plt.subplots(nrows,
                             ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False,
                             dpi=dpi,
                             sharex=True,
                             sharey=True)
    ax_arr = axes.ravel()

    y_positions = {m: i for i, m in enumerate(methods)}

    first_panel = True
    legend_labels_seen, legend_handles, legend_labels = set(), [], []

    for j, col in enumerate(task_cols):
        ax = ax_arr[j]
        title = col.split('-')[-1]

        for method in methods:
            cell  = dfs.at[method, col]
            sizes = dfs_ft_gset_size.at[method, col]
            if not isinstance(cell, dict) or not isinstance(sizes, dict):
                continue

            for mtype, ranks in cell.items():
                size = sizes.get(mtype)
                if not size or not ranks:
                    continue

                xs = []
                for r in ranks:
                    if r is None or r <= 0 or r > size:
                        continue
                    p = r / size
                    xs.append(-math.log10(p))

                if not xs:
                    continue

                ys = [y_positions[method]] * len(xs)
                color = palette.get(mtype, None)
                sc = ax.scatter(xs, ys, 
                                s=marker_size, alpha=alpha, color=color,
                                edgecolors='none')

                if first_panel and (mtype not in legend_labels_seen):
                    legend_labels_seen.add(mtype)
                    legend_handles.append(sc)
                    legend_labels.append(mtype)

        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, global_max * 1.05) 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='x', linestyle=':', linewidth=1.0, alpha=0.8)
        ax.set_yticks(range(n_methods))
        if (j % ncols) == 0:
            ax.set_yticklabels(methods, fontsize=10)
        else:
            ax.tick_params(axis='y', which='both', labelleft=False)
        ax.set_xlabel(r"$-\log_{10}(\text{rank percentile})$", fontsize=10)

        first_panel = False

    axes[0, 0].invert_yaxis()

    for k in range(n_panels, nrows * ncols):
        fig.delaxes(ax_arr[k])

    if legend_handles and plot_legend:
        fig.legend(legend_handles, legend_labels, title="Multi-omics data types",
                   loc='lower right', bbox_to_anchor=(0.9, 0.1), borderaxespad=0., fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(f"../../figures/raw/rank_position_per_bk_{task}.pdf", dpi=300)
    plt.show()

#%% standardize model names
dfs.index = dfs.index.map(MODEL_NAME_CORRECTION_MAP)
dfs_ft_gset_size.index = dfs_ft_gset_size.index.map(MODEL_NAME_CORRECTION_MAP)

#%%
task = "survival_BRCA"
plot_task_rank_percentiles(
    dfs, dfs_ft_gset_size, task,
    nrows=3,
    ncols=5)

#%%
task = "survival_LUAD"
plot_task_rank_percentiles(
    dfs, dfs_ft_gset_size, task,
    nrows=3,
    ncols=5)

#%%
task = "survival_COADREAD"
plot_task_rank_percentiles(
    dfs, dfs_ft_gset_size, task,
    nrows=3,
    ncols=5)

#%%
task = "drug_response_Cisplatin-BLCA"
plot_task_rank_percentiles(
    dfs, dfs_ft_gset_size, task,
    nrows=1,
    ncols=1,
    plot_legend=False)

#%%
task = "drug_response_Temozolomide-LGG"
plot_task_rank_percentiles(
    dfs, dfs_ft_gset_size, task,
    nrows=1,
    ncols=2,
    plot_legend=False)

