# execution time ~6min

#%%
from setups import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import math

MODELS = np.array(NON_CLASSICAL_MODELS)
MODELS = MODELS[~np.isin(MODELS, ['DPM','GNNSubNet'])] # these two have no omics specific feature scores
ALL_SINGLE_MOD = MODS

RES_PATH = "/home/athan.li/eval_bk/result/"

#%%
with open(RES_PATH + "res_all_ft_scores_raw.pkl", 'rb') as f:
    res = pickle.load(f) # task, mm data type, fold, model

#%% plot funcs
def plot_prop_curves_for_single_model(df, fig_size=(8,6), title='', N_THRES=100):
    r"""
    
    Args:
        df (pd.DataFrame): a sorted (descending) dataframe of feature scores
    usage:
        plot_prop_curves_for_single_model(ft, title=f"{model}_{task}_{mm}", N_THRES=100)
    """
    mods = df.index.str.split('@').str[0].values.astype(str)

    total = len(mods)

    mod_types = [m for m in PALETTE_OMICS.keys() if m in mods]
    if not mod_types:
        raise ValueError("no mod.")

    proportions = {mod: [] for mod in mod_types}
    percent_steps = np.arange(1, 1 + N_THRES)

    for p in percent_steps:
        top_k = max(int(total * p / N_THRES), 1)
        top_mods = mods[:top_k]
        for mod in mod_types:
            proportions[mod].append(np.sum(top_mods == mod) / top_k)

    plt.figure(figsize=fig_size)
    for mod in mod_types:
        plt.plot(
            percent_steps,
            proportions[mod],
            label=mod,
            color=PALETTE_OMICS[mod],
            linewidth=2
        )
    plt.xlabel('Top X% of features')
    plt.ylabel('Proportion of omics type')
    plt.title(title)
    plt.legend(title='')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_prop_curves(ft_dict,
                         save=None,
                         title=None,
                         N_THRES=100,
                         per_col_width=0.26,
                         per_row_height=0.28,
                         lw=0.9,
                         show_headers_col=True,
                         show_headers_row=True,
                         show_legend=True,
                         rasterize=True,
                         close=False,
                         group_size=None,
                         section_gap_frac=0.35,
                         section_line=True,
                         section_line_kwargs=None,
                         wspace=0.05,
                         hspace=0.10):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.lines import Line2D

    models = [MODEL_NAME_CORRECTION_MAP[m] for m in MODELS]
    tasks = [TASK_NAME_CORRECTION_MAP[t] for t in TASKS]
    mm_types = list(MULTIOMICS_DATA_TYPES)

    if group_size is None:
        group_size = len(mm_types)

    n_rows = len(models)
    n_tasks = len(tasks)
    n_cols = n_tasks * group_size
    if n_rows == 0 or n_cols == 0:
        raise ValueError("empty grid")

    col_pairs = []
    for task in tasks:
        for mm in mm_types:
            col_pairs.append((task, mm))

    cols_with_gaps = n_cols + (n_tasks - 1)
    width_ratios = []
    spacer_cols = []
    cur_idx = 0
    for g in range(n_tasks):
        width_ratios.extend([1.0] * group_size)
        cur_idx += group_size
        if g < n_tasks - 1:
            width_ratios.append(section_gap_frac)
            spacer_cols.append(cur_idx)
            cur_idx += 1

    effective_cols = n_cols + section_gap_frac * (n_tasks - 1)
    fig_w = effective_cols * per_col_width
    fig_h = n_rows * per_row_height

    fig, axes = plt.subplots(
        n_rows, cols_with_gaps,
        figsize=(fig_w, fig_h),
        gridspec_kw={'width_ratios': width_ratios},
        sharex=False, sharey=False
    )

    if n_rows == 1 and cols_with_gaps == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif cols_with_gaps == 1:
        axes = axes[:, np.newaxis]

    percent_steps = np.arange(1, N_THRES + 1)

    leg_handles = [mlines.Line2D([], [], color=PALETTE_OMICS[m], lw=lw, label=m)
                   for m in PALETTE_OMICS.keys()]

    def _format_ax(ax):
        ax.set_xlim(1, N_THRES)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.margins(0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
            sp.set_alpha(0.6)

    for i in range(n_rows):
        for sc in spacer_cols:
            ax_gap = axes[i, sc]
            ax_gap.axis('off')

    for i, model in enumerate(models):
        for j, (task, mm) in enumerate(col_pairs):
            grid_j = j + (j // group_size)
            ax = axes[i, grid_j]
            df = ft_dict.get(model, {}).get(task, {}).get(mm, None)

            if df is None or len(df.index) == 0:
                _format_ax(ax)
            else:
                mods = np.array([str(idx).split('@', 1)[0] for idx in df.index], dtype=object)
                total = mods.shape[0]
                _format_ax(ax)

                if total > 0:
                    mod_types_present = [m for m in PALETTE_OMICS.keys() if np.any(mods == m)]
                    if mod_types_present:
                        top_k = (total * percent_steps) // N_THRES
                        top_k = np.clip(top_k, 1, total)
                        top_idx = top_k - 1

                        eq_cache = {}
                        for m in mod_types_present:
                            eq = eq_cache.get(m)
                            if eq is None:
                                eq = (mods == m).astype(np.int32)
                                eq_cache[m] = eq
                            cum = np.cumsum(eq)
                            props = cum[top_idx] / top_k
                            ax.plot(percent_steps, props,
                                    color=PALETTE_OMICS[m], lw=lw,
                                    rasterized=rasterize)

            if show_headers_col and i == 0:
                ax.set_title(f"{task}\n{mm}", fontsize=6, pad=2,
                rotation=90) # NOTE
            if show_headers_row and (j == 0):
                ax.text(-0.04, 0.5, model, transform=ax.transAxes,
                        ha="right", va="center", fontsize=6)

    if show_legend:
        fig.legend(handles=leg_handles,
                   labels=[h.get_label() for h in leg_handles],
                   loc="upper center",
                   ncol=len(leg_handles),
                   frameon=False,
                   bbox_to_anchor=(0.5, 1.02),
                   fontsize=7)

    plt.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.92,
        bottom=0.05,
        wspace=wspace, hspace=hspace)

    if section_line and spacer_cols:
        if section_line_kwargs is None:
            section_line_kwargs = dict(linestyle="--", linewidth=0.6, color="k", alpha=0.6)
        y0 = axes[-1, 0].get_position().y0
        y1 = axes[0, 0].get_position().y1
        for sc in spacer_cols:
            bb = axes[0, sc].get_position()
            x = bb.x0 + bb.width * 0.5
            fig.add_artist(Line2D([x, x], [y0, y1],
                                  transform=fig.transFigure, clip_on=False, zorder=5,
                                  **section_line_kwargs))

    if title:
        fig.suptitle(title, y=1.08 if show_legend else 1.02, fontsize=10)

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight",
                    pad_inches=0.01) # TODO
        if close:
            plt.close(fig)
    return fig, axes

#%% get data for plot
ft_dict = {
    MODEL_NAME_CORRECTION_MAP[model] : {
        TASK_NAME_CORRECTION_MAP[task] : {
            mm : np.nan for mm in MULTIOMICS_DATA_TYPES
        } for task in TASKS
    } for model in MODELS
}
for model, task, mm in product(MODELS, TASKS, MULTIOMICS_DATA_TYPES):
    # concatenate by folds
    fts = []
    for fold in FOLDS:
        fts.append(res[task][mm][fold][model].copy()) # concatenation of folds
    ft = pd.concat(fts, axis=0)
    ft = ft.sort_values(by='score', ascending=False)
    ft_dict[MODEL_NAME_CORRECTION_MAP[model]][TASK_NAME_CORRECTION_MAP[task]][mm] = ft

#%% save a summary
models = [MODEL_NAME_CORRECTION_MAP[m] for m in MODELS]
tasks = [TASK_NAME_CORRECTION_MAP[t] for t in TASKS]
topom = {
    task : pd.DataFrame(
        index=models,
        columns=MULTIOMICS_DATA_TYPES,
        dtype=str,
    ) for task in tasks
}
for task in tasks:
    for model in models:
        for mm in MULTIOMICS_DATA_TYPES:
            # compute top omics type
            ft = ft_dict[model][task][mm].copy()
            mods = ft.index.str.split('@').str[0].values.astype(str)
            a1 = np.unique(mods[:int(0.01*len(mods))], return_counts=True)[0]
            a2 = np.unique(mods[:int(0.01*len(mods))], return_counts=True)[1]
            top_type = a1[np.where(np.max(a2)==a2)[0][0]]
            # store
            topom[task].loc[model, mm] = top_type

with open(RES_PATH + 'summary_top_omics_types_per_method_task_mm.pkl', 'wb') as f:
    pickle.dump(topom, f)

#%% plot
fig, axes = plot_all_prop_curves(
    ft_dict,
    N_THRES=1000,
    per_col_width=0.24,
    per_row_height=0.28,
    section_gap_frac=0.35,
    section_line=True,
    section_line_kwargs=dict(linestyle="--", linewidth=0.6, color="k", alpha=0.6),
    lw=0.9,
    show_headers_col=False, # NOTE
    show_headers_row=True, # NOTE
    show_legend=False, # NOTE
    save='../../figures/raw/fig_omics_type_prop_curve_nthres=1000.tiff', # NOTE
    title=None,
)
plt.show()

