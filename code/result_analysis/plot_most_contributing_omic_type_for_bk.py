#%% 
import matplotlib.pyplot as plt
import pickle as pkl
from setups import *

RES_PATH = "/home/athan.li/eval_bk/result/"

PALETTE_OMICS = {
    'CNV'   : "#ffd8b1",
    'DNAm'  : "#95D378",
    'SNV'   : "#EB8888",
    'mRNA'  : "#5C90E4",
    'miRNA' : "#A0DBD5",
}
MODS   = list(PALETTE_OMICS.keys())
COLORS = [PALETTE_OMICS[m] for m in MODS]

def plot_fanchart_grid(df_counts,
                       palette=PALETTE_OMICS,
                       ring_width=0.8,
                       bg_zero="#F5F5F5",
                       startangle=90,
                       tight=True,
                       figsize=None,
                       show_headers=True,
                       col_rotation=0,
                       row_fontsize=14,
                       col_fontsize=14,
                       per_col_width=1.58,
                       per_row_height=0.9,
                       cap=None,
                       wspace=0.02, hspace=0.02,
                       gap_after=None,
                       gap_ratio=0.30, 
                       vlines_after=None, 
                       vline_kwargs=None,
                       vline_at='gap_center'
                       ):
    import numpy as np
    import matplotlib.pyplot as plt

    if cap is None or cap <= 0:
        raise ValueError("cap must be a positive integer.")

    mods   = list(palette.keys())
    colors = [palette[m] for m in mods]
    n_rows, n_cols = df_counts.shape

    if figsize is None:
        figsize = (n_cols * per_col_width, n_rows * per_row_height)

    gap_after = sorted({int(k) for k in (gap_after or []) if 1 <= int(k) <= (n_cols - 1)})
    width_ratios, is_gap_col = [], []
    for j in range(1, n_cols + 1):
        width_ratios.append(1.0); is_gap_col.append(False)
        if j in gap_after:
            width_ratios.append(gap_ratio); is_gap_col.append(True)
    gcols = len(width_ratios)

    fig, ax_grid = plt.subplots(n_rows, gcols, figsize=figsize, constrained_layout=False,
                                gridspec_kw={'width_ratios': width_ratios})
    if n_rows == 1 and gcols == 1:
        ax_grid = np.array([[ax_grid]])
    elif n_rows == 1:
        ax_grid = ax_grid[np.newaxis, :]
    elif gcols == 1:
        ax_grid = ax_grid[:, np.newaxis]

    if tight:
        fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.01,
                            wspace=wspace, hspace=hspace)

    axes = np.empty((n_rows, n_cols), dtype=object)
    jgrid = 0
    for jdata in range(n_cols):
        for i in range(n_rows):
            axes[i, jdata] = ax_grid[i, jgrid]
        jgrid += 1
        if (jdata + 1) in gap_after:
            for i in range(n_rows):
                ax_grid[i, jgrid].axis('off') 
            jgrid += 1

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            cell = df_counts.iat[i, j]
            if not isinstance(cell, dict):
                cell = {}
            vals = np.array([cell.get(m, 0.0) for m in mods], dtype=float)
            s = float(vals.sum())

            if s <= 0:
                parts, part_colors = [1.0], [bg_zero]
            else:
                filled = min(s / cap, 1.0)
                props  = vals / s
                parts  = (props * filled).tolist()
                if filled < 1.0:
                    parts.append(1.0 - filled)
                    part_colors = colors + [bg_zero]
                else:
                    part_colors = colors

            ax.pie(parts, colors=part_colors, startangle=startangle, radius=1.0,
                   wedgeprops=dict(width=ring_width, linewidth=0))
            ax.set(aspect='equal'); ax.axis('off')

    if show_headers:
        for j, col_name in enumerate(df_counts.columns):
            bk_name = str(col_name).split('-')[-1]
            ax_top = axes[0, j]
            ax_top.text(0.5, 1.10,
                        bk_name, # NOTE
                        transform=ax_top.transAxes,
                        ha='center', va='bottom', fontsize=col_fontsize, rotation=col_rotation)
        for i, row_name in enumerate(df_counts.index):
            ax_left = axes[i, 0]
            ax_left.text(-0.05, 0.5, str(row_name), transform=ax_left.transAxes,
                         ha='right', va='center', fontsize=row_fontsize)

    if vlines_after:
        cols = sorted({int(k) for k in vlines_after if 1 <= int(k) <= n_cols})
        if cols:
            if vline_kwargs is None:
                vline_kwargs = dict(color="0.75", linewidth=1.2) # NOTE
            top    = max(axes[0, j].get_position().y1 for j in range(n_cols))
            bottom = min(axes[-1, j].get_position().y0 for j in range(n_cols))

            gap_grid_indices = []
            jgrid = 0
            for jdata in range(n_cols):
                jgrid += 1
                if (jdata + 1) in gap_after:
                    gap_grid_indices.append(jgrid) 
                    jgrid += 1
                else:
                    gap_grid_indices.append(None)

            for k in cols:
                if vline_at == 'gap_center' and (k <= n_cols - 1) and gap_grid_indices[k-1] is not None:
                    gidx = gap_grid_indices[k-1]
                    pos = ax_grid[0, gidx].get_position()
                    x = 0.5 * (pos.x0 + pos.x1)
                else:
                    x = axes[0, k-1].get_position().x1

                line = plt.Line2D([x, x],
                                  [bottom, top],
                                  transform=fig.transFigure,
                                  linestyle='--',
                                  clip_on=False, zorder=1000, **vline_kwargs)
                fig.add_artist(line)

    return fig, axes


#%% the top0.01 cases only version
with open(RES_PATH + "bk_most_contributing_omic_type_count_top0.01_cases.pkl", "rb") as f:
    dfs = pkl.load(f)
dfs = [dfs[MULTIOMICS_DATA_TYPES[i]] for i in range(len(MULTIOMICS_DATA_TYPES))]

idx = dfs[0].index
cols = dfs[0].columns
for d in dfs[1:]:
    idx = idx.union(d.index)
    cols = cols.union(d.columns)
dfs = [d.reindex(index=idx, columns=cols) for d in dfs]

zero = {k: 0 for k in MODS}
df_total = pd.DataFrame(
    [[zero.copy() for _ in range(len(cols))] for __ in range(len(idx))],
    index=idx, columns=cols, dtype=object
)

for d in dfs:
    for i in idx:
        for j in cols:
            cell = d.at[i, j]
            if isinstance(cell, dict):
                for k in MODS:
                    df_total.at[i, j][k] += cell.get(k, 0)

print(MULTIOMICS_DATA_TYPES)

#%%
for i in range(len(MULTIOMICS_DATA_TYPES)):
    df_counts = dfs[i].copy()
    df_counts.index = df_counts.index.map(MODEL_NAME_CORRECTION_MAP)
    fig, axes = plot_fanchart_grid(
        df_counts,
        show_headers=True,
        col_rotation=90,
        row_fontsize=30,
        col_fontsize=30,
        ring_width=1,
        per_col_width=0.9,
        per_row_height=0.9,
        cap=5,
        gap_after=[14, 28, 40, 41],
        gap_ratio=0.6,
        vline_at='gap_center',
        wspace=0.02,
        vlines_after=[14, 28, 40, 41],
        vline_kwargs=dict(color="#A1A1A1", linewidth=2)
    )
    plt.savefig(f'../../figures/raw/top0.01_bk_dominant_omic_type_fanchart_{MULTIOMICS_DATA_TYPES[i]}.pdf', dpi=300)
    plt.show()

df_counts = df_total.copy()
df_counts.index = df_counts.index.map(MODEL_NAME_CORRECTION_MAP)
fig, axes = plot_fanchart_grid(
    df_counts,
    show_headers=True,
    col_rotation=90, 
    row_fontsize=30,
    col_fontsize=30,
    ring_width=1, 
    per_col_width=0.9,
    per_row_height=0.9,
    cap=30,
    gap_after=[14, 28, 40, 41, 43],
    gap_ratio=0.6, 
    vline_at='gap_center',  
    wspace=0.02,
    vlines_after=[14, 28, 40, 41, 43],
    vline_kwargs=dict(color="#A1A1A1", linewidth=2)
)
plt.savefig(f'../../figures/raw/top0.01_bk_dominant_omic_type_fanchart_all.pdf', dpi=300)
plt.show()

#%%
def add_tiny_legend(palette=PALETTE_OMICS,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    marker_size=8,
                    hollow=False):
    from matplotlib.lines import Line2D
    handles = []
    for m, c in palette.items():
        handles.append(
            Line2D([0], [0],
                   marker='o',
                   linestyle='',
                   markersize=marker_size,
                   markerfacecolor=('none' if hollow else c),
                   markeredgecolor=(c if hollow else 'none'),
                   label=m)
        )
    leg = plt.legend(handles=handles,
                     ncol=len(handles),
                     loc=loc, bbox_to_anchor=bbox_to_anchor,
                     frameon=False,
                     handlelength=0, 
                     handletextpad=0.3,
                     columnspacing=0.8,
                     markerscale=1.0) 
    return leg

add_tiny_legend()
plt.savefig("../../figures/raw/legend_circles_single_omic_types.pdf", dpi=300)
plt.show()

