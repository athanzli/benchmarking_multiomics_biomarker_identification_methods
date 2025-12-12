#%%
from setups import *
import cmocean
import matplotlib.pyplot as plt

MODEL_NAME_CORRECTION_MAP_REVERSE = {
    v : k for k, v in MODEL_NAME_CORRECTION_MAP.items()
}

MODELS = [MODEL_NAME_CORRECTION_MAP_REVERSE[m] for m in NON_CLASSICAL_MODELS]

# %%
###############################################################################
# similarity between mods
###############################################################################
with open("../../result/res_TCGA_model_similarity_RBO_p=0.98.pkl", 'rb') as f:
    sim1 = pickle.load(f)
with open("../../result/res_TCGA_model_similarity_KendallTau.pkl", 'rb') as f:
    sim2 = pickle.load(f)

metrics = ['RBO', 'KendallTau']
sims = [sim1, sim2]

# v1
for metric_idx, sim in enumerate(sims):
    all_dfss = []
    for task in TASKS:
        print(task, "...")
        # sim[task][0]['CNV+DNAm+mRNA']
        all_dfs = [df.loc[MODELS, MODELS]
                for fold_dict in sim[task].values()
                for df in fold_dict.values()]
        if [all_dfs[i].isna().any().any() for i in range(len(all_dfs))].count(True) > 0:
            print("Warning: contains nan", task)
        all_dfss.extend(all_dfs)

    sum_df   = pd.DataFrame(0, index=all_dfss[0].index, columns=all_dfss[0].columns)
    for df in all_dfss:
        sum_df   = sum_df.add(df)
    sim_mean = sum_df / len(all_dfss)

    ########### plot
    data = sim_mean
    data = data.rename(
        columns=MODEL_NAME_CORRECTION_MAP,
        index=MODEL_NAME_CORRECTION_MAP
    )

    # fill diagonal with 0
    np.fill_diagonal(data.values, 0)
    vmax = data.max().max()

    mask = np.triu(np.ones_like(data, dtype=bool), k=0)
    
    #
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data.iloc[1:, :-1],
        mask=mask[1:, :-1],
        cmap="cmo.deep", # YlGnBu
        vmin=0, vmax=vmax,
        square=True,
        linewidths=0.75,
        cbar_kws={"shrink": .75, "label":"score"},
        ax=ax,
        # annot=True,
        # fmt='.2f',
    )
    if metrics[metric_idx] == 'KendallTau':
        metric_name = "Kendall's \N{GREEK SMALL LETTER TAU}"
    else:
        metric_name = 'RBO'
    ax.set_title(f"Methods ranking similarity ({metric_name})")
    plt.tight_layout()
    plt.savefig('../raw/model_similarity_' + metrics[metric_idx] + '.pdf', dpi=300)
    plt.show()

