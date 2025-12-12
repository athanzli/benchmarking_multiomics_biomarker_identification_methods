#%%
from setups import *

PVAL_SIG_THRESH = -np.log10(0.05)

MODEL_COLOR_PALETTE = {
'DeePathNet':"#F6A29C",
 'DeepKEGG':"#F6A29C",
 'MOGLAM':"#F6A29C",
 'TMO-Net':"#F6A29C",
 'CustOmics':"#F6A29C",
 'GENIUS':"#F6A29C",
 'Pathformer':"#F6A29C",
 'GNN-SubNet':"#F6A29C",
 'P-Net':"#F6A29C",
 'MOGONET':"#F6A29C",
 'MORE':"#F6A29C",
 'MoAGL-SA':"#F6A29C",

 'DIABLO':"#F6A29C",
 'GAUDI':"#F6A29C",
 'GDF':"#F6A29C",
 'Stabl':"#F6A29C",
 'asmbPLS-DA':"#F6A29C",
 'MOFA':"#F6A29C",
 'DPM':"#F6A29C",
 'MCIA':"#F6A29C"
}

TASK_NAME_CORRECTION_MAP = {
'survival_BRCA' : "Survival BRCA",
 'survival_LUAD' : "Survival LUAD",
 'survival_COADREAD' : "Survival COADREAD",
 'drug_response_Cisplatin-BLCA' : "Drug response Cisplatin (BLCA)",
 'drug_response_Temozolomide-LGG' : "Drug response Temozolomide (LGG)"
}

MODELS = NON_CLASSICAL_MODELS

#%% 
fig, axes = plt.subplots(
    nrows=1, ncols=5,
    figsize=(17, 7),
    sharex=False,
    sharey=False
)

for sutplot_idx, task in enumerate(TASKS):

    with open(f"/home/athan.li/eval_bk/result/bkacc_res_mwtestpval_exact_TCGA_{task}.pkl", 'rb') as f:
        res = pickle.load(f)

    model_res_all = {}
    for model in MODELS:

        model_res = []
        for fold in FOLDS:
            model_res.append(res[fold].loc[model].to_frame().rename(columns={model:f"{model}_fold{fold}"}))
        model_res = pd.concat(model_res, axis=1)

        ###
        model_res_all[MODEL_NAME_CORRECTION_MAP[model]] = model_res.values.flatten().astype(np.float64)

    data = pd.DataFrame(model_res_all)

    #plot half violin
    df_neglog = -np.log10(data.clip(lower=1e-300))

    order = [MODEL_NAME_CORRECTION_MAP[model] for model in MODELS]
    
    violin_data = [ df_neglog[model].dropna().values
                    for model in order ]

    ax = axes[sutplot_idx]

    color_list = [ MODEL_COLOR_PALETTE[m] for m in order ]

    # horizontal violin
    sns.violinplot(
        data=violin_data,  
        orient="h",
        palette=color_list,
        # order=order,
        cut=0,    
        bw_adjust=0.6, # smoother KDE
        linewidth=0.0,
        ax=ax,
        zorder=1
    )

    #  covering the lower half with a white rectangle
    x_min, x_max = ax.get_xlim()
    for i in range(len(order)):
        # rectangle: (x0, y0, width, height)
        ax.add_patch(
            plt.Rectangle(
                (x_min, i),  
                x_max - x_min, 0.5, 
                color="white",
                zorder=2 
            )
        )


    # 
    for i, model in enumerate(order):
        x = df_neglog[model].values
        x = x[~np.isnan(x)]
        if len(x)==0: x = np.asarray([-0.1] * len(df_neglog[model].values))
        # jitter y within the lower half [i-0.5, i]
        y = np.random.uniform(low=i , high=i + 0.15, size=len(x))
        colors = np.asarray([MODEL_COLOR_PALETTE[model] if xi > PVAL_SIG_THRESH else 'grey' for xi in x])
        sig_mask = np.asarray([True if xi > PVAL_SIG_THRESH else False for xi in x])
        
        ax.scatter(
            x[sig_mask], y[sig_mask],
            c=colors[sig_mask],
            s=8,
            alpha=1.0,
            edgecolors='none',
            zorder=4 
        )
        ax.scatter(
            x[~sig_mask], y[~sig_mask],
            c=colors[~sig_mask],
            s=5,
            alpha=0.6,
            edgecolors='none',
            zorder=4  
        )

    ax.axvline(x=PVAL_SIG_THRESH, color='grey', linestyle='--', linewidth=1, zorder=5, ymin=0.0, ymax=1)

    ax.set_xlabel(r'$-\log_{10}(p)$')
    ax.set_xlim(left=0)        # origin at zero
    
    if sutplot_idx == 0:
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
    else:
        ax.set_yticks(range(len(order)))
        ax.tick_params(left=False, labelleft=False)
    
    # per‑task title
    ax.set_title(TASK_NAME_CORRECTION_MAP.get(task, task), fontsize=10)

    sns.despine(left=False, bottom=False, top=False, right=False)

    ax.margins(y=0.005)   # 2% vertical margin instead of default ~5%

    # turn on horizontal grid lines at each major y-tick
    ax.grid(
        axis='y',  
        which='major', # at major tick locations
        linestyle='-',  
        linewidth=1.5, 
        color='lightgrey', 
        zorder=6  
    )
    ax.grid(False, axis='x')

fig.suptitle("Distribution of p-values for ranking significance of biomarkers in real data results", fontsize=14, y=1.02)
fig.tight_layout(pad=0, rect=[0, 0, 1, 1])  
plt.savefig('../../figures/raw/mwtest_pval_distribution_for_all_real_resutls.pdf',dpi=300)
plt.show()

