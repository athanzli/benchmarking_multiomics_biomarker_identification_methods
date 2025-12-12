#%%
import pickle
from setups import *
RES_PATH = "/home/athan.li/eval_bk/result/"
MODELS = NON_CLASSICAL_MODELS
MODELS = list(np.asarray(MODELS)[~np.isin(np.asarray(MODELS), MODELS_WITH_BARE_GENES_FT_SCORES)])
MAPPINGS = {
    'DNAm' : C2G,
    'miRNA': R2G
}

TASK_NAME_CORRECTION_MAP_REVERSE = {
    v : k for k, v in TASK_NAME_CORRECTION_MAP.items()
}
MODEL_NAME_CORRECTION_MAP_REVERSE = {
    v : k for k, v in MODEL_NAME_CORRECTION_MAP.items()
}

def mol_gene_level_conversion(ft, model):
    r"""
    map CpGs to gene-level for non-gene-centric models.
    map gene-level miRNA to mol level for gene-centric models.

    Args:
        ft (pd.DataFrame): single-col dataframe with scores.
        model (str): model name.
    Returns:
        ft (pd.DataFrame): converted single-col dataframe with scores.
    """
    mdic = mod_mol_dict(ft.index)
    assert len(np.unique(ft.index))==len(ft)
    if ('miRNA' in mdic['mods_uni']):
        mols = mdic['mols'][mdic['mods']=='miRNA']
        if model not in MODELS_GENE_CENTRIC:
            assert np.array(['hsa-' in mol for mol in mols]).all() 
        else:
            assert np.array(['hsa-' not in mol for mol in mols]).all()
            r2g = R2G.loc[R2G['gene'].isin(mols)]
            g2r = r2g
            g2r.index = g2r['gene']
            g2r = g2r.drop_duplicates().drop(columns=['gene'])

            mirna_ft = ft.loc[mdic['mods']=='miRNA']
            mirna_ft.index = mdic['mols'][mdic['mods']=='miRNA']
            g2r['score'] = mirna_ft.loc[g2r.index, 'score'].values.flatten()
            g2r = g2r.groupby('miRNA').mean() #NOTE

            ft = ft.loc[mdic['mods']!='miRNA']
            assert np.array(['hsa-' in mol for mol in g2r.index]).all()
            g2r.index = ['miRNA@' + tmp for tmp in g2r.index]
            ft = pd.concat([ft, g2r], axis=0).sort_index()
    
    mdic = mod_mol_dict(ft.index)
    if ('DNAm' in mdic['mods_uni']) and (model not in MODELS_GENE_CENTRIC):
        df_cpg = C2G.loc[mdic['mols'][mdic['mods']=='DNAm'], ['gene']]
        cpg_ft = ft.loc[mdic['mods']=='DNAm']
        cpg_ft.index = mdic['mols'][mdic['mods']=='DNAm']
        df_cpg['score'] = cpg_ft.loc[df_cpg.index].values.flatten()
        df_cpg = df_cpg.groupby('gene').mean() #NOTE
        df_cpg.index = 'DNAm@' + df_cpg.index
        
        ft = ft.loc[mdic['mods']!='DNAm']
        ft = pd.concat([ft, df_cpg], axis=0).sort_index()
    return ft

def run_RRA(list_rankings):
    """
    Args:
        list_rankings: a list of ranked lists. higher-ranked elements are at earlier rankings in the list.
    Returns:
        a dataframe with information of the RRA result.
    """
    from rpy2.robjects import ListVector
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    RRA = importr('RobustRankAggreg')

    # convert to str
    py_lists = []
    for ranking in list_rankings:
        py_lists.append([str(elem) for elem in ranking])
    # num of total ele
    unique_items = set(item for lst in py_lists for item in lst)
    N = len(unique_items)

    r_lists = ListVector({
        f"rank{i+1}": StrVector(lst)
        for i, lst in enumerate(py_lists)
    })

    r_res = RRA.aggregateRanks(glist=r_lists, N=N, method='RRA')

    consensus_df = pandas2ri.rpy2py(r_res)
    consensus_df.columns = ['name', 'p-value']
    consensus_df = consensus_df.sort_values('p-value', ascending=True).reset_index(drop=True) # lower the better
    consensus_df.index = consensus_df['name']
    consensus_df = consensus_df.drop(columns=['name'])

    return consensus_df

#%%
with open(RES_PATH + 'summary_top_omics_types_per_method_task_mm.pkl', 'rb') as f:
    topom = pickle.load(f)
with open(RES_PATH + 'res_all_ft_scores_raw.pkl', 'rb') as f:
    ft_all = pickle.load(f)

#%% 
consensus_panel = {
    task : {
        mod : None for mod in MODS
    } for task in TASKS
}
for task in TASKS:
    print("==================== Processing task:", task, "====================")
    # avg. cv folds, and store into a dataframe
    with open(RES_PATH + f"bkacc_res_AR_TCGA_{task}.pkl", 'rb') as f:
        res1 = pickle.load(f)
    with open(RES_PATH + f"bkacc_res_NDCG_TCGA_{task}.pkl", 'rb') as f:
        res2 = pickle.load(f)
    with open(RES_PATH + f"bkacc_res_RR_TCGA_{task}.pkl", 'rb') as f:
        res3 = pickle.load(f)
    frames  = [res1[fold].loc[MODELS, MULTIOMICS_DATA_TYPES] for fold in FOLDS]
    df1 = pd.concat(frames).groupby(level=0).mean().loc[MODELS, MULTIOMICS_DATA_TYPES]
    frames  = [res2[fold].loc[MODELS, MULTIOMICS_DATA_TYPES] for fold in FOLDS]
    df2 = pd.concat(frames).groupby(level=0).mean().loc[MODELS, MULTIOMICS_DATA_TYPES]
    frames  = [res3[fold].loc[MODELS, MULTIOMICS_DATA_TYPES] for fold in FOLDS]
    df3 = pd.concat(frames).groupby(level=0).mean().loc[MODELS, MULTIOMICS_DATA_TYPES]
    df_acc = (df1+df2+df3)/3

    with open(RES_PATH + f"stability_res_RBO_TCGA_{task}.pkl", 'rb') as f: # NOTE 0.98
        res1 = pickle.load(f)
    with open(RES_PATH + f"stability_res_RPSD_TCGA_{task}.pkl", 'rb') as f:
        res2 = pickle.load(f)
    with open(RES_PATH + f"stability_res_KendallTau_TCGA_{task}.pkl", 'rb') as f:
        res3 = pickle.load(f)
    df1 = res1.loc[MODELS, MULTIOMICS_DATA_TYPES]
    df2 = res2.loc[MODELS, MULTIOMICS_DATA_TYPES]
    df3 = res3.loc[MODELS, MULTIOMICS_DATA_TYPES]
    df_sta = (df1 - df2 + df3)/3

    # rank
    acc = df_acc.stack(dropna=True).rename_axis(["model", "mm"]).reset_index(name="value")
    acc = acc.sort_values(by='value', ascending=False)
    sta = df_sta.stack(dropna=True).rename_axis(["model", "mm"]).reset_index(name="value")
    sta = sta.sort_values(by='value', ascending=False)

    # add the top type info
    task = TASK_NAME_CORRECTION_MAP[task]
    models = [MODEL_NAME_CORRECTION_MAP[m] for m in MODELS]
    acc['model'] = acc['model'].map(MODEL_NAME_CORRECTION_MAP)
    sta['model'] = sta['model'].map(MODEL_NAME_CORRECTION_MAP)
    s = topom[task].stack()
    acc['top type'] = acc.set_index(['model','mm']).index.map(s)
    sta['top type'] = sta.set_index(['model','mm']).index.map(s)
    acc['model-mm'] = acc['model'] + '-' + acc['mm']
    sta['model-mm'] = sta['model'] + '-' + sta['mm']
    lookup = pd.concat([acc[['top type', 'model-mm']], sta[['top type', 'model-mm']]], axis=0).drop_duplicates() # look up table for the top type
    lookup.index = lookup['model-mm']

    ## (weighted) average acc and sta overall
    acc.index = acc['model-mm']
    sta.index = sta['model-mm']
    res = (acc[['value']].loc[sta.index] + sta[['value']]) / 2
    res = res.sort_values(by='value', ascending=False)
    res['top type'] = lookup.loc[res.index, 'top type']
    # till full coverage of 5 multi-omics types, each having 3 runs to aggregate from
    AGG_NUM_PER_OM = 3 # each having 3 runs to aggregate from
    for topn in range(len(res)):
        flag = True
        for mm in MODS:
            if (res.iloc[:topn]['top type']==mm).sum() < AGG_NUM_PER_OM: flag = False
        if flag is False: continue
        else: break
    print("Stopped at topn =",topn)
    info = res.iloc[:topn].copy()
    model_mm = info.index.str.rsplit('-', n=1, expand=True)
    info['model'] = [a for a, _ in model_mm]
    info['mm']    = [b for _, b in model_mm]

    # start aggregation
    ## first, aggregate
    assert len(np.unique(info.index))==len(info)
    for mod in MODS:
        cnt_cur_mod = 0
        cur_mod_list_rankings = []
        model_agg_info = {}
        for i, run in enumerate(info.index):
            model = info.iloc[i].loc['model']
            mm = info.iloc[i].loc['mm']
            top_type = info.loc[run, 'top type']
            if top_type != mod: continue
            
            list_rankings = []
            for fold in FOLDS:
                ft_tmp = ft_all[TASK_NAME_CORRECTION_MAP_REVERSE[task]][mm][fold][MODEL_NAME_CORRECTION_MAP_REVERSE[model]].copy()
                # mol conversion.
                #    map gene-level miRNA to mol level for gene-centric models.
                if (model in MODELS_GENE_CENTRIC) and (mod == 'miRNA'):
                    ft_tmp = mol_gene_level_conversion(ft_tmp, model)
                #    map CpGs to gene-level for non-gene-centric models.
                if (model not in MODELS_GENE_CENTRIC) and (mod == 'DNAm'):
                    ft_tmp = mol_gene_level_conversion(ft_tmp, model)
                mods = ft_tmp.index.str.split('@').str[0]
                cur_fold_ranking = ft_tmp.loc[mods==top_type].sort_values(by='score', ascending=False)
                list_rankings.append(cur_fold_ranking.index.values.astype(str))
            agg_cur = run_RRA(list_rankings)
            model_agg_info[model+'-'+mm] = agg_cur.index.values.astype(str)
            cur_mod_list_rankings.append(agg_cur.index.values.astype(str))
            cnt_cur_mod += 1
            if cnt_cur_mod >= AGG_NUM_PER_OM: break
        cur_mod_rra = run_RRA(cur_mod_list_rankings)
        # add model info
        for k, v in model_agg_info.items():
            cur_mod_rra[k] = pd.Index(v).get_indexer(cur_mod_rra.index)
            cur_mod_rra[k+'-'+'size'] = len(v)
        consensus_panel[TASK_NAME_CORRECTION_MAP_REVERSE[task]][mod] = cur_mod_rra
    # save
    info.reset_index()[['model','mm','top type','value']].to_csv(f"../../result/consensus_model_mm_info_{TASK_NAME_CORRECTION_MAP_REVERSE[task]}.csv")

#%%
# save
with open(RES_PATH + 'consensus_panel.pkl', 'wb') as f:
    pickle.dump(consensus_panel, f)
