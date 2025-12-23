#%%
import pickle as pkl
from typing import Optional, Union, List
from sklearn.preprocessing import StandardScaler
import os
from utils import *

DATASET_CODE_MAP = {
    0: 'survival_BRCA',
    1: 'survival_LUAD',
    2: 'survival_COADREAD',
    3: 'drug_response_Cisplatin-BLCA',
    4: 'drug_response_Temozolomide-LGG'
}

DATA_PATH_MAP = {
    0 : './data/TCGA/survival_BRCA/survival_BRCA_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    1 : './data/TCGA/survival_LUAD/survival_LUAD_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    2 : './data/TCGA/survival_COADREAD/survival_COADREAD_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    3 : './data/TCGA/drug_response_Cisplatin-BLCA/drug_response_Cisplatin-BLCA_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    4 : './data/TCGA/drug_response_Temozolomide-LGG/drug_response_Temozolomide-LGG_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    'survival_BRCA' : './data/TCGA/survival_BRCA/survival_BRCA_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    'survival_LUAD' : './data/TCGA/survival_LUAD/survival_LUAD_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    'survival_COADREAD' : './data/TCGA/survival_COADREAD/survival_COADREAD_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    'drug_response_Cisplatin-BLCA' : './data/TCGA/drug_response_Cisplatin-BLCA/drug_response_Cisplatin-BLCA_CNV+DNAm+SNV+mRNA+miRNA.pkl',
    'drug_response_Temozolomide-LGG' : './data/TCGA/drug_response_Temozolomide-LGG/drug_response_Temozolomide-LGG_CNV+DNAm+SNV+mRNA+miRNA.pkl',
}

TRI_OMICS_COMBS = [ # tri-omics combinations with mRNA included
    ['DNAm', 'mRNA', 'miRNA'],
    ['CNV', 'mRNA', 'miRNA'],
    ['SNV', 'mRNA', 'miRNA'],
    ['DNAm', 'CNV', 'mRNA'],
    ['DNAm', 'SNV', 'mRNA'],
    ['CNV', 'SNV', 'mRNA'],
]

BK_PATH = './data/bk_set/processed/'

def convert_ft_score_to_gene_level(
    ft_score: pd.DataFrame,
    mode: int,
    omics_types: Optional[List[str]] = None,
):
    r"""
    Args:
        ft_score (pd.DataFrame): feature importance scores with index as feature names
            and a single column as scores.
        mode (int): indicates the mode of conversion.
            0: molecule-centric to gene-centric. The feature names in ft_score
                are on the original molecule level, e.g., 'cg00000029' for DNAm,
                'hsa-miR-100-5p' for miRNA. For mRNA, CNV, SNV, the feature names
                are assumed to be on gene level.
            1: gene-centric. The feature names in ft_score are already on gene level,
                for all modalities, e.g., 'DNAm@TP53', 'mRNA@KRAS'.
            2: gene-centric without modality information to gene-centric. The
                feature names in ft_score are only gene names, without modality
                information, e.g., 'TP53', 'KRAS'.
        omics_types (List[str], optional): list of omics types used in the model.
            Only required if mode == 2.
    Returns:
        ft (pd.DataFrame): feature importance scores on gene level,
            with index as gene names and a single column 'score'.
    """
    # rename ft_score's single column to 'score'
    ft = ft_score.copy()
    ft.columns = ['score']
    ft = ft.loc[~(ft.isna().values.flatten()), :]

    ##
    if mode == 2:
        assert omics_types is not None, "omics_types must be provided for mode 2."
        ft = pd.DataFrame(
            index = np.concatenate([
                [f"{mod}@{mol}" for mol in ft.index] for mod in omics_types
            ]),
            data = list(ft.values.flatten())*len(omics_types),
            columns=['score']
        )
    elif mode == 1:
        pass # no operation needed
    elif mode == 0:
        mmdic = mod_mol_dict(ft.index)
        fts = []
        fts.append(ft.loc[~np.isin(mmdic['mods'], ['DNAm', 'miRNA'])])
        if 'miRNA' in mmdic['mods_uni']:
            df_mirna = R2G.loc[np.intersect1d(mmdic['mols'][mmdic['mods']=='miRNA'], R2G.index), ['gene']]
            mirna_ft = ft.loc[mmdic['mods']=='miRNA']
            mirna_ft.index = mmdic['mols'][mmdic['mods']=='miRNA']
            df_mirna['score'] = mirna_ft.loc[df_mirna.index].values.flatten()
            df_mirna = df_mirna.groupby('gene').mean() #NOTE
            df_mirna.index = 'miRNA@' + df_mirna.index
            fts.append(df_mirna)
        if 'DNAm' in mmdic['mods_uni']:
            df_cpg = C2G.loc[mmdic['mols'][mmdic['mods']=='DNAm'], ['gene']]
            cpg_ft = ft.loc[mmdic['mods']=='DNAm']
            cpg_ft.index = mmdic['mols'][mmdic['mods']=='DNAm']
            df_cpg['score'] = cpg_ft.loc[df_cpg.index].values.flatten()
            df_cpg = df_cpg.groupby('gene').mean() #NOTE
            df_cpg.index = 'DNAm@' + df_cpg.index
            fts.append(df_cpg)
        ft = pd.concat(fts, axis=0)
    else:
        raise NotImplementedError("Mode not implemented.")
    ft = ft.loc[ft.isna().values.flatten()==False, :]
    ft = ft.loc[np.sort(ft.index)].astype(float)
    # max pooling
    ft.index = ft.index.str.split('@').str[1]
    ft = ft.groupby(ft.index, sort=False).max() # max scores for genes
    return  ft

def run_benchmark(
    run_method_custom_func,
    omics_types : Optional[Union[List[str], List[List[str]]]] = None,
    fold_to_run: Optional[Union[str, int, List[Union[str, int]]]] = None,
    datasets_to_run : Optional[Union[int, str, List[Union[int, str]]]] = None,
    surv_op: Optional[str] = 'binary',
    scaling: Optional[str] = 'standard',
    res_save_path: Optional[str] = './result/',
    plot: Optional[bool] = True,
):
    r""" Run the benchmark pipeline with a custom function to run a specific method.
    
    Args:
        custom_func: function handle to the custom method to be benchmarked.
            The function should have the signature as run_method_custom.
        omics_types: list of omics types to use, e.g., ['DNAm', 'mRNA', 'miRNA'],
            or a list of such lists for multiple combinations.
            Defaults to None, which uses all tri-omics types (with mRNA included).
        dataset: indicates which dataset to run the benchmark on. It can be an
            integer code (0-4), a string name, or a list of integer codes or string names.
            If None (default), runs on all datasets.
        fold_to_run: which fold(s) to run. If None (default), runs all folds.
            If an integer or string, runs that specific fold. If a list, runs the specified folds.
        surv_op: for survival datasets, indicates the type of survival time labels.
            If 'binary' (default), converts survival times into binary labels
                based on median survival time.
            Otherwise, uses the original continuous survival times (in days),
                where the method should handle survival analysis,
                with censoring information included. Specifically, y_trn, y_val,
                and y_tst will be DataFrames with columns 'T' (survival time)
                and 'E' (event indicator: 1 if event occurred, 0 if censored).
        scaling: method for scaling the data. Currently only 'standard' (z-score) 
            or 'minmax' (min-max normalization; 0-1 scaling) is implemented. If
            None, no scaling is applied, and mRNA, miRNA data will be left as 
            log-transformed values, DNAm data will be left as beta values,
            CNV and SNV data will be left as raw values. Default is 'standard'.
        res_save_path: path to save the benchmark results. Default is './result/'.
        plot: whether to plot the benchmark results. Default is True.
    
    Returns:
        acc_res: accuracy results dictionary.
            {metric_name: {(dataset_name, omics_comb, fold): ft_score, ...}, ...}
        sta_res: stability results dictionary.
            {metric_name: {(dataset_name, omics_comb): score, ...}, ...}
    """
    # check if mode in run_method_custom_func is defined
    import inspect
    signature = inspect.signature(run_method_custom_func)
    if 'mode' not in signature.parameters:
        raise ValueError("The custom function must have a 'mode' parameter specified \
                         as either 0, 1, or 2.")
    else:
        mode = signature.parameters['mode'].default

    ft_score_res = {}
    ####################### run benchmark #########################
    if datasets_to_run is None:
        datasets_to_run = [0,1,2,3,4]
    elif isinstance(datasets_to_run, list):
        datasets_to_run = datasets_to_run
    else:
        datasets_to_run = [datasets_to_run]

    for dataset in datasets_to_run:
        print("=============================================================")
        print("Running benchmark on dataset:", DATASET_CODE_MAP[dataset])
        print("=============================================================")
        data_path = DATA_PATH_MAP[dataset]
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        task = DATASET_CODE_MAP[dataset].split('_')[0]
        # mkdir for DATASET_CODE_MAP[dataset] if not exists
        save_dir = os.path.join(res_save_path, DATASET_CODE_MAP[dataset])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if omics_types is None:
            omics_combs_to_run = TRI_OMICS_COMBS
        elif isinstance(omics_types, list) and all(isinstance(ot, str) for ot in omics_types):
            omics_combs_to_run = [omics_types]
        elif isinstance(omics_types, list) and all(isinstance(ot, list) for ot in omics_types):
            omics_combs_to_run = omics_types
        else:
            raise ValueError("omics_types should be None, a list of strings, or a list of list of strings.")
        for omics_comb in omics_combs_to_run:
            print("----------------------------------------")
            print("Using omics types:", omics_comb)

            for fold in range(5):
                print("----------------------------------------")
                print("Fold:", fold)
                if fold_to_run is not None:
                    if isinstance(fold_to_run, list):
                        fold_to_run_strs = [str(f) for f in fold_to_run]
                        if str(fold) not in fold_to_run_strs:
                            print(f"Skipping fold {fold} as it is not in fold_to_run.")
                            continue
                    else:
                        assert isinstance(fold_to_run, (str, int)), "fold_to_run should be str or int"
                        if str(fold) != str(fold_to_run):
                            print(f"Skipping fold {fold} as it is not equal to fold_to_run {fold_to_run}.")
                            continue
            
                mmdic = mod_mol_dict(data['X'].columns)
                for i in range(len(omics_comb)):
                    print("Number of {cur_mod} features:".format(cur_mod=omics_comb[i]), (mmdic['mods'] == omics_comb[i]).sum())
                X = data['X'].loc[:, np.isin(mmdic['mods'], omics_comb)]
                y = data['y']

                ## splitting and preparing labels
                splits = data[f'fold{fold}']
                X_trn = X.loc[splits=='trn'].copy()
                X_val = X.loc[splits=='val'].copy()
                X_tst = X.loc[splits=='tst'].copy()
                if task=='survival':
                    y_trn = y.loc[splits=='trn']
                    y_val = y.loc[splits=='val']
                    y_tst = y.loc[splits=='tst']
                    print("Sample size:", len(y))
                    if surv_op == 'binary':
                        y_trn = (y_trn['T'] > np.median(y_trn['T'])).map({True: 'long', False: 'short'}).to_frame().rename(columns={'T': 'label'})
                        y_val = (y_val['T'] > np.median(y_val['T'])).map({True: 'long', False: 'short'}).to_frame().rename(columns={'T': 'label'})
                        y_tst = (y_tst['T'] > np.median(y_tst['T'])).map({True: 'long', False: 'short'}).to_frame().rename(columns={'T': 'label'})
                    else:
                        # keep as is, with censoring info
                        pass
                else: # for drug response tasks
                    y_trn = pd.DataFrame(index=X_trn.index, columns=['label'], data=y[splits=='trn'])
                    y_val = pd.DataFrame(index=X_val.index, columns=['label'], data=y[splits=='val'])
                    y_tst = pd.DataFrame(index=X_tst.index, columns=['label'], data=y[splits=='tst'])
                if ('label' in y_trn.columns) and y_trn['label'].nunique() == 2: # if not survival conti'
                    print("Trn sample size:", np.unique(y_trn, return_counts=True))
                    print("Val sample size:", np.unique(y_val, return_counts=True))
                    print("Tst sample size:", np.unique(y_tst, return_counts=True))

                ## data preprocessing and scaling
                mmdic = mod_mol_dict(X.columns)
                # for 'CNV'
                if 'CNV' in mmdic['mods_uni']:
                    mask = mmdic['mods'] == 'CNV'
                    # capping by 2
                    X_trn.loc[:, mask] = X_trn.loc[:, mask].apply(lambda x: np.minimum(x - 2, 2))
                    X_val.loc[:, mask] = X_val.loc[:, mask].apply(lambda x: np.minimum(x - 2, 2))
                    X_tst.loc[:, mask] = X_tst.loc[:, mask].apply(lambda x: np.minimum(x - 2, 2))
                # for 'mRNA', 'miRNA', 'DNAm'
                print("Scaling data using {scaling} scaling...".format(scaling=scaling))
                if scaling == 'standard':
                    scaler = StandardScaler()
                    mask = np.isin(mmdic['mods'], ['mRNA', 'miRNA', 'DNAm'])
                    if np.sum(mask) > 0:
                        X_trn.loc[:, mask] = scaler.fit_transform(X_trn.loc[:, mask])
                        X_val.loc[:, mask] = scaler.transform(X_val.loc[:, mask])
                        X_tst.loc[:, mask] = scaler.transform(X_tst.loc[:, mask])
                elif scaling == 'minmax':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    mask = np.isin(mmdic['mods'], ['mRNA', 'miRNA', 'DNAm'])
                    if np.sum(mask) > 0:
                        X_trn.loc[:, mask] = scaler.fit_transform(X_trn.loc[:, mask])
                        X_val.loc[:, mask] = scaler.transform(X_val.loc[:, mask])
                        X_tst.loc[:, mask] = scaler.transform(X_tst.loc[:, mask])
                elif scaling is None:
                    pass
                else:
                    raise ValueError("scaling method not recognized.")
                assert X_trn.isna().sum().sum() == 0, "X_trn has missing values"
                assert X_val.isna().sum().sum() == 0, "X_val has missing values"
                assert X_tst.isna().sum().sum() == 0, "X_tst has missing values"

                ## run method
                print("\nRunning custom method...\n")
                ft_score = run_method_custom_func(
                    X_train = X_trn,
                    y_train = y_trn,
                    X_val = X_val,
                    y_val = y_val,
                    X_test = X_tst,
                    y_test = y_tst
                )

                # save return_vals to a pkl file
                save_path = os.path.join(save_dir, f'ft_score_fold{fold}.csv')
                ft_score.to_csv(save_path)

                omics_comb_str = '+'.join(list(np.sort(omics_comb)))
                ft_score_res[(DATASET_CODE_MAP[dataset], omics_comb_str, fold)] = ft_score

    ############################### evaluate #########################
    print()
    print("=============================================================")
    print("Evaluating benchmark results...")
    print("=============================================================")
    # load gold bks
    surv_bk = pd.read_csv(BK_PATH + "survival_task_bks.csv", index_col=0)
    drug_bk = pd.read_csv(BK_PATH + "drug_response_task_bks.csv", index_col=0)
    gold_bks = pd.concat([surv_bk, drug_bk], axis=0)
    ###### accuracy evaluation ######
    score_ndcg_res = {}
    score_rr_res = {}
    score_ar_res = {}
    pval_mw_res = {}
    acc_res = {}
    for key in ft_score_res.keys():
        dataset_name, omics_comb_str, fold = key
        omics_comb = omics_comb_str.split('+')
        ft_score = ft_score_res[key].copy()
        print()
        print("-----------------------------------------")
        print("Evaluating accuracy for", dataset_name, "...")
        print("omics comb.:", omics_comb_str, "\tFold:", fold)
        print()
        # get bk for the current task        
        bks = gold_bks.loc[gold_bks['Task']==dataset_name, 'Gene'].unique().astype(str)
        # convert ft_score to gene level
        ft_score = convert_ft_score_to_gene_level(
            ft_score = ft_score,
            mode = mode,
            omics_types = omics_comb
        )
        # calculate accuracy metrics
        score_ndcg, score_rr, score_ar, pval_mw = evaluate_accuracy(ft_score = ft_score, bk = bks)
        # print("### Accuracy ###")
        # print("NDCG:", score_ndcg)
        # print("RR:", score_rr)
        # print("AR:", score_ar)
        # print("MW p-value:", pval_mw)
        score_ndcg_res[key] = score_ndcg
        score_rr_res[key] = score_rr
        score_ar_res[key] = score_ar
        pval_mw_res[key] = pval_mw
    acc_res['NDCG'] = score_ndcg_res
    acc_res['RR'] = score_rr_res
    acc_res['AR'] = score_ar_res
    acc_res['MW_pval'] = pval_mw_res

    ###### stability evaluation ######
    sta_res = {}
    score_kendall_res = {}
    score_rbo_res = {}
    score_psd_res = {}
    for dataset in datasets_to_run:
        task = DATASET_CODE_MAP[dataset]
        # get bk for the current task
        bks = gold_bks.loc[gold_bks['Task']==task, 'Gene'].unique().astype(str)
        print()
        print("-----------------------------------------")
        print("Evaluating stability for", task, "...")
        for omics_comb in omics_combs_to_run:
            rankings = []
            for fold in range(5):
                # print("----------------------------------------")
                # print("Fold:", fold)
                # print("----------------------------------------")
                omics_comb_str = '+'.join(list(np.sort(omics_comb)))
                key = (task, omics_comb_str, fold)
                if key not in ft_score_res:
                    print(f"Skipping omics_comb {omics_comb} - fold {fold} for task {task}, as no results scores are found.")
                    continue
                if fold_to_run is not None:
                    if isinstance(fold_to_run, list):
                        fold_to_run_strs = [str(f) for f in fold_to_run]
                        if str(fold) not in fold_to_run_strs:
                            print(f"Skipping fold {fold} as it is not in fold_to_run.")
                            continue
                    else:
                        assert isinstance(fold_to_run, (str, int)), "fold_to_run should be str or int"
                        if str(fold) != str(fold_to_run):
                            print(f"Skipping fold {fold} as it is not equal to fold_to_run {fold_to_run}.")
                            continue
                ft_score = ft_score_res[key].copy()
                # convert ft_score to gene level
                ft = convert_ft_score_to_gene_level(
                    ft_score = ft_score,
                    mode = mode,
                    omics_types = omics_comb
                )
                ft = ft.sort_values(by='score', ascending=False)
                # permute zero scores
                ranking = ft.index.values.astype(str)
                mask0 = ft.values.flatten()==0
                ranking[mask0] = np.random.permutation(ranking[mask0])
                rankings.append(ranking)
            # calculate stability metrics
            score_kendall, score_rbo, score_psd = evaluate_stability(
                rankings = rankings,
                bk = bks
            )
            score_kendall_res[(task, omics_comb_str)] = score_kendall
            score_rbo_res[(task, omics_comb_str)] = score_rbo
            score_psd_res[(task, omics_comb_str)] = score_psd
    sta_res['Kendall_tau'] = score_kendall_res
    sta_res['RBO'] = score_rbo_res
    sta_res['PSD'] = score_psd_res

    # save as pickle
    save_path = os.path.join(res_save_path, 'your_method_accuracy_results.pkl')
    with open(save_path, 'wb') as f:
        pkl.dump(acc_res, f)
    save_path = os.path.join(res_save_path, 'your_method_stability_results.pkl')
    with open(save_path, 'wb') as f:
        pkl.dump(sta_res, f)
    
    print("\nBenchmarking completed. Results saved to:", res_save_path)
    
    # plot
    if plot == True:
        plot_benchmark_results(
            acc_res=acc_res,
            sta_res=sta_res,
            omics_types=omics_types,
            fold_to_run=fold_to_run
        )

    return acc_res, sta_res
