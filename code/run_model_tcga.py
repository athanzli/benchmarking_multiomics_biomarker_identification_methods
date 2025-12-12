# python -u run_model_tcga.py --device "cuda:0" --task survival_COADREAD --survival_op binary --model_name DeepKEGG
# python -u run_model_tcga.py --device "cuda:1" --task drug_response_Temozolomide-LGG --fold_to_run 0 --mods_to_run CNV+mRNA+miRNA --model_name CustOmics

###############################################################################
# import
###############################################################################

import os
from utils import *
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
from models import *

# TODO NOTE
RESULT_PATH = "/home/athan.li/eval_bk/result/"
RESULT_PATH = "../result/"

MODS = ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA'] # ordered
MODS_COMB = [
    ['CNV', 'DNAm',        'mRNA'],
    ['CNV',         'SNV', 'mRNA'],
    ['CNV',                'mRNA', 'miRNA'],
    [       'DNAm', 'SNV', 'mRNA'],
    [       'DNAm',        'mRNA', 'miRNA'],
    [               'SNV', 'mRNA', 'miRNA'],
]

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str)
argparser.add_argument('--device', type=str) 
argparser.add_argument('--task', type=str)
argparser.add_argument('--survival_op', type=str, default='continuous')
argparser.add_argument('--scaling', type=str, default='standard')
argparser.add_argument('--bk_identification_only', type=int, default=0)
argparser.add_argument('--run_bk_identification', type=int, default=1)
argparser.add_argument('--use_dl_original_mods', type=bool, default=None, required=False)
argparser.add_argument('--mods_to_run', type=str, default=None, required=False)
argparser.add_argument('--fold_to_run', type=str, default=None, required=False)
args = argparser.parse_args()
model_name = args.model_name 
device = args.device
task = args.task
surv_op = args.survival_op
scaling = args.scaling
bk_identification_only = args.bk_identification_only
run_bk_identification = args.run_bk_identification

if __name__ == '__main__':
    ###############################################################################
    # load data
    ###############################################################################
    cur_data_mods_comb_str = '+'.join(MODS)

    with open(f'../data/TCGA/{task}/{task}_{cur_data_mods_comb_str}.pkl', 'rb') as   f:
        data = pickle.load(f)

    if task.split('_')[0] == 'survival':
        task = task + '_' + surv_op

    dir_path = f"{RESULT_PATH}{task}"
    os.makedirs(dir_path, exist_ok=True)

    ###############################################################################
    # run model for each mods_comb
    ###############################################################################
    mods_combs = [
        ['CNV', 'DNAm', 'mRNA'],
        ['CNV', 'SNV', 'mRNA'],
        ['CNV', 'mRNA', 'miRNA'],
        ['DNAm', 'SNV', 'mRNA'],
        ['DNAm', 'mRNA', 'miRNA'],
        ['SNV', 'mRNA', 'miRNA'],
    ]
    mods_combs_str=['+'.join(ml) for ml in mods_combs]

    cnt = 0
    for fold in range(5):

        print("Fold:", fold)

        for mods_comb in mods_combs:
            mods_comb_str = '+'.join(mods_comb)
            cnt += 1
            print("cnt:", cnt)

            # NOTE
            if (args.mods_to_run is not None) and (mods_comb_str!=args.mods_to_run): continue
            if (args.fold_to_run is not None):
                assert isinstance(args.fold_to_run, str), "fold_to_run should be a string"
                if (str(fold)!=args.fold_to_run): continue

            if ~(np.isin(mods_comb, cur_data_mods_comb_str.split('+')).all()):
                print(f"Skipping mods_comb {mods_comb_str} as it is not in the data.")
                continue
 
            print(f"Running model for {mods_comb}")
            mmdic = mod_mol_dict(data['X'].columns)
            for i in range(len(mods_comb)):
                print("Number of {cur_mod} features:".format(cur_mod=mods_comb[i]), (mmdic['mods'] == mods_comb[i]).sum())
            X = data['X'].loc[:, np.isin(mmdic['mods'], mods_comb)]
            y = data['y']
            
            ## TCGA data
            splits = data[f'fold{fold}']
            X_trn = X.loc[splits=='trn']
            X_val = X.loc[splits=='val']
            X_tst = X.loc[splits=='tst']
            if task.split("_")[0]=='survival':
                y_trn = y.loc[splits=='trn']
                y_val = y.loc[splits=='val']
                y_tst = y.loc[splits=='tst']
                print("Sample size:", len(y))
                if surv_op == 'binary':
                    y_trn = (y_trn['T'] > np.median(y_trn['T'])).map({True: 'long', False: 'short'}).to_frame().rename(columns={'T': 'label'})
                    y_val = (y_val['T'] > np.median(y_val['T'])).map({True: 'long', False: 'short'}).to_frame().rename(columns={'T': 'label'})
                    y_tst = (y_tst['T'] > np.median(y_tst['T'])).map({True: 'long', False: 'short'}).to_frame().rename(columns={'T': 'label'})
            else:
                y_trn = pd.DataFrame(index=X_trn.index, columns=['label'], data=y[splits=='trn'])
                y_val = pd.DataFrame(index=X_val.index, columns=['label'], data=y[splits=='val'])
                y_tst = pd.DataFrame(index=X_tst.index, columns=['label'], data=y[splits=='tst'])
            if ('label' in y_trn.columns) and y_trn['label'].nunique() == 2:
                print("Trn sample size:", np.unique(y_trn, return_counts=True))
                print("Val sample size:", np.unique(y_val, return_counts=True))
                print("Tst sample size:", np.unique(y_tst, return_counts=True))

            ## preprocessing
            mmdic = mod_mol_dict(X.columns)
            # for 'CNV'
            if 'CNV' in mmdic['mods_uni']:
                mask = mmdic['mods'] == 'CNV'
                # NOTE capping by 2
                X_trn.loc[:, mask] = X_trn.loc[:, mask].apply(lambda x: np.minimum(x - 2, 2))
                X_val.loc[:, mask] = X_val.loc[:, mask].apply(lambda x: np.minimum(x - 2, 2))
                X_tst.loc[:, mask] = X_tst.loc[:, mask].apply(lambda x: np.minimum(x - 2, 2))
            # for 'mRNA', 'miRNA', 'DNAm'
            print("Scaling data using {scaling}.".format(scaling=scaling))
            if scaling == 'standard':
                scaler = StandardScaler()
                mask = np.isin(mmdic['mods'], ['mRNA', 'miRNA', 'DNAm'])
                if np.sum(mask) > 0:
                    X_trn.loc[:, mask] = scaler.fit_transform(X_trn.loc[:, mask])
                    X_val.loc[:, mask] = scaler.transform(X_val.loc[:, mask])
                    X_tst.loc[:, mask] = scaler.transform(X_tst.loc[:, mask])
            else:
                raise NotImplementedError

            assert X_trn.isna().sum().sum() == 0, "X_trn has missing values"
            assert X_val.isna().sum().sum() == 0, "X_val has missing values"
            assert X_tst.isna().sum().sum() == 0, "X_tst has missing values"

            # run model
            if model_name in MODEL_NAMES:
                if model_name in MOGONET_FAMILY_MODELS:
                    print("bk_identification_only:", bk_identification_only)
                    print("run_bk_identification:", run_bk_identification)
                    ft, perf = run_model(
                        model_name=model_name,
                        data_trn=X_trn,
                        label_trn=y_trn,
                        data_val=X_val,
                        label_val=y_val,
                        data_tst=X_tst,
                        label_tst=y_tst,
                        device=device,
                        bk_identification_only=bk_identification_only,
                        run_bk_identification=run_bk_identification,
                        save_model_suffix=f"{task}_fold{fold}_{mods_comb_str}")
                    if ft is not None:
                        ft.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    if perf is not None:
                        with open (f"{RESULT_PATH}{task}/perf_{model_name}_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                            pickle.dump(perf, f)
                elif model_name in ['ttest', 'mannwhitneyu','DPM']:
                    ft = run_model(
                        model_name=model_name,
                        data=X_trn,
                        label=y_trn)
                    ft.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                elif model_name in ['SVM_RFE', 'SVM_ONE', 'RF_VI', 'RF_RFE']:
                    ft, perf = run_model(
                        model_name=model_name,
                        data=X_trn,
                        label=y_trn,
                        data_tst=X_tst,
                        label_tst=y_tst)
                    ft.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    with open (f"{RESULT_PATH}{task}/perf_{model_name}_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
                elif model_name == 'SVM_ONE_and_SVM_RFE':             
                    fts, perf = run_model(
                        model_name=model_name,
                        data=X_trn,
                        label=y_trn,
                        data_tst=X_tst,
                        label_tst=y_tst)
                    fts[0].to_csv(f"{RESULT_PATH}{task}/ft_SVM_ONE_{mods_comb_str}_fold{fold}.csv")
                    fts[1].to_csv(f"{RESULT_PATH}{task}/ft_SVM_RFE_{mods_comb_str}_fold{fold}.csv")
                    with open (f"{RESULT_PATH}{task}/perf_SVM_ONE_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
                    with open (f"{RESULT_PATH}{task}/perf_SVM_RFE_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
                elif model_name == 'RF_VI_and_RF_RFE':
                    fts, perf = run_model(
                        model_name=model_name,
                        data=X_trn,
                        label=y_trn,
                        data_tst=X_tst,
                        label_tst=y_tst)
                    fts[0].to_csv(f"{RESULT_PATH}{task}/ft_RF_VI_{mods_comb_str}_fold{fold}.csv")
                    fts[1].to_csv(f"{RESULT_PATH}{task}/ft_RF_RFE_{mods_comb_str}_fold{fold}.csv")
                    with open (f"{RESULT_PATH}{task}/perf_RF_VI_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
                    with open (f"{RESULT_PATH}{task}/perf_RF_RFE_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
                elif model_name=='Pathformer':
                    ft, perf = run_model(
                        model_name=model_name,
                        data_trn=X_trn,
                        label_trn=y_trn,
                        data_val=X_val,
                        label_val=y_val,
                        data_tst=X_tst,
                        label_tst=y_tst,
                        device=device,
                        run_bk_identification=run_bk_identification # TODO NOTE
                        )
                    if run_bk_identification==1:
                        ft.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    with open (f"{RESULT_PATH}{task}/perf_{model_name}_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
                elif model_name in ['DIABLO','asmPLSDA']:
                    data_trn = pd.concat([X_trn, X_val], axis=0)
                    label_trn = pd.concat([y_trn, y_val], axis=0)
                    ft, ft_rank, perf = run_model(
                        model_name=model_name,
                        data_trn=X_trn,
                        label_trn=y_trn,
                        data_tst=X_tst,
                        label_tst=y_tst)
                    ft.to_csv(f"{RESULT_PATH}{task}/ft_raw_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    ft_rank.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    with open (f"{RESULT_PATH}{task}/perf_{model_name}_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)                    
                elif model_name in ['Stabl','GDF']:
                    data_trn = pd.concat([X_trn, X_val], axis=0)
                    label_trn = pd.concat([y_trn, y_val], axis=0)
                    ft_score, perf = run_model(
                        model_name=model_name,
                        data_trn=data_trn,
                        label_trn=label_trn,
                        data_tst=X_tst,
                        label_tst=y_tst)
                    ft_score.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    with open (f"{RESULT_PATH}{task}/perf_{model_name}_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
                elif model_name == 'MOFA':
                    ft, ft_score_only = run_model(
                        model_name=model_name,
                        data=pd.concat([X_trn,  X_val], axis=0),
                        save_model_suffix=f"{task}_fold{fold}_{mods_comb_str}")
                    ft.to_csv(f"{RESULT_PATH}{task}/ft_raw_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    ft_score_only.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                elif model_name in ['GAUDI','MCIA']:
                    ft = run_model(
                        model_name=model_name,
                        data=pd.concat([X_trn,  X_val], axis=0))
                    ft.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                else:
                    ft, perf = run_model(
                        model_name=model_name,
                        data_trn=X_trn,
                        label_trn=y_trn,
                        data_val=X_val,
                        label_val=y_val,
                        data_tst=X_tst,
                        label_tst=y_tst,
                        device=device)
                    ft.to_csv(f"{RESULT_PATH}{task}/ft_{model_name}_{mods_comb_str}_fold{fold}.csv")
                    with open (f"{RESULT_PATH}{task}/perf_{model_name}_{mods_comb_str}_fold{fold}.pkl", 'wb') as f:
                        pickle.dump(perf, f)
            else:
                raise ValueError(f"model_name {model_name} not found")

    print("Script successfully completed for all combinations of mods and all folds.")
