###############################################################################
# import
###############################################################################

from models import *
from utils import *

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

import argparse

RESULT_PATH = "../result/" # TODO NOTE
# import os
# os.chdir('/home/athan.li/eval_bk/code/')
# RESULT_PATH = "/home/athan.li/eval_bk/result/" # TODO NOTE

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str)
argparser.add_argument('--device', type=str) 
argparser.add_argument('--bk_identification_only', type=int, default=0)
argparser.add_argument('--run_bk_identification', type=int, default=1)
argparser.add_argument('--n', type=int, required=True) # total sample size of two classes
argparser.add_argument('--p', type=float, required=True) # prop
argparser.add_argument('--s', type=str, required=True) # signal strength
args = argparser.parse_args()
model_name = args.model_name 
device = args.device
bk_identification_only = args.bk_identification_only
run_bk_identification = args.run_bk_identification

FOLDS = [4,3,2,1,0]
mods_comb = ['DNAm', 'mRNA', 'protein']

##################### 
total_samples=args.n
prop = args.p
effect = args.s
ref = 'survival_BRCA'
##################### 

base_name = f"InterSIM_ref={ref}_n={total_samples}_p.dmp={prop}_p.deg4p=0.1_shift={effect}"
###############################################################################
# load data
###############################################################################
# NOTE change path accordingly
# DATA_PATH = "/home/athan.li/eval_bk/"
DATA_PATH = "../"
data = pd.read_csv(DATA_PATH + f"data/synthetic/InterSIM/{base_name}_data.csv", index_col=0)
label = pd.read_csv(DATA_PATH + f"data/synthetic/InterSIM/{base_name}_label.csv", index_col=0).astype(str)
splits  = pd.read_csv(DATA_PATH + f"data/synthetic/InterSIM/{base_name}_splits.csv", index_col=0)

###############################################################################
# run model for each mods_comb
###############################################################################
mods_comb_str = '+'.join(mods_comb)
print(f"Running model for {mods_comb}")
mmdic = mod_mol_dict(data.columns)
for i in range(len(mods_comb)):
    print("Number of {cur_mod} features:".format(cur_mod=mods_comb[i]), (mmdic['mods'] == mods_comb[i]).sum())

assert (label.index==data.index).all()
assert (splits.index==data.index).all()
X = data
y = label

for fold in FOLDS:
    X_trn, y_trn = X.loc[splits['fold'+str(fold)]=='trn'], y.loc[splits['fold'+str(fold)]=='trn']
    X_val, y_val = X.loc[splits['fold'+str(fold)]=='val'], y.loc[splits['fold'+str(fold)]=='val']
    X_tst, y_tst = X.loc[splits['fold'+str(fold)]=='tst'], y.loc[splits['fold'+str(fold)]=='tst']

    print("Trn class size:", y_trn["label"].value_counts())
    print("Val class size:", y_val["label"].value_counts())
    print("Tst class size:", y_tst["label"].value_counts())

    assert (X.columns==X_val.columns).all() and (X_trn.columns==X_val.columns).all()
    
    ## preprocessing
    mmdic = mod_mol_dict(X.columns)
    print("Scaling data using {scaling}.".format(scaling='standard'))
    scaler = StandardScaler()
    X_trn.loc[:, :] = scaler.fit_transform(X_trn.loc[:, :])
    X_val.loc[:, :] = scaler.transform(X_val.loc[:, :])
    X_tst.loc[:, :] = scaler.transform(X_tst.loc[:, :])

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
                save_model_suffix=f"{base_name}")
            if ft is not None:
                ft.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
            if perf is not None:
                with open (f"{RESULT_PATH}InterSIM/perf_{model_name}_{base_name}_fold{fold}.pkl", 'wb') as f:
                    pickle.dump(perf, f)
        elif model_name in ['ttest', 'mannwhitneyu', 'DPM']:
            ft = run_model(
                model_name=model_name,
                data=X_trn,
                label=y_trn)
            ft.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
        elif model_name in ['SVM_RFE', 'SVM_ONE', 'RF_RFE', 'RF_VI']:
            ft, perf = run_model(
                model_name=model_name,
                data=X_trn,
                label=y_trn,
                data_tst=X_tst,
                label_tst=y_tst)
            ft.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
            with open (f"{RESULT_PATH}InterSIM/perf_{model_name}_{base_name}_fold{fold}.pkl", 'wb') as f:
                pickle.dump(perf, f)
        elif model_name in ['SVM_ONE_and_SVM_RFE']:
            fts, perf = run_model(
                model_name=model_name,
                data=X_trn,
                label=y_trn,
                data_tst=X_tst,
                label_tst=y_tst)
            fts[0].to_csv(f"{RESULT_PATH}InterSIM/ft_SVM_ONE_{base_name}_fold{fold}.csv")
            fts[1].to_csv(f"{RESULT_PATH}InterSIM/ft_SVM_RFE_{base_name}_fold{fold}.csv")
            with open (f"{RESULT_PATH}InterSIM/perf_SVM_{base_name}_fold{fold}.pkl", 'wb') as f:
                pickle.dump(perf, f)
        elif model_name in ['RF_VI_and_RF_RFE']:
            fts, perf = run_model(
                model_name=model_name,
                data=X_trn,
                label=y_trn,
                data_tst=X_tst,
                label_tst=y_tst)
            fts[0].to_csv(f"{RESULT_PATH}InterSIM/ft_RF_VI_{base_name}_fold{fold}.csv")
            fts[1].to_csv(f"{RESULT_PATH}InterSIM/ft_RF_RFE_{base_name}_fold{fold}.csv")
            with open (f"{RESULT_PATH}InterSIM/perf_RF_{base_name}_fold{fold}.pkl", 'wb') as f:
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
            ft.to_csv(f"{RESULT_PATH}InterSIM/ft_raw_{model_name}_{base_name}_fold{fold}.csv")
            ft_rank.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
            with open (f"{RESULT_PATH}InterSIM/perf_{model_name}_{base_name}_fold{fold}.pkl", 'wb') as f:
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
            ft_score.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
            with open (f"{RESULT_PATH}InterSIM/perf_{model_name}_{base_name}_fold{fold}.pkl", 'wb') as f:
                pickle.dump(perf, f)
        elif model_name == 'MOFA':
            ft, ft_score_only = run_model(
                model_name=model_name,
                data=pd.concat([X_trn,  X_val], axis=0),
                save_model_suffix=f"{base_name}")
            ft.to_csv(f"{RESULT_PATH}InterSIM/ft_raw_{model_name}_{base_name}_fold{fold}.csv")
            ft_score_only.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
        elif model_name in ['GAUDI','MCIA']:
            ft = run_model(
                model_name=model_name,
                data=pd.concat([X_trn,  X_val], axis=0))
            ft.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
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
            ft.to_csv(f"{RESULT_PATH}InterSIM/ft_{model_name}_{base_name}_fold{fold}.csv")
            with open (f"{RESULT_PATH}InterSIM/perf_{model_name}_{base_name}_fold{fold}.pkl", 'wb') as f:
                pickle.dump(perf, f)
    else:
        raise ValueError(f"model_name {model_name} not found")

print(f"Script successfully completed.")

