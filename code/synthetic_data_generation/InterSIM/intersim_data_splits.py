###############################################################################
# import
###############################################################################
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pandas as pd
import numpy as np
import glob
import os

def extract_unique_file_name_bases(directory, pattern="*"):
    """
    Use glob to list matching files in directory, split each filename on the last underscore,
    and collect unique base strings.

    directory='/home/athan.li/eval_bk/data/synthetic/InterSIM'
    """
    base_names = set()
    search_pattern = os.path.join(directory, pattern)
    for filepath in glob.glob(search_pattern):
        if not os.path.isfile(filepath):
            continue
        filename = os.path.basename(filepath)
        # Remove the last underscore-separated segment
        if "_" in filename:
            base = filename.rsplit("_", 1)[0]
        else:
            base = filename
        base_names.add(base)
    return sorted(base_names)


uniq_names = extract_unique_file_name_bases(
    directory='/home/athan.li/eval_bk/data/synthetic/InterSIM'
)

########## NOTE
mods_comb = ['DNAm', 'mRNA', 'protein']
cnt = 0
##########

for uniq_name in uniq_names:

    ###############################################################################
    # load data
    ###############################################################################
    data = pd.read_csv(f"../../../data/synthetic/InterSIM/{uniq_name}_data.csv", index_col=0)
    label = pd.read_csv(f"../../../data/synthetic/InterSIM/{uniq_name}_label.csv", index_col=0).astype(str)

    mods_comb_str = '+'.join(mods_comb)

    print(f"Running model for {mods_comb}")
    
    assert (label.index==data.index).all()
    X = data
    y = label

    splits_df = pd.DataFrame(index=X.index, dtype=str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_val_index, test_index) in enumerate(skf.split(X, y)):
        X_trn_val = X.iloc[train_val_index].copy()
        y_trn_val = y.iloc[train_val_index].copy()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1/8, random_state=42)
        train_index_within_trainval, val_index_within_trainval = next(sss.split(X_trn_val, y_trn_val))
        
        X_trn, y_trn = X_trn_val.iloc[train_index_within_trainval], y_trn_val.iloc[train_index_within_trainval]
        X_val, y_val = X_trn_val.iloc[val_index_within_trainval], y_trn_val.iloc[val_index_within_trainval]
        X_tst, y_tst = X.iloc[test_index], y.iloc[test_index]

        train_index = np.arange(len(X))[X.index.isin(X_trn.index)]
        val_index = np.arange(len(X))[X.index.isin(X_val.index)]

        assert set(np.concatenate([train_index, val_index, test_index]))==set(np.arange(len(X)))
        
        splits = np.full(len(X), 'nan')
        splits[train_index] = 'trn'
        splits[val_index] = 'val'
        splits[test_index] = 'tst'
        assert splits[splits=='nan'].shape[0] == 0
        splits_df[f'fold{fold}'] = splits


        print("Trn class size:", y_trn["label"].value_counts())
        print("Val class size:", y_val["label"].value_counts())
        print("Tst class size:", y_tst["label"].value_counts())

        assert (X.columns==X_val.columns).all() and (X_trn.columns==X_val.columns).all()

        assert X_trn.isna().sum().sum() == 0, "X_trn has missing values"
        assert X_val.isna().sum().sum() == 0, "X_val has missing values"
        assert X_tst.isna().sum().sum() == 0, "X_tst has missing values"

    splits_df.to_csv(f"../../../data/synthetic/InterSIM/{uniq_name}_splits.csv")

print(f"Script successfully completed for InterSIM at cnt {cnt}.")
