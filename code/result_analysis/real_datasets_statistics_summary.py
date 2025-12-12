import pandas as pd
import numpy as np
MODS = ['CNV', 'DNAm', 'mRNA', 'miRNA', 'SNV']
tasks = [
    "survival_BRCA_binary",
    "survival_LUAD_binary",
    "survival_COADREAD_binary",
    "drug_response_Cisplatin-BLCA",
    "drug_response_Temozolomide-LGG",
]

stats_list = []

for task in tasks:
    # task = 'drug_response_Cisplatin'
    cur_mods_comb_str = "CNV+DNAm+SNV+mRNA+miRNA"
    ref = task + "_" + cur_mods_comb_str

    print("=============================================================")
    print("Task:", task)

    if 'binary' in task:
        with open(f"../../data/TCGA/{task.replace('_binary', '')}/{ref.replace('_binary', '')}.pkl", 'rb') as f:
            data = pd.read_pickle(f)
    else:
        with open(f"../../data/TCGA/{task}/{ref}.pkl", 'rb') as f:
            data = pd.read_pickle(f)

    X = data['X']
    y = data['y']

    if ('survival' in task) and ('binary' in task):
        y = (y['T'] > np.median(y['T'])).map({True: 'long', False: 'short'}).to_frame().rename(columns={'T': 'label'})
    y = pd.DataFrame(index=X.index, columns=['label'], data=y)

    class_sizes = y['label'].value_counts()
    print(class_sizes)

    print("Feature numbers:")
    for mod in MODS:
        print(mod, sum(X.columns.str.split("@").str[0]==mod))

    feat_counts = {mod: (X.columns.str.split("@").str[0] == mod).sum()
                for mod in MODS}
    row = {
        'task': task,
        'n_samples': len(y),
    }
    for cls, cnt in class_sizes.items():
        row[f'class_size_{cls}'] = cnt
    for mod, cnt in feat_counts.items():
        row[f'features_{mod}'] = cnt
    stats_list.append(row)

stats_df = (
    pd.DataFrame(stats_list)
    .set_index('task')
    .fillna(0)
    .astype(int)
    .sort_index(axis=1)
)
stats_df.to_csv("/home/athan.li/eval_bk/result/tcga_data_statistics_summary.csv")

