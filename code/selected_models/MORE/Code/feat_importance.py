import os
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from .utils import load_model_dict 
from .models import init_model_dict
from .train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch
from .param import parameter_parser
cuda = True if torch.cuda.is_available() else False

def cal_feat_imp(
    data_trn,
    y_trn,
    data_tst,
    y_tst,
    num_class,
    model_dict,
    device
):
    import pandas as pd
    data = [pd.concat([data_trn[i], data_tst[i]], axis=0) for i in range(len(data_trn))]

    y = np.concatenate([y_trn, y_tst])
    labels_trte = y # 1 x N. can be binary or multi class. Ordering of samples the same as in data_trte_list
    # dict of 'tr' and 'te' as keys, and list of integers as indices. Indices are directly used with labels_trte, data_trte_list
    trte_idx = {'tr': list(np.arange(data_trn[0].shape[0])), 
                'te': list(np.arange(data_trn[0].shape[0], data_trn[0].shape[0] + data_tst[0].shape[0]))}
    if cuda:
        data_tensors = [torch.tensor(data[i].values).float().to(device) for i in range(len(data))]
    else:
        data_tensors = [torch.tensor(data[i].values).float() for i in range(len(data))]
    data_tr_list = [data_tensors[i][trte_idx['tr']] for i in range(len(data))] # list of tensors N_tr x D_1, N_tr x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_te_list = [data_tensors[i][trte_idx['te']] for i in range(len(data))] # list of tensors N_te x D_1, N_te x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_trte_list = data_tensors # list of tensors N_trte x D_1, N_trte x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    featname_list = [list(data[i].columns.values) for i in range(len(data))] # list of lists of strings. Each list corresponds to the feature names of a view

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx)
    
    dim_list = [x.shape[1] for x in data_tr_list]
    for m in model_dict:
        if cuda:
            model_dict[m].to(device)
    te_prob = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict)
    if num_class == 2:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
    else:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
    feat_imp_list = []

    import time
    st_time = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(device) # reset memory stats for the device

    for i in range(len(featname_list)):
        feat_imp = {"feat_name":featname_list[i]}
        feat_imp['imp'] = np.zeros(dim_list[i])
        for j in range(dim_list[i]):
            feat_tr = data_tr_list[i][:,j].clone()
            feat_te = data_te_list[i][:,j].clone()
            data_tr_list[i][:,j] = 0
            data_te_list[i][:,j] = 0
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx)
            te_prob = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict)
            if num_class == 2:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
            feat_imp['imp'][j] = (f1-f1_tmp)*dim_list[i]
            
            data_tr_list[i][:,j] = feat_tr.clone()
            data_te_list[i][:,j] = feat_te.clone()
            if j % 200 == 0:
                print(f"Feature importance calculation for view {i}, feature {j} done.")
        feat_imp_list.append(pd.DataFrame(data=feat_imp))

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")

    ed_time = time.perf_counter()
    print(f"BK identification running time: {ed_time - st_time} seconds")
    return feat_imp_list


def summarize_imp_feat(featimp_list_list, topn=50):
    num_rep = len(featimp_list_list)

    num_view = len(featimp_list_list[0])

    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    for r in range(num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
            df_featimp = pd.concat([df_featimp, df_tmp.copy(deep=True)], ignore_index=True)
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp',ascending=False)
    df_featimp_top = df_featimp_top.iloc[:topn]
    print('{:}\t{:}'.format('Rank','Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i+1,df_featimp_top.iloc[i]['feat_name']))