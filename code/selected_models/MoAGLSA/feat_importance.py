import os
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, accuracy_score
from .utils import load_model_dict
from .models import init_model_dict
from .train_test import prepare_trte_data, test_epoch, gen_trte_adj_mat

cuda = True if torch.cuda.is_available() else False

def cal_feat_imp(
    data_trn,
    y_trn,
    data_tst,
    y_tst,
    num_class,
    model_dict,
    device,
    adj_parameter=8 # default as in train_test
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

    torch.cuda.reset_peak_memory_stats(device) if cuda else None

    adj_tr_list, adj_trte_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, device=device)

    dim_list = [x.shape[1] for x in data_tr_list]
    for m in model_dict:
        model_dict[m].to(device)

    te_prob = test_epoch(data_trte_list, adj_trte_list, trte_idx["te"], model_dict, adj_parameter=8, trte_idx=trte_idx, adaption=False)
    
    acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
    feat_imp_list = []

    for i in range(len(featname_list)):
        feat_imp = {"feat_name":featname_list[i]}
        feat_imp['imp'] = np.zeros(dim_list[i])
        for j in range(dim_list[i]):
            feat_tr = data_tr_list[i][:,j].clone()
            feat_trte = data_trte_list[i][:,j].clone()
            data_tr_list[i][:,j] = 0
            data_trte_list[i][:,j] = 0
            te_prob = test_epoch(data_trte_list, adj_trte_list, trte_idx["te"], model_dict, adj_parameter=None, trte_idx=None, adaption=None)
            acc_tmp = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            feat_imp['imp'][j] = (acc-acc_tmp)*dim_list[i]
            
            data_tr_list[i][:,j] = feat_tr.clone()
            data_trte_list[i][:,j] = feat_trte.clone()
            if j % 200 == 0: print('Finished feature {}/{} for view {}.'.format(j, dim_list[i], i))
        feat_imp_list.append(pd.DataFrame(data=feat_imp))

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")

    return feat_imp_list
