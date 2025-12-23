import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from .models import init_model_dict, init_optim
from .param import parameter_parser
from .utils import (Eu_dis, hyperedge_concat, generate_G_from_H, construct_H_with_KNN, one_hot_tensor, cal_sample_weight)
from .utils import save_model_dict
cuda = True if torch.cuda.is_available() else False

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score

def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_test_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx):

    H_tr = []
    H_te = []
    for i in range(len(data_tr_list)):
        H_1 = construct_H_with_KNN(data_tr_list[i], K_neigs=3, split_diff_scale=False, is_probH=True, m_prob=1)

        H_tr.append(H_1)

        H_2 = construct_H_with_KNN(data_te_list[i], K_neigs=3, split_diff_scale=False, is_probH=True, m_prob=1)
        H_te.append(H_2)

    H_train = hyperedge_concat(H_tr[0],H_tr[1],H_tr[2])

    H_test = hyperedge_concat(H_te[0],H_te[1],H_te[2])

    adj_train_list = generate_G_from_H(H_train, variable_weight=False)

    adj_test_list = generate_G_from_H(H_test, variable_weight=False)

    return adj_train_list, adj_test_list

def train_epoch(num_cls, data_list, adj_list, label, one_hot_label,
                sample_weight, model_dict, optim_dict, train_MOSA=True,
                class_weights=None):
    loss_dict = {}

    criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights)

    for m in model_dict:
      model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()

    if train_MOSA and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["E{:}".format(i+1)](data_list[i], adj_list))

        new_data = torch.cat([ci_list[0], ci_list[1], ci_list[2]], dim=1)

        c = model_dict["C"](new_data)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()


    return loss_dict
    

def test_epoch(num_cls, data_list, adj_list, te_idx, model_dict):

    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["E{:}".format(i+1)](data_list[i], adj_list))

    if num_view >= 2:

        new_data = torch.cat([ci_list[0], ci_list[1], ci_list[2]], dim=1)
        c = model_dict["C"](new_data)
    else:
        c = ci_list[0]


    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob

def get_test_output(num_cls, data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["E{:}".format(i+1)](data_list[i], adj_list))
    if num_view >= 2:
        new_data = torch.cat([ci_list[0], ci_list[1], ci_list[2]], dim=1)
        c = model_dict["C"](new_data)
    else:
        c = ci_list[0]
    return c

def train_test(
    data_trn,
    y_trn,
    data_val,
    y_val,
    data_tst,
    y_tst,
    view_list = [1,2,3],
    num_epoch_pretrain = 500,
    num_epoch = 1500,
    lr_e_pretrain = 1e-3,
    lr_e = 5e-4,
    lr_c = 1e-3,
    num_class = 5,
    device='cuda:7',
    class_weights=None
):
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    dim_he_list = [100, 10]

    data_tst0 = data_tst.copy()
    y_tst0 = y_tst.copy()
    data_tst = data_val.copy()
    y_tst = y_val.copy()

    import pandas as pd
    data = [pd.concat([data_trn[i], data_tst[i]], axis=0) for i in range(len(data_trn))]

    y = np.concatenate([y_trn, y_tst])
    labels_trte = y # 1 x N. can be binary or multi class. Ordering of samples the same as in data_trte_list
    # dict of 'tr' and 'te' as keys, and list of integers as indices. Indices are directly used with labels_trte, data_trte_list
    trte_idx = {'tr': list(np.arange(data_trn[0].shape[0])), 
                'te': list(np.arange(data_trn[0].shape[0], data_trn[0].shape[0] + data_tst[0].shape[0]))}
    data_tensors = [torch.tensor(data[i].values).float().to(device) for i in range(len(data))] # NOTE changed by using device
    data_tr_list = [data_tensors[i][trte_idx['tr']] for i in range(len(data))] # list of tensors N_tr x D_1, N_tr x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_te_list = [data_tensors[i][trte_idx['te']] for i in range(len(data))] # list of tensors N_te x D_1, N_te x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_trte_list = data_tensors # list of tensors N_trte x D_1, N_trte x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.to(device)
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)
        sample_weight_tr = sample_weight_tr.to(device)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx)
    dim_list = [x.shape[1] for x in data_tr_list]
    
    input_data_dim = [dim_he_list[-1], dim_he_list[-1], dim_he_list[-1]]
    hyperparam = {
        'dropout': 0.1,
        'n_hidden': 10,
        'n_head': 4,
        'nmodal': 3,
    }
    model_dict = init_model_dict(input_data_dim, hyperparam, num_view, num_class, dim_list, dim_he_list, dim_hvcdn, device=device)

    for m in model_dict:
        if cuda:
            model_dict[m].to(device)
                                                                           
    import time   
    print("\nPretrain MOHE...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    st_time = time.perf_counter()
    for epoch in range(num_epoch_pretrain):
        train_epoch(num_class, data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_MOSA=False,
                    class_weights=class_weights)
    print(f"MOHE Pre-Training time for {num_epoch_pretrain} epochs: {time.perf_counter() - st_time:.2f} seconds.")
    print("\nTraining...")
    import copy
    patience = 100
    no_imprv_count = 0
    best_loss = float('inf')
    best_model_dict = None
    torch.cuda.reset_peak_memory_stats(device)

    total_train_time = 0.0

    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        st_time = time.perf_counter()
        train_epoch(num_class, data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_MOSA=True,
                    class_weights=class_weights)

        ed_time = time.perf_counter()
        total_train_time += ed_time - st_time    

        with torch.no_grad():
            te_prob = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict)

            c_test = get_test_output(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict)
            te_labels_tensor = torch.LongTensor(labels_trte[trte_idx["te"]]).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            test_loss_value = criterion(c_test, te_labels_tensor)
            current_loss = test_loss_value.item()
        
            print(f"Epoch {epoch}, Test Loss: {current_loss:.3f}")

            if current_loss < best_loss:
                best_loss = current_loss
                no_imprv_count = 0
                best_model_dict = {k: copy.deepcopy(v.state_dict()) for k, v in model_dict.items()}
            else:
                no_imprv_count += 1
                if no_imprv_count >= patience:
                    print(f"early stopping at epoch {epoch}.")
                    break

        if epoch % test_inverval == 0:
            te_prob = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()
    
    print(f"MORE Training time for {epoch} epochs: {total_train_time:.2f} seconds.")
    param_counts = {}
    total_params = 0
    for model_name, model in model_dict.items():
        model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[model_name] = model_param_count
        total_params += model_param_count
    print(f"MORE model parameters: {total_params}.")

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during BK training: {peak_mb:.1f} MB")

    if best_model_dict is not None:
        for k, v in best_model_dict.items():
            model_dict[k].load_state_dict(v)

    data_tst = data_tst0.copy()
    y_tst = y_tst0.copy()
    data = [pd.concat([data_trn[i], data_tst[i]], axis=0) for i in range(len(data_trn))]
    y = np.concatenate([y_trn, y_tst])
    labels_trte = y # 1 x N. can be binary or multi class. Ordering of samples the same as in data_trte_list
    # dict of 'tr' and 'te' as keys, and list of integers as indices. Indices are directly used with labels_trte, data_trte_list
    trte_idx = {'tr': list(np.arange(data_trn[0].shape[0])), 
                'te': list(np.arange(data_trn[0].shape[0], data_trn[0].shape[0] + data_tst[0].shape[0]))}
    data_tensors = [torch.tensor(data[i].values).float().to(device) for i in range(len(data))] # NOTE changed by using device
    data_tr_list = [data_tensors[i][trte_idx['tr']] for i in range(len(data))] # list of tensors N_tr x D_1, N_tr x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_te_list = [data_tensors[i][trte_idx['te']] for i in range(len(data))] # list of tensors N_te x D_1, N_te x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_trte_list = data_tensors # list of tensors N_trte x D_1, N_trte x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.to(device)
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)
        sample_weight_tr = sample_weight_tr.to(device)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx)
    # ---------------------------------------------------------------------

    te_prob = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict)

    print("====================================================================")
    ################################ perf ################################
    y_true = labels_trte[trte_idx["te"]]
    y_pred = te_prob.argmax(1)
    task_is_biclassif = te_prob.shape[1] == 2
    roc_auc = None
    aucpr = None
    recall = None
    precision = None
    f1 = None
    f1_weighted = None
    f1_macro = None
    acc = None
    mcc = None
    balanced_acc = None
    if task_is_biclassif:
        y_proba = te_prob[:, 1]
        roc_auc = roc_auc_score(y_true, y_proba)
        aucpr = average_precision_score(y_true, y_proba)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        print(f"AUC-ROC:        {roc_auc:.4f}")
        print(f"AUCPR:          {aucpr:.4f}")
        print(f"F1:  {f1:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
    else:
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        print(f"F1-weighted:    {f1_weighted:.4f}")
        print(f"F1-macro:       {f1_macro:.4f}")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy:       {acc:.4f}")
    perf = {
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'aucpr': aucpr,
        'mcc': mcc if task_is_biclassif else None,
        'balanced_acc': balanced_acc if task_is_biclassif else None,
    }
    #######################################################################

    return model_dict, perf

