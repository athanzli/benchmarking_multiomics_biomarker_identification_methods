##############################################################################
# using one device
##############################################################################
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score, precision_score, balanced_accuracy_score, matthews_corrcoef
import torch
# from param import parameter_parser
import torch.nn.functional as F
from .models import init_model_dict, init_optim
from .utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, cal_adj_mat_parameter
from .utils import GraphConstructLoss, normalize_adj

import time

cuda = True if torch.cuda.is_available() else False

def gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter, device):
    adj_metric = "cosine"
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric, device))
        adj_test_list.append(gen_adj_mat_tensor(data_te_list[i], adj_parameter_adaptive, adj_metric, device))
    return adj_train_list, adj_test_list

def compute_dot(WF):
    """
    Compute the full dot product WF * WF.T.
    WF is expected to have shape (num_features, featuresSelect).
    """
    return torch.mm(WF, WF.T)

def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, theta_smooth, theta_degree, theta_sparsity, neta, train_MOAM_OIRL=True, class_weights=None): # NOTE added class_weights
    loss_dict = {}
    for m in model_dict:
        model_dict[m].train()

    criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights)

    weight1 = list(model_dict['E1'].parameters())[0]
    weight2 = list(model_dict['E2'].parameters())[0]
    weight3 = list(model_dict['E3'].parameters())[0]
    WF_weight = [weight1, weight2, weight3]

    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0

        adj_train = model_dict["GL{:}".format(i + 1)](data_list[i])
        graph_loss = GraphConstructLoss(data_list[i], adj_train, adj_list[i], theta_smooth, theta_degree, theta_sparsity)
        final_adj = neta * adj_train + (1-neta) * adj_list[i]
        normalized_adj = normalize_adj(final_adj)

        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],normalized_adj))
        ci_loss = torch.mean(torch.mul(criterion(ci,label), sample_weight))

        '''inner product regularization'''
        new_WF_weight = torch.mm(WF_weight[i], WF_weight[i].T) 
        WF_L1_list = torch.norm(new_WF_weight, p=1)
        WF_L2_list = torch.pow(torch.norm(WF_weight[i], p=2), 2)
        WF_L12_loss = WF_L1_list - WF_L2_list

        tol_loss = ci_loss + graph_loss + 0.0001 * WF_L12_loss
        tol_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = tol_loss.detach().cpu().numpy().item()

    if train_MOAM_OIRL and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        GCN_list = []
        for i in range(num_view):
            adj_train = model_dict["GL{:}".format(i + 1)](data_list[i])
            final_adj = neta * adj_train + (1 - neta) * adj_list[i]
            normalized_adj = normalize_adj(final_adj)

            GCN_list.append(model_dict["E{:}".format(i+1)](data_list[i],normalized_adj))

        atten_data_list = model_dict["MOAM"](GCN_list)
        new_data = torch.cat([atten_data_list[0],atten_data_list[1],atten_data_list[2]],dim= 1)
        c = model_dict["OIRL"](new_data)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, adj_list, model_dict, neta):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        adj_test = model_dict["GL{:}".format(i + 1)](data_list[i])
        final_adj = neta * adj_test + (1 - neta) * adj_list[i]
        normalized_adj = normalize_adj(final_adj)

        ci_list.append(model_dict["E{:}".format(i+1)](data_list[i],normalized_adj))

    atten_data_list = model_dict["MOAM"](ci_list)
    new_data = torch.cat([atten_data_list[0], atten_data_list[1], atten_data_list[2]], dim=1)
    if num_view >= 2:
        c = model_dict["OIRL"](new_data)
    else:
        c = ci_list[0]

    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob

def train_test(
    data_trn,
    y_trn,
    data_val,
    y_val,
    data_tst,
    y_tst,
    device,
    class_weights,
    num_class,
    lr_e_pretrain,
    lr_e,
    lr_c, 
    num_epoch_pretrain,
    num_epoch,
    theta_smooth,
    theta_degree,
    theta_sparsity,
    neta,
    reg,
    view_list):

    test_inverval = 5
    adj_parameter = 8 # default
    mode = 'weighted-cosine'
    featuresSelect_list = [400, 400, 400] # default
    dim_he_list = [400,400] # default
    input_data_dim = [dim_he_list[-1], dim_he_list[-1], dim_he_list[-1]]
    num_view = len(view_list)

    data_tst0 = data_tst.copy()
    y_tst0 = y_tst.copy()
    data_tst = data_val.copy()
    y_tst = y_val.copy()

    print("Preparing data...")
    import pandas as pd
    data = [pd.concat([data_trn[i], data_tst[i]], axis=0) for i in range(len(data_trn))]

    y = np.concatenate([y_trn, y_tst])
    labels_trte = y # 1 x N. can be binary or multi class. Ordering of samples the same as in data_trte_list
    # dict of 'tr' and 'te' as keys, and list of integers as indices. Indices are directly used with labels_trte, data_trte_list
    trte_idx = {'tr': list(np.arange(data_trn[0].shape[0])), 
                'te': list(np.arange(data_trn[0].shape[0], data_trn[0].shape[0] + data_tst[0].shape[0]))}
    data_tensors = [torch.tensor(data[i].values).float().to(device) for i in range(len(data))] 
    data_tr_list = [data_tensors[i][trte_idx['tr']] for i in range(len(data))] # list of tensors N_tr x D_1, N_tr x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_te_list = [data_tensors[i][trte_idx['te']] for i in range(len(data))]
    data_trte_list = data_tensors

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    labels_tr_tensor = labels_tr_tensor.to(device)
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)
    sample_weight_tr = sample_weight_tr.to(device)
    print('Finished prep data.')

    print('Calculating adjacency matrix...')
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter, device)
    print("Finished calculating adjacency matrix.")

    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, input_data_dim, adj_parameter, mode, featuresSelect_list)
    for m in model_dict:
        model_dict[m].to(device)
    
    st_time = time.perf_counter()
    print("\nPretrain FSDGCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c, reg)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, theta_smooth, theta_degree, theta_sparsity, neta, train_MOAM_OIRL=False)
    print(f"MOGLAM Model Pretraininig time for {epoch} epochs is {time.perf_counter() - st_time} seconds.")

    print("\nTraining...")
    torch.cuda.reset_peak_memory_stats(device) # reset memory stats for the device
    # -------------------------------------------------------------------------
    # early stopping
    import copy
    patience = 100
    no_imprv_count = 0
    best_loss = float('inf')
    best_model_dict = None
    test_inverval = 5
    # -------------------------------------------------------------------------
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c,reg)
    total_train_time = 0.0
    for epoch in range(num_epoch+1):
        st_time = time.perf_counter()
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, theta_smooth, theta_degree, theta_sparsity, neta)
        total_train_time += time.perf_counter() - st_time
        
        # ---------------------------------------------------------------------
        # early stopping: compute loss
        with torch.no_grad():
            te_prob = test_epoch(data_te_list, adj_te_list, model_dict, neta)

            data_list = data_te_list
            adj_list = adj_te_list
            for m in model_dict:
                model_dict[m].eval()
            num_view = len(data_list)
            ci_list = []
            for i in range(num_view):
                adj_test = model_dict["GL{:}".format(i + 1)](data_list[i])
                final_adj = neta * adj_test + (1 - neta) * adj_list[i]
                normalized_adj = normalize_adj(final_adj)
                ci_list.append(model_dict["E{:}".format(i+1)](data_list[i],normalized_adj))
            atten_data_list = model_dict["MOAM"](ci_list)
            new_data = torch.cat([atten_data_list[0], atten_data_list[1], atten_data_list[2]], dim=1)
            if num_view >= 2:
                c = model_dict["OIRL"](new_data)
            else:
                c = ci_list[0]
            te_labels_tensor = torch.LongTensor(labels_trte[trte_idx["te"]]).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            test_loss_value = criterion(c, te_labels_tensor)
            current_loss = test_loss_value.item()
        
            if current_loss < best_loss:
                best_loss = current_loss
                no_imprv_count = 0
                best_model_dict = {k: copy.deepcopy(v.state_dict()) for k, v in model_dict.items()}
            else:
                no_imprv_count += 1
                if no_imprv_count >= patience:
                    print(f"early stopping at epoch {epoch}.")
                    break
        # ---------------------------------------------------------------------
        
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_te_list, adj_te_list, model_dict, neta)
            print("\nVal: Epoch {:d}".format(epoch))
            print("Val ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Val F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            print("Val F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
    print(f"MOGLAM Model Training time for {epoch} epochs:", total_train_time)

    param_counts = {}
    total_params = 0
    for model_name, model in model_dict.items():
        model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[model_name] = model_param_count
        total_params += model_param_count
    print(f"MOGLAM model parameters: {total_params}.")

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during traininng: {peak_mb:.1f} MB")

    # -------------------------------------------------------------------------
    # early stopping: load the best model in case early stopping was not triggered
    if best_model_dict is not None:
        for k, v in best_model_dict.items():
            model_dict[k].load_state_dict(v)
    # -------------------------------------------------------------------------
    
    ####################### on test data #########################
    data_tst = data_tst0.copy()
    y_tst = y_tst0.copy()
    data = [pd.concat([data_trn[i], data_tst[i]], axis=0) for i in range(len(data_trn))]
    y = np.concatenate([y_trn, y_tst])
    labels_trte = y # 1 x N. can be binary or multi class. Ordering of samples the same as in data_trte_list
    # dict of 'tr' and 'te' as keys, and list of integers as indices. Indices are directly used with labels_trte, data_trte_list
    trte_idx = {'tr': list(np.arange(data_trn[0].shape[0])), 
                'te': list(np.arange(data_trn[0].shape[0], data_trn[0].shape[0] + data_tst[0].shape[0]))}
    data_tensors = [torch.tensor(data[i].values).float().to(device) for i in range(len(data))] 
    data_tr_list = [data_tensors[i][trte_idx['tr']] for i in range(len(data))] # list of tensors N_tr x D_1, N_tr x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_te_list = [data_tensors[i][trte_idx['te']] for i in range(len(data))]
    data_trte_list = data_tensors # list of tensors N_trte x D_1, N_trte x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    labels_tr_tensor = labels_tr_tensor.to(device)
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)
    sample_weight_tr = sample_weight_tr.to(device)
    
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter, device)
    # ---------------------------------------------------------------------

    te_prob = test_epoch(data_te_list, adj_te_list, model_dict, neta)

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
        print(f"AUC-ROC:        {roc_auc:.4f}")
        print(f"AUCPR:          {aucpr:.4f}")
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"F1:  {f1:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"MCC: {mcc:.4f}")
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        print(f"Balanced ACC: {balanced_acc:.4f}")
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
        'mcc': mcc,
        'balanced_acc': balanced_acc
    }
    #######################################################################

    return model_dict, perf
