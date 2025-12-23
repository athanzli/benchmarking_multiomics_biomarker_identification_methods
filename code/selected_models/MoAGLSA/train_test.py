""" Training and testing of the MoAGL-SA
"""
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
import torch
import torch.nn.functional as F
from .models import init_model_dict, init_optim, init_scheduler
from .utils import KNN, cal_sample_weight,one_hot_tensor

torch.autograd.set_detect_anomaly(True)

cuda = True if torch.cuda.is_available() else False

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#INPUT DATA
def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    labels = np.concatenate((labels_tr, labels_te))
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    #single
    # data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(view_list[0]) + "_tr.csv"), delimiter=','))
    # data_te_list.append(np.loadtxt(os.path.join(data_folder, str(view_list[0]) + "_te.csv"), delimiter=','))

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    idx_dict = {}
    num = num_tr+num_te
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, num))
    print(idx_dict["tr"])
    print(idx_dict["te"])
    data_tr_newlist = []
    data_te_newlist = []
    for i in range(len(data_mat_list)):
        data_tr_newlist.append(data_mat_list[i][idx_dict["tr"]])
        data_te_newlist.append(data_mat_list[i][idx_dict["te"]])

    labels_newtr = labels[idx_dict["tr"]]
    labels_newte = labels[idx_dict["te"]]
    labels_all = np.concatenate((labels_newtr, labels_newte))

    data_tr_tensor_list = []
    data_te_tensor_list = []
    for i in range(num_view):
        data_tr_tensor_list.append(torch.FloatTensor(data_tr_newlist[i]))
        data_te_tensor_list.append(torch.FloatTensor(data_te_newlist[i]))
        if cuda:
            data_tr_tensor_list[i] = data_tr_tensor_list[i].cuda()
            data_te_tensor_list[i] = data_te_tensor_list[i].cuda()

    trte_dict = {}
    trte_dict["tr"] = list(range(num_tr))
    trte_dict["te"] = list(range(num_tr, (num_tr+num_te)))

    return data_tr_tensor_list, data_te_tensor_list, trte_dict, labels_all


#CONSINE_N
def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, device):
    adj_train_list = []
    adj_train_test_list = []
    for i in range(len(data_tr_list)):
        adj_train_list.append(KNN(data_tr_list[i], adj_parameter).to(device))
        adj_train_test_list.append(KNN(data_trte_list[i], adj_parameter).to(device))
    return adj_train_list, adj_train_test_list


#TRAIN
def train_epoch(data_list, adj_tr_list, label, onehot_labels_tr_tensor, model_dict, optim_dict, sample_weight,scheduler, train_Ml=True, class_weights=None):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)

    ci_list = []
    adp_list = []
    if train_Ml and num_view >= 2:
        optim_dict["H"].zero_grad()
        for i in range(num_view):
            adj_tr, gl_loss = model_dict["GL{:}".format(i + 1)](data_list[i], adj_tr_list[i])
            adp_list.append(adj_tr)
            out = model_dict["E{:}".format(i + 1)](data_list[i],  adp_list[i])
            adp_list.append( adj_tr_list[i])
            ci_list.append(out)

        z = model_dict["H"](num_view, ci_list, adp_list)
        c = model_dict["C"](z)
        c_loss_decoder = model_dict["D"](num_view, ci_list, z, adp_list)

        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight)) + 1e-3 * c_loss_decoder + 1e-3 * gl_loss

        c_loss.backward()
        optim_dict["H"].step()
        scheduler.step()
        loss_dict["H"] = c_loss.detach().cpu().numpy().item()
    return loss_dict

#TEST
def test_epoch(
        data_list,
        adj_trte_list,
        te_idx, model_dict, adj_parameter, trte_idx, adaption):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)

    ci_list = []
    adp_list = []

    for i in range(num_view):
        adj_trte, gl_loss = model_dict["GL{:}".format(i + 1)](data_list[i], adj_trte_list[i])
        adp_list.append(adj_trte) # NOTE adj_te changed to adj_trte
        x = model_dict["E{:}".format(i + 1)](data_list[i], adp_list[i])
        ci_list.append(x)

    if num_view >= 2:
        z = model_dict["H"](num_view, ci_list, adp_list)
        c = model_dict["C"](z)
        c = c[te_idx, :] 
    else:
        z = ci_list[0]
        c = model_dict["C1"](z)
        c = c[te_idx, :] 
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob

def train_test(
    data_trn,
    y_trn,
    data_val,
    y_val,
    data_tst,
    y_tst,
    view_list = [1,2,3],
    num_epoch = 2500,
    lr_e = 1e-3, # default
    lr_c = 1e-4, # default
    num_class = 5,
    adj_parameter = 8, # default
    dim_he_list = [400,200,100], # according to paper
    device='cuda:7',
    class_weights=None,
    adaption=False 
    ):
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
    data_tensors = [torch.tensor(data[i].values).float().to(device) for i in range(len(data))] # NOTE changed by using device
    data_tr_list = [data_tensors[i][trte_idx['tr']] for i in range(len(data))] # list of tensors N_tr x D_1, N_tr x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True
    data_trte_list = data_tensors # list of tensors N_trte x D_1, N_trte x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    labels_tr_tensor = labels_tr_tensor.to(device)
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)
    sample_weight_tr = sample_weight_tr.to(device)
    print('Finished prep data.')

    torch.cuda.reset_peak_memory_stats(device) # Reset peak memory stats for the device

    print("Computing adj...")
    if adaption:
        pass
    else:
        adj_tr_list, adj_trte_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, device)
    print("Finished computing adj.")

    print("Init...")

    dim_list = [x.shape[1] for x in data_tr_list]

    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list)
    for m in model_dict:
        model_dict[m].to(device)

    optim_dict = init_optim(num_view, model_dict, lr_c, lr_e)
    scheduler = init_scheduler(optim_dict["H"])
    train_loss = []
    import time
    print("\nTraining...")

    # -------------------------------------------------------------------------
    # early stopping
    import copy
    patience = 100 # NOTE
    no_imprv_count = 0
    best_loss = float('inf')
    best_model_dict = None
    test_inverval = 5
    # -------------------------------------------------------------------------
    total_train_time = 0.0
    for epoch in range(num_epoch + 1):
        st_time = time.perf_counter()
        loss = train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, onehot_labels_tr_tensor, model_dict, optim_dict,
                           sample_weight_tr,scheduler, train_Ml=True, class_weights=class_weights) # NOTE added class_weights
        ed_time = time.perf_counter()
        total_train_time += ed_time - st_time    
        train_loss.append(loss["H"])

        # ---------------------------------------------------------------------
        # early stopping: compute loss
        with torch.no_grad():
            te_prob = test_epoch(data_trte_list, adj_trte_list, trte_idx["te"], model_dict, adj_parameter, trte_idx, adaption)
            
            data_list = data_trte_list
            for m in model_dict:
                model_dict[m].eval()
            num_view = len(data_list)
            ci_list = []
            adp_list = []
            for i in range(num_view):
                adj_te, gl_loss = model_dict["GL{:}".format(i + 1)](data_list[i], adj_trte_list[i])
                adp_list.append(adj_te)
                x = model_dict["E{:}".format(i + 1)](data_list[i], adp_list[i])
                ci_list.append(x)
            if num_view >= 2:
                z = model_dict["H"](num_view, ci_list, adp_list)
                c = model_dict["C"](z)
                c = c[trte_idx["te"], :] 
            else:
                raise ValueError("num_view should be >= 2 for now")
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
                te_prob = test_epoch(data_trte_list, adj_trte_list, trte_idx["te"], model_dict, adj_parameter, trte_idx, adaption)

                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                f1weighted = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                f1macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

                print(" Epoch {:d}".format(epoch), "Val ACC: {:.5f}".format(acc),
                    "  F1 weighted: {:.5f}".format(f1weighted),
                    " F1 macro: {:.5f}".format(f1macro), "Train Loss:{:.5f}".format(train_loss[epoch]),
                    " Val Loss:{:.5f}".format(current_loss))

    print(f"MoAGLSA Training time for {epoch} epochs: {total_train_time:.2f} seconds")

    param_counts = {}
    total_params = 0
    for model_name, model in model_dict.items():
        model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[model_name] = model_param_count
        total_params += model_param_count
    print(f"MoAGLSA model parameters: {total_params}.")

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")
    # -------------------------------------------------------------------------
    # early stopping: load the best model in case early stopping was not triggered
    if best_model_dict is not None:
        for k, v in best_model_dict.items():
            model_dict[k].load_state_dict(v)
    # -------------------------------------------------------------------------

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
    data_trte_list = data_tensors # list of tensors N_trte x D_1, N_trte x D_2, ...  Can be on cpu, will be moved to gpu if cuda is True

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    labels_tr_tensor = labels_tr_tensor.to(device)
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)
    sample_weight_tr = sample_weight_tr.to(device)
    
    adj_tr_list, adj_trte_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, device)
    # ---------------------------------------------------------------------

    te_prob = test_epoch(data_trte_list, adj_trte_list, trte_idx["te"], model_dict, adj_parameter, trte_idx, adaption)

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

