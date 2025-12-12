#%%
from .model import get_pnet  # assuming get_pnet is available from model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score, precision_score, matthews_corrcoef, balanced_accuracy_score
import copy
import sys
sys.path.append('/home/athan.li/eval_bk/code/') # TODO change accordingly
from utils import factorize_label, mod_mol_dict, modmol_gene_set_tcga, convert_omics_to_gene_level, P2G, C2G, R2G, SPLITTER

from captum.attr import DeepLift

def run_ftscoring(model, X, fact_dict, omics_order):
    assert (fact_dict is not None) and (omics_order is not None)
    model.eval()
    device = next(model.parameters()).device

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)
    
    baseline = torch.zeros_like(X, dtype=torch.float32)

    class ModelWrapper(nn.Module):
        def __init__(self, mdl):
            super().__init__(); self.mdl = mdl
        def forward(self, x):
            outs = self.mdl(x)
            return torch.stack(outs).mean(dim=0) # unweighted

    dl = DeepLift(ModelWrapper(model))

    n_features = X.shape[1]
    features_per_gene = len(omics_order) # NOTE
    assert n_features % features_per_gene == 0
    
    genes = list(model.genes)

    scores = pd.DataFrame(index=np.concatenate([[omics_order[i] + '@' + g for g in genes] for i in range(len(omics_order))]),
                          columns=[f'score_{l}' for l in fact_dict],
                          dtype=np.float32)

    # hub correction on gene deg
    fan_in = features_per_gene
    fan_out = np.sum(model.layers[0].SparseTF_layer.map, axis=1)
    deg   = fan_out + fan_in
    mu, sigma  = deg.mean(), deg.std()
    thr   = mu + 5*sigma
    penalty = np.repeat(np.where(deg > thr, 1.0/deg, 1.0), repeats=features_per_gene)

    for label, c in fact_dict.items():
        attr = dl.attribute(X, baselines=baseline, target=c)
        per_feat = attr.sum(dim=0).abs().cpu().detach().numpy()
        scores[f'score_{label}'] = per_feat * penalty

    return scores

class SCDataset(Dataset):
    """
    Adapted from Pathformer GitHub
    Citation:
    Xiaofan Liu, Yuhuan Tao, Zilin Cai, Pengfei Bao, Hongli Ma, Kexing Li, Mengtao Li, Yunping Zhu, Zhi John Lu, Pathformer: a biological pathway informed transformer for disease diagnosis and prognosis using multi-omics data, Bioinformatics, Volume 40, Issue 5, May 2024, btae316, https://doi.org/10.1093/bioinformatics/btae316
    """
    def __init__(self, data, labels):
        """
        data: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
        """
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y

    def __len__(self):
        return self.data.shape[0]

def run_pnet(
    data_trn, label_trn, data_val, label_val, data_tst, label_tst,
    device='cuda:6',
    ):
    """
    """
    assert len(np.unique(label_trn.values.flatten()))==2, "Only two classes are supported."
    ###########################################################################
    ### to dict and convert mols to gene-level, ensuring same gene features
    ###########################################################################
    omics_gset = modmol_gene_set_tcga(data_trn.columns, op='union', c2g=C2G, p2g=P2G, r2g=R2G)
    mdic = mod_mol_dict(data_trn.columns)
    # assert 'mRNA' in mdic['mods_uni'], "mRNA must be present"

    data_trn.columns = data_trn.columns.str.split(SPLITTER).str[1]
    data_trn = {k: data_trn.loc[:, mdic['mods'] == k] for k in mdic['mods_uni']}
    data_val.columns = data_val.columns.str.split(SPLITTER).str[1]
    data_val = {k: data_val.loc[:, mdic['mods'] == k] for k in mdic['mods_uni']}
    data_tst.columns = data_tst.columns.str.split(SPLITTER).str[1]
    data_tst = {k: data_tst.loc[:, mdic['mods'] == k] for k in mdic['mods_uni']}
    data_trn = convert_omics_to_gene_level(data_trn, omics_gset)
    data_val = convert_omics_to_gene_level(data_val, omics_gset)
    data_tst = convert_omics_to_gene_level(data_tst, omics_gset)
    print("Finished converting mols to gene-level.")

    gene_names = omics_gset
    n_genes = len(gene_names)
    n_features = sum([data_trn[k].shape[1] for k in data_trn.keys()])
    ###########################################################################

    assert (np.array(list(data_trn.keys()))==np.unique(list(data_trn.keys()))).all(), \
        "omic blocks no ordered correctly."

    # concatenate different omics, assumming orders aligned
    df_trn = pd.concat(list(data_trn.values()), axis=1)
    df_val = pd.concat(list(data_val.values()), axis=1)
    df_tst = pd.concat(list(data_tst.values()), axis=1)

    # sample ordering
    df_trn = df_trn.loc[label_trn.index]
    df_val = df_val.loc[label_val.index]
    df_tst = df_tst.loc[label_tst.index]
    
    X_trn = df_trn.values.astype(np.float32)
    X_val = df_val.values.astype(np.float32)
    X_tst = df_tst.values.astype(np.float32)
    y_trn, fact_dict = factorize_label(label_trn['label'])
    y_val = label_val['label'].map(fact_dict).values
    y_tst = label_tst['label'].map(fact_dict).values

    
    # data
    train_dataset = SCDataset(X_trn, y_trn)
    val_dataset   = SCDataset(X_val, y_val)
    test_dataset  = SCDataset(X_tst, y_tst)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # model
    n_hidden_layers = 5 
    dropout = [0.5] + [0.1] * (n_hidden_layers + 1) # c.f.  https://github.com/marakeby/pnet_prostate_paper/blob/master/train/params/P1000/pnet/crossvalidation_average_reg_10_tanh.py
    activation = 'tanh'
    direction = 'root_to_leaf'
    use_bias = True
    kernel_initializer = 'glorot_uniform'
    bias_initializer = 'zeros'
    batch_normal = False
    repeated_outcomes = True
    
    num_classes = len(np.unique(y_trn))
    
    ###################################################
    model = get_pnet(
        genes=gene_names,
        n_features=n_features,
        n_genes=n_genes,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout,
        num_classes=num_classes,
        activation=activation,
        direction=direction,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        batch_normal=batch_normal,
        repeated_outcomes=repeated_outcomes,
    )
    model.to(device)
    
    ### train
    # class weights
    n_samples = y_trn
    print(np.unique(n_samples, return_counts=True))
    class_labels, counts = np.unique(n_samples, return_counts=True)
    C = len(class_labels)
    class_weights = len(n_samples) / (C * counts)
    class_weights = class_weights.astype(np.float32)
    print(class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001) # as described in PNet paper.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.75) # drop by 0.25, c.f.  https://github.com/marakeby/pnet_prostate_paper/blob/master/train/params/P1000/pnet/crossvalidation_average_reg_10_tanh.py
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counts = 0
    
    # NOTE
    num_epochs = 1000
    patience = 100

    import time
    total_training_time = 0.0

    # train
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        st_time = time.perf_counter()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # NOTE binary classif only
            weight = (batch_y * class_weights[1] + (1-batch_y) * class_weights[0]).clone()

            optimizer.zero_grad()
            outputs = model(batch_x)
            assert isinstance(outputs, list)
            n_out = len(outputs) # 6 in total. c.f.  "n_outputs=n_hidden_layers + 1" https://github.com/marakeby/pnet_prostate_paper/blob/master/train/params/P1000/pnet/crossvalidation_average_reg_10_tanh.py
            loss_weights = [np.exp(i + 1) for i in range(n_out)] # c.f. [2, 7, 20, 54, 148, 400] from PNet github
            total_weight = sum(loss_weights)
            loss = sum(loss_weights[i] * nn.functional.binary_cross_entropy(outputs[i][:,1], batch_y.to(torch.float32), weight=weight, reduction='mean') \
                        / total_weight for i in range(n_out))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        scheduler.step()

        train_loss = running_train_loss / len(train_dataset)
        current_lr = optimizer.param_groups[0]['lr']
        total_training_time += time.perf_counter() - st_time

        # validation Loop
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)

                # NOTE binary classif only
                weight = (batch_y * class_weights[1] + (1-batch_y) * class_weights[0]).clone()
                assert isinstance(outputs, list)
                n_out = len(outputs)
                loss_weights = [1 for i in range(n_out)]
                total_weight = sum(loss_weights)
                loss = sum(loss_weights[i] * nn.functional.binary_cross_entropy(outputs[i][:,1], batch_y.to(torch.float32), weight=weight, reduction='mean') \
                            / total_weight for i in range(n_out))
                logits = sum(loss_weights[i] * outputs[i] for i in range(n_out)) / total_weight

                running_val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss = running_val_loss / len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, lr: {current_lr}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            early_stopping_counts = 0
        else:
            early_stopping_counts += 1
            if early_stopping_counts >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"PNet model Training Time for {epoch} epochs (s):", total_training_time)

    model.load_state_dict(best_model_state)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")

    # testing 
    model.eval()
    running_test_loss = 0.0
    test_preds = []
    test_labels = []
    all_probs = []  # to store predicted probabilities for AUC calculation
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)

            # NOTE binary classif only
            weight = (batch_y * class_weights[1] + (1-batch_y) * class_weights[0]).clone()
            assert isinstance(outputs, list)
            n_out = len(outputs)
            loss_weights = [1 for i in range(n_out)]
            total_weight = sum(loss_weights)
            loss = sum(loss_weights[i] * nn.functional.binary_cross_entropy(outputs[i][:,1], batch_y.to(torch.float32), weight=weight, reduction='mean') \
                        / total_weight for i in range(n_out))
            logits = sum(loss_weights[i] * outputs[i] for i in range(n_out)) / total_weight

            running_test_loss += loss.item() * batch_x.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    all_probs = np.vstack(all_probs)

    ################################ perf ################################
    y_true = test_labels
    y_pred = test_preds
    task_is_biclassif = np.unique(y_true).shape[0] == 2 # NOTE
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
        y_proba = all_probs[:, 1]
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
        print(f"MCC: {mcc:.4f}")
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
        'mcc' : mcc,
        'balanced_acc': balanced_acc
    }
    #######################################################################

    ###########################################################################
    # BK identification
    ###########################################################################

    torch.cuda.reset_peak_memory_stats(device)

    st_time = time.perf_counter()
    ft_score = run_ftscoring(model, torch.tensor(X_tst.astype(np.float32)).to(device), fact_dict=fact_dict, omics_order=mdic['mods_uni'])
    print("BK identification running time (s):", time.perf_counter() - st_time)

    cover_gset = gene_names

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")
    ## remove padded features
    mappings = {
        'DNAm' : C2G,
        'miRNA' : R2G,
        'protein' : P2G
    }
    to_rmv = []
    for mod in mdic['mods_uni']:
        if mod in ['DNAm','miRNA','protein']:
            inter_mols = np.intersect1d(mappings[mod].index.values.astype(str), mdic['mols'][mdic['mods']==mod])
            tmp = set(cover_gset) - set(mappings[mod].loc[inter_mols, 'gene'].unique())
        else:
            tmp = set(cover_gset) - set(mdic['mols'][mdic['mods']==mod])
        to_rmv.append([mod + '@' + tmp_mol for tmp_mol in tmp])
    to_rmv = np.concatenate(to_rmv)
    ft_score = ft_score.loc[~(ft_score.index.isin(to_rmv))]
    v1 = (~(ft_score.index.isin(to_rmv))).sum()
    v2 = len(ft_score)
    print(f"padding feature removal - kept {v1}/{v2}")

    return ft_score, perf
