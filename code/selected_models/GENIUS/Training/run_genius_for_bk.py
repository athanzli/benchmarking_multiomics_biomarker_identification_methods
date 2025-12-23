#%%
##########################################################################################

import sys

import pandas as pd
import numpy as np
import time

from tqdm import tqdm

sys.path.append("../")

from ..GenomeImage.utils import make_image

import captum
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

##########################################################################################
import numpy as np
import torch
from torch.utils.data import DataLoader
from .Models.AE_Square import AE
import copy
##########################################################################################

import sys
sys.path.append('/home/athan.li/eval_bk/code/')
from utils import factorize_label, mod_mol_dict, modmol_gene_set_tcga, convert_omics_to_gene_level, P2G, C2G, SPLITTER, R2G


##########################################################################################
#%%
def run_genius(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device # device='cuda:3'
):
    y_trn = label_trn
    y_val = label_val
    y_tst = label_tst

    # default
    LR = 1e-4
    batch_size = 64
    lr_decay = 1e-6
    weight_decay = 1e-6
    epochs = 1000
    start_of_lr_decrease = 25

    print("LR:", LR)
    print("batch_size:", batch_size)
    print("lr_decay:", lr_decay)
    print("weight_decay:", weight_decay)
    print("epochs:", epochs)
    print("start_of_lr_decrease:", start_of_lr_decrease)

    # ascat_data = '../data/example_data/ascat.csv'
    all_genes_included = 'selected_models/GENIUS/data/example_data/all_genes_ordered_by_chr_no_sex_chr.csv'
    # mutation_data = '../data/example_data/muts.csv'
    # gene_exp_data = '../data/example_data/gene_exp_matrix.csv'
    # gene_methyl_data = '../data/example_data/methylation.csv'
    # print("Reading clinical...")
    # clinical = pd.read_csv(clinical_data)
    # clinical_format = clinical.copy()
    # print("Reading ascat...")
    # ascat = pd.read_csv(ascat_data)
    # ascat_loss = ascat.loc[ascat['loss'] == True]
    # ascat_gain = ascat.loc[ascat['gain'] == True]
    # print("Reading all gene definition...")
    all_genes = pd.read_csv(all_genes_included)
    # print("Reading Muts...")
    # muts = pd.read_csv(mutation_data)
    # print("Reading gene exp...")
    # gene_exp = pd.read_csv(gene_exp_data)
    # print("Reading Methylation...")
    # methy = pd.read_csv(gene_methyl_data)
    mdic = mod_mol_dict(data_trn.columns)
    omics_gset = modmol_gene_set_tcga(mod_mol_ids=data_trn.columns, op='union')
    
    # only lost 1 or 2.
    # from matplotlib_venn import venn2
    # GENIUS_gset = all_genes['name2'].unique()
    # venn2([set(gset), set(GENIUS_gset)], set_labels=('TCGA', 'GENIUS'))
    # intergs = np.intersect1d(gset, GENIUS_gset)
    # bk_set_remaining(intergs)

    clinical = pd.concat([y_trn, y_val, y_tst], axis=0)
    clinical['bcr_patient_barcode'] = clinical.index

    clinical_fact, _ = factorize_label(clinical['label'])
    clinical['label'] = clinical_fact

    data_cat = pd.concat([data_trn, data_val, data_tst], axis=0)
    data = {}
    for i in range(len(mdic['mods_uni'])):
        data[mdic['mods_uni'][i]] = data_cat.loc[:, mdic['mods'] == mdic['mods_uni'][i]]
        data[mdic['mods_uni'][i]].columns = data[mdic['mods_uni'][i]].columns.str.split(SPLITTER).str[1] # NOTE remove mod@ prefix

    ###########################################################################
    ### convert mols to gene-level
    ###########################################################################
    cover_gset = np.intersect1d(omics_gset, all_genes['name2'].unique()) # auto unique and sorted
    mods_gsets = [modmol_gene_set_tcga(mod_mol_ids=data_trn.columns[mdic['mods']==mdic['mods_uni'][i]], op='union') for i in range(len(mdic['mods_uni']))]
    for i in range(len(mdic['mods_uni'])):
        mods_gsets[i] = np.intersect1d(cover_gset, mods_gsets[i])

    data = convert_omics_to_gene_level(data, cover_gset)

    print("Building gimages...")
    start_time = time.perf_counter()
    # cnt=0
    images = {}
    for row in tqdm(clinical.itertuples(index=False), total=len(clinical)):
        id = row.bcr_patient_barcode
        met = row.label
        # st_time = time.perf_counter()
        image = make_image(id, met, all_genes)
        # print("Time for make_image:", time.perf_counter() - st_time)

        # st_time = time.perf_counter()
        # NOTE for eval_bk
        vals_mods = [data[mdic['mods_uni'][i]].loc[id, cover_gset] for i in range(len(mdic['mods_uni']))]
        for g, *mod_vals in zip(cover_gset, *vals_mods):
            cell = image.dict_of_cells[g]
            for idx, val in enumerate(mod_vals, start=1):
                setattr(cell, f'mod{idx}_val', val)
        # print("Time for setting values:", time.perf_counter() - st_time)

        # st_time = time.perf_counter()
        image.make_image_matrces()
        # print("Time for make_image_matrces:", time.perf_counter() - st_time)

        if len(mdic['mods_uni']) == 3:
            images['{}'.format(id)] = image.make_3_dim_image()
        elif len(mdic['mods_uni']) == 4:
            images['{}'.format(id)] = image.make_4_dim_image()
        else:
            raise ValueError("only 3 or 4 views are supported")

        # cnt+= 1
        # if cnt == 10:
            # break

    print("Done in --- %s minutes ---" % ((time.perf_counter() - start_time) / 60))

    # csv_file = config['meta_data']
    class TCGAImageLoader_for_evalbk(Dataset):
        def __init__(self, images, clinical):
            self.annotation = clinical
            self.images = images

        def __len__(self):
            return len(self.annotation)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            x = list(images.values())[idx]
            y = np.array(self.annotation.iloc[idx, 0], dtype="long")

            return x, y, self.annotation.iloc[idx, 1]

    dataset = TCGAImageLoader_for_evalbk(images, clinical)
    trn_idx = np.arange(len(y_trn))
    val_idx = np.arange(len(y_trn), len(y_trn) + len(y_val))
    tst_idx = np.arange(len(y_trn) + len(y_val), len(y_trn) + len(y_val) + len(y_tst))
    train_set, val_set, test_set = torch.utils.data.Subset(dataset, trn_idx), torch.utils.data.Subset(dataset, val_idx), torch.utils.data.Subset(dataset, tst_idx)
    ##########################################################################################

    ### class weights
    n_samples = clinical['label'].values[trn_idx]
    # we follow the same formula as in the original implementation but extends it to multi-class
    class_labels, counts = np.unique(n_samples, return_counts=True)
    print("Class labels:", class_labels)
    print("Class counts:", counts)
    C = len(class_labels)
    class_weights = len(n_samples) / (C * counts)
    class_weights = class_weights.astype(np.float32)
    print("Class weights:")
    print(class_weights)
    class_weights = torch.tensor(class_weights).to(device)

    trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=0, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=batch_size,num_workers=0, shuffle=True)

    num_views = len(mdic['mods_uni'])
    num_class = len(np.unique(clinical['label']))
    
    ###########################################################################
    #  train
    ###########################################################################
    torch.cuda.reset_peak_memory_stats(device) if device.startswith('cuda') else None
    net, cost_func = AE(output_size=num_class, image_channels=num_views), torch.nn.CrossEntropyLoss(weight=class_weights)
    net.to(device)

    optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
    lambda1 = lambda epoch: 0.99 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    best_loss = float("+Inf")
    best_model = None
    trigger_times = 0
    patience = 100 # NOTE

    def batch_train(x, y,net):
        net.train()
        y_hat,recon,L = net(x)
        loss = cost_func(y_hat, y)
        cost_func.zero_grad()
        loss.backward()
        optimizer.step()

    def batch_valid(x,net):
        with torch.no_grad():
            net.eval()
            y_hat,recon,L = net(x)
            return y_hat

    total_train_time = 0.0
    for ep in range(epochs):
        st_time = time.perf_counter()
        for x, y_dat, id in trainLoader:
            batch_train(x.to(device), y_dat.to(device),net)
        total_train_time += time.perf_counter() - st_time

        batch_y_val_true, batch_y_val_hat = torch.tensor([], device=device), torch.tensor([],device=device)
        for x, y_dat,id in valLoader:
            y_dat = y_dat.to(device)
            y_hat = batch_valid(x.to(device),net)
            batch_y_val_true = torch.cat((batch_y_val_true, y_dat), dim=0)
            batch_y_val_hat = torch.cat((batch_y_val_hat, y_hat), dim=0)

        if ep >= start_of_lr_decrease:
            scheduler.step()

        epoch_val_loss = cost_func(batch_y_val_hat, batch_y_val_true.to(torch.long))
        epoch_val_accuracy = torch.sum(torch.argmax(batch_y_val_hat, dim=1) == batch_y_val_true).item() / len(batch_y_val_true)

        print("Epoch:", ep, "Val Loss:", epoch_val_loss.item(), "Val Accuracy:", epoch_val_accuracy)

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = copy.deepcopy(net)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping at epoch: ", ep)
                break
    print(f"GENIUS model Training time for {ep} epochs: {total_train_time} seconds")

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")


    ################################ perf ################################
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
    y_prob = torch.tensor([], device='cpu')
    y_true = torch.tensor([], device='cpu')
    for x, y_dat, id in testLoader:
        y_hat = batch_valid(x.to(device),net)
        y_prob = torch.cat((y_prob, y_hat.to('cpu')), dim=0)
        y_true = torch.cat((y_true, y_dat.to('cpu')), dim=0)
    y_pred = torch.argmax(y_prob, dim=1).cpu().numpy()
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
        y_proba = y_prob[:, -1].to('cpu').detach().numpy()
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
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        print(f"MCC:            {mcc:.4f}")
        print(f"Balanced Acc:   {balanced_acc:.4f}")
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

    ################################ BK identification ################################
    torch.cuda.reset_peak_memory_stats(device) if device.startswith('cuda') else None

    attribution_n_steps = 50 # original: 10

    def wrapped_model(inp):
        return net(inp)[0]

    net = best_model.to(device)
    net.eval()

    occlusion = captum.attr.IntegratedGradients(wrapped_model)

    # iterate sample by sample
    all_samples = [test_set[i] for i in range(len(test_set))]
    x, y_dat, ids = zip(*all_samples)
    x = torch.tensor(x, device=device)
    y_dat = torch.tensor([y_dat[i].item() for i in range(len(y_dat))], device=device)
    batch_size = 1
    label_uni = np.unique(clinical['label'].values.flatten()).astype(int)

    #
    names = all_genes['name2'].values # using the fixed gene set for fixed image size instead of using cover gset
    ft_score = pd.DataFrame(
        index=np.concatenate([mdic['mods_uni'][i] + '@' + names for i in range(len(mdic['mods_uni']))]),
        columns=[f'score_{label_uni[i]}' for i in range(len(label_uni))])
    ## grid location to feature name mapping
    IMG_SIZE = 194
    # gridloc2gene = {}
    # genes_count = all_genes.shape[0]
    # for cnt, (i, j) in enumerate(np.ndindex(IMG_SIZE, IMG_SIZE)):
    #     if cnt < genes_count:
    #         gridloc2gene[(i, j)] = names[cnt]
    #     else:
    #         gridloc2gene[(i, j)] = ""
    # gridloc2gene_flatten = dict(zip(range(IMG_SIZE*IMG_SIZE), np.concatenate([names, np.array(['']*(IMG_SIZE*IMG_SIZE - names.shape[0]))])))

    start_time = time.perf_counter()
    
    for target in label_uni:
        print("IG processing class:", target)
        baseline = torch.zeros((1, *x.shape[1:]), device=device)

        attribution = occlusion.attribute(
            x, baseline, target=int(target), n_steps=int(attribution_n_steps),
            internal_batch_size=x.shape[0]*1) \
        .cpu().detach().numpy()

        for k in range(num_views):
            ft_names = mdic['mods_uni'][k] + '@' + names
            ft_score.loc[ft_names, f'score_{target}'] = attribution.mean(0)[k].flatten()[:len(names)]

    print("GENIUS BK identification time (s):", time.perf_counter() - start_time)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")

    tmp1 = ft_score.index.str.split(SPLITTER).str[0]
    tmp2 = ft_score.index.str.split(SPLITTER).str[1]
    mask = (tmp1==mdic['mods_uni'][0]) & (tmp2.isin(mods_gsets[0]))
    for i in range(len(mdic['mods_uni'])):
        mask |= (tmp1==mdic['mods_uni'][i]) & (tmp2.isin(mods_gsets[i]))
    ft_score = ft_score.loc[mask]

    return ft_score, perf
