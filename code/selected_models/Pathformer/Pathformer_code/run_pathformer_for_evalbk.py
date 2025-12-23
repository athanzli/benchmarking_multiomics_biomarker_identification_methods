import os

# TODO NOTE change acoordingly
import sys
# sys.path.append('/home/athan.li/eval_bk/code/')
from utils import factorize_label, mod_mol_dict, modmol_gene_set_tcga, P2G, C2G, R2G, SPLITTER
os.chdir("/home/athan.li/eval_bk/code/selected_models/Pathformer/Pathformer_code/")
# os.chdir("/workspace/eval_bk/code/selected_models/Pathformer/Pathformer_code/")

import pandas as pd
import numpy as np

import torch
from einops import repeat
from torch.optim import Adam
from torch.utils.data import DataLoader
import copy

import shap
import gc

from .Pathformer import pathformer_model
from .pathformer_utils import *

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef

def aggregate_mol(
        mapping_df: pd.DataFrame,
        mol_df: pd.DataFrame,
        mol_name: str,
        cover_gene_set) -> torch.Tensor:
    """
    Create modal embeddings.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, num_genes, num_modals)
        order is the same as cover_gene_set
    """
    mol_names = mol_df.columns.str.split(SPLITTER).str[1].values
    gene_to_mols = mapping_df.groupby("gene")[mol_name].apply(list).to_dict()
    available_genes = {}
    for gene, mol_list in gene_to_mols.items():
        valid_mols = [mol for mol in mol_list if mol in mol_names]
        if valid_mols:
            available_genes[gene] = valid_mols
    # per-gene aggregated feature arrays
    gene_feature_list = []
    # replace zeros with NaN (as in the original code) to avoid aggregation bias
    mol_df_proc = mol_df.replace(0, np.nan).astype(np.float32)
    num_samples = mol_df_proc.shape[0]
    mol_df_proc.columns = mol_df_proc.columns.str.split(SPLITTER).str[1]
    # for each gene in sorted order, compute aggregation if available; otherwise output zeros
    for gene in cover_gene_set:
        if gene in available_genes:
            mols = available_genes[gene]
            sub_df = mol_df_proc[mols]
            count_series = sub_df.notna().sum(axis=1)
            # if all values are NaN, fill with 0.0
            if (count_series == 0).any():
                mask = (sub_df.isna().sum(axis=1)==sub_df.shape[1]).values
                sub_df.loc[mask, :] = sub_df.loc[mask].fillna(0.0)
            max_series = sub_df.max(axis=1, skipna=True)
            min_series = sub_df.min(axis=1, skipna=True)
            mean_series = sub_df.mean(axis=1, skipna=True)
            gene_array = pd.concat([count_series, max_series, min_series, mean_series], axis=1).to_numpy().astype(np.float32)
        else:
            gene_array = np.zeros((num_samples, 4), dtype=np.float32)
        if np.isnan(gene_array).any():
            raise ValueError(f"NaN values found in gene {gene} aggregation.")
        gene_feature_list.append(gene_array)
    return np.stack(gene_feature_list, axis=1)

def data_to_data_with_modal_embedding(data, cover_gene_set):
    """
    Convert data to data with modal embedding.
    """
    mdic = mod_mol_dict(data.columns)
    num_modals = 0
    for omics_type in mdic['mods_uni']:
        if omics_type in ['DNAm', 'protein', 'miRNA']:
            num_modals += 4
        elif omics_type in ['mRNA', 'CNV', 'SNV']:
            num_modals += 1
        else:
            raise ValueError(f"Unsupported mod: {omics_type}")
    print("Number of modals:", num_modals)

    dwm_final = np.zeros((data.shape[0], len(cover_gene_set), num_modals), dtype=np.float32)
    pos = 0
    for omics_type in mdic['mods_uni']:
        if 'DNAm' == omics_type:
            dwm = aggregate_mol(
                mapping_df=C2G.rename(columns={'cpg.1': 'cpg'}).copy(),
                mol_df=data.loc[:, mdic['mods']=='DNAm'],
                mol_name='cpg',
                cover_gene_set=cover_gene_set)
            dwm_final[:, :, pos:(pos+4)] = dwm.astype(np.float32)
            pos += 4
        elif 'protein' == omics_type:
            dwm = aggregate_mol(
                mapping_df=P2G.rename(columns={'AGID.1': 'protein'}).copy(),
                mol_df=data.loc[:, mdic['mods']=='protein'],
                mol_name='protein',
                cover_gene_set=cover_gene_set)
            dwm_final[:, :, pos:(pos+4)] = dwm.astype(np.float32)
            pos += 4
        elif 'miRNA' == omics_type:
            dwm = aggregate_mol(
                mapping_df=R2G.copy(),
                mol_df=data.loc[:, mdic['mods']=='miRNA'],
                mol_name='miRNA',
                cover_gene_set=cover_gene_set)
            dwm_final[:, :, pos:(pos+4)] = dwm.astype(np.float32)
            pos += 4
        elif omics_type in ['mRNA', 'CNV', 'SNV']:
            data_tmp = data.loc[:, mdic['mods'] == omics_type]
            data_tmp.columns = data_tmp.columns.str.split(SPLITTER).str[1]
            assert data_tmp.columns.isin(cover_gene_set).all()
            data_tmp2 = pd.DataFrame(index=data.index, columns=cover_gene_set, data=0.0).astype(np.float32)
            data_tmp2.loc[:, data_tmp.columns] = data_tmp.astype(np.float32)
            dwm = np.array(torch.tensor(data_tmp2.values, device='cpu').unsqueeze(axis=2)).astype(np.float32)
            dwm_final[:, :, pos:(pos+1)] = dwm
            pos += 1
        else:
            raise ValueError(f"Unsupported mod: {omics_type}")
    return dwm_final

def run_pathformer(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:7',
    ############### pathformer default ############
    gene_select='../reference/Pathformer_select_gene.txt',
    pathway_gene_w='../reference/Pathformer_pathway_gene_weight.npy',
    pathway_crosstalk_network='../reference/Pathformer_pathway_crosstalk_network_matrix.npy',
    dataset=1,
    batch_size=16,
    gradient_num=3,

    epoch_num=2000,
    patience=100,

    depth=3,
    heads=8,
    dim_head=32,
    beta=1,
    attn_dropout=0.2,
    ff_dropout=0.2,
    classifier_dropout=0.3,
    lr_max=1e-5,
    lr_min=1e-8,
    classifier_dim=[300, 200, 100],
    bk_identification_only=0,
    run_bk_identification=1,
    save_model_suffix=None,
):
    """
    device='cuda:3'
    ############### pathformer default ############
    gene_select='../reference/Pathformer_select_gene.txt'
    pathway_gene_w='../reference/Pathformer_pathway_gene_weight.npy'
    pathway_crosstalk_network='../reference/Pathformer_pathway_crosstalk_network_matrix.npy'
    dataset=1
    batch_size=16
    gradient_num=3
    epoch_num=2000 # NOTE originally 2000
    patience=100 # NOTE changed from 10 to 100
    depth=3
    heads=8
    dim_head=32
    beta=1
    attn_dropout=0.2
    ff_dropout=0.2
    classifier_dropout=0.3
    lr_max=1e-5
    lr_min=1e-8
    classifier_dim=[300, 200, 100]
    bk_identification_only=0
    run_bk_identification=1
    save_model_suffix=None
    """
    mdic = mod_mol_dict(data_trn.columns)

    #######################
    setup_seed(2022)
    BATCH_SIZE = batch_size

    #############################
    #########Data load###########
    #############################
    ### load labels
    # label = pd.read_csv(label_path, sep='\t') # NOTE original
    label_trn['CaseID'] = label_trn.index
    label_val['CaseID'] = label_val.index
    label_tst['CaseID'] = label_tst.index
    label = pd.concat([label_trn, label_val, label_tst])
    label.index = np.arange(len(label))
    factorized_labels, value_to_index = factorize_label(label['label'])
    assert set(list(value_to_index.keys())).issubset(set(label_trn['label']))
    assert set(list(value_to_index.keys())).issubset(set(label_val['label']))
    assert set(list(value_to_index.keys())).issubset(set(label_tst['label']))
    label['label'] = factorized_labels
    label['y'] = label['label']
    label['dataset_1_new'] = [
        'train' if i in label_trn['CaseID'] else 'validation' if i in label_val['CaseID'] else 'test'
        for i in label['CaseID']
    ]
    train_sample=list(label.loc[label['dataset_' + str(dataset)+'_new'] == 'train', :].index)
    validation_sample=list(label.loc[label['dataset_' + str(dataset)+'_new'] == 'validation', :].index)
    train_label = label.loc[train_sample, ['y']].values
    train_label = train_label.astype(int)
    validation_label = label.loc[validation_sample, ['y']].values
    validation_label = validation_label.astype(int)
    test_sample = list(label.loc[label['dataset_' + str(dataset)+'_new'] == 'test', :].index)
    test_label = label.loc[test_sample, ['y']].values
    test_label = test_label.astype(int)

    ### select genes
    tcga_ginfo=pd.read_csv("../../../../data/TCGA/mRNA_gene_list.csv")
    assert tcga_ginfo['gene_id'].nunique() == tcga_ginfo.shape[0]
    tmp = tcga_ginfo['gene_id'].str.split('.').str[0]
    # dup_mask = tmp.isin(tmp[tmp.duplicated()])
    # dup = tcga_ginfo.loc[dup_mask]
    # dup['gene_id']=dup['gene_id'].str.split('.').str[0]
    # dup=dup.drop_duplicates()
    # assert (dup.groupby('gene_id').size()==1).all() # each ENSG ID in tmp (taking the part before '.') uniquely corresponds to one gene_name
    tcga_ginfo['gene_id'] = tmp
    tcga_ginfo = tcga_ginfo.drop_duplicates()
    tcga_ginfo.index = tcga_ginfo['gene_id']
    tcga_ginfo = tcga_ginfo.drop(columns=['gene_id'])
    # inter_gid = np.intersect1d(tcga_ginfo.index, gene_select_data[0].values) # has 11560
    pathformer_sel_genes = pd.read_csv(gene_select, header=None)[0].values # pathformer's selected gene set, 11560. '../reference/Pathformer_select_gene.txt'
    tmp = tcga_ginfo.loc[pathformer_sel_genes].copy()
    # there are two genes 'PDE11A', 'POLR2J3' with double gene_ids, the first gene_id ['ENSG00000128655', 'ENSG00000168255'] is selected to be used, the second gene_id ['ENSG00000284741', 'ENSG00000285437'] is removed from pathformer ref data; also removing non-protein coding genes
    print(tmp[tmp['gene_name'].duplicated(keep=False)])
    print(tmp.loc[tmp['gene_name'].isin(['PDE11A', 'POLR2J3'])])
    to_rmv1 = ['ENSG00000284741', 'ENSG00000285437'] # the second id in double gene_ids
    to_rmv2 = tmp.loc[tmp['gene_type']!='protein_coding'].index.unique().values # non-protein coding
    omics_gset = modmol_gene_set_tcga(data_trn.columns)
    to_rmv3 = np.array(list(
        set(pathformer_sel_genes) - 
        set(tcga_ginfo.loc[tcga_ginfo['gene_name'].isin(omics_gset)].index.unique().values))) # those in the pathformer gene set but not in  the input omics_gset 
    to_rmv = np.concatenate([to_rmv1, to_rmv2, to_rmv3])
    sel_from_pathformer_genes_mask = ~np.isin(pathformer_sel_genes, to_rmv)
    sel_genes = pathformer_sel_genes[sel_from_pathformer_genes_mask]
    sel_gene_names = tcga_ginfo.loc[sel_genes, 'gene_name'].values
    gene_select_index = np.arange(len(sel_genes)) # using full mask, since we construct data using only sel_genes
    assert len(sel_genes) == len(sel_gene_names)
    assert np.unique(sel_genes).shape[0] == len(sel_genes)
    assert np.unique(sel_gene_names).shape[0] == len(sel_gene_names)

    ### construct data modals
    data = pd.concat([data_trn, data_val, data_tst], axis=0)
    # CNV_count, CNV_max, CNV_min, CNV_mean,
    # DNAm_count, DNAm_max, DNAm_min, DNAm_mean,
    # SNV_count, SNV_max, SNV_min, SNV_mean,
    # mRNA,
    # protein_count, protein_max, protein_min, protein_mean
    # data0 = data.copy() # data = data0.copy()
    cover_gset = np.unique(sel_gene_names)
    assert len(cover_gset) == len(sel_genes)
    gset_tmp = modmol_gene_set_tcga(data.columns, op='union')
    assert set(cover_gset).issubset(set(gset_tmp))
    c2g = C2G.loc[C2G['gene'].isin(cover_gset)]
    p2g = P2G.loc[P2G['gene'].isin(cover_gset)]
    r2g = R2G.loc[R2G['gene'].isin(cover_gset)]
    # filter columns to keep those within cover_gset
    filtered_data = []
    for omics_type in mdic['mods_uni']:
        data_tmp = data.loc[:, mdic['mods'] == omics_type].copy()
        if omics_type in ['CNV', 'SNV', 'mRNA']:
            filtered_data.append(data_tmp.loc[:, data_tmp.columns.str.split(SPLITTER).str[1].isin(cover_gset)])
        elif omics_type in ['DNAm']:
            cols = data_tmp.columns.str.split(SPLITTER).str[1]
            filtered_data.append(data_tmp.loc[:, cols.isin(c2g['cpg.1'].unique())])
        elif omics_type in ['protein']:
            cols = data_tmp.columns.str.split(SPLITTER).str[1]
            filtered_data.append(data_tmp.loc[:, cols.isin(p2g['AGID.1'].unique())])
        elif omics_type in ['miRNA']:
            cols = data_tmp.columns.str.split(SPLITTER).str[1]
            filtered_data.append(data_tmp.loc[:, cols.isin(r2g['miRNA'].unique())])
    data = pd.concat(filtered_data, axis=1)
    gset_tmp = modmol_gene_set_tcga(data.columns, op='union', c2g=c2g, p2g=p2g, r2g=r2g)
    assert (cover_gset==gset_tmp).all()
    data = data_to_data_with_modal_embedding(data, cover_gset)
    assert np.isnan(data).any().any() == False

    ### load modals: this code block is to get modal_select_index
    # NOTE original
    # if (modal_all_path == 'None')|(modal_select_path == 'None'):
    #     modal_select_index=list(range(data.shape[2]))
    # else:
    #     modal_all_data=pd.read_csv(modal_all_path,header=None)
    #     modal_all_data.columns=['modal_type']
    #     modal_all_data['index'] = range(len(modal_all_data))
    #     modal_all_data = modal_all_data.set_index('modal_type')
    #     modal_select_data = pd.read_csv(modal_select_path, header=None)
    #     modal_select_index = list(modal_all_data.loc[list(modal_select_data[0]), 'index'])
    modal_select_index = np.arange(data.shape[2]) # using full mask, assuming we already have constructed data modals

    assert len(gene_select_index) == len(cover_gset)
    ### load data
    # data = np.load(file=data_path) # NOTE original
    data_train = data[train_sample, :, :][:, gene_select_index, :][:, :, modal_select_index]
    data_validation = data[validation_sample, :, :][:, gene_select_index, :][:, :, modal_select_index]
    train_dataset = SCDataset(data_train, train_label)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    val_dataset = SCDataset(data_validation, validation_label)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    data_test = data[test_sample, :, :][:, gene_select_index, :][:, :, modal_select_index]
    test_dataset = SCDataset(data_test, test_label)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    ### load Pathway crosstalk netwark
    gene_pathway = np.load(file=pathway_gene_w)
    gene_pathway = torch.LongTensor(gene_pathway)
    gene_pathway = gene_pathway[sel_from_pathformer_genes_mask] # NOTE
    pathway_network = np.load(file=pathway_crosstalk_network)
    pathway_network[np.isnan(pathway_network)] = 0
    pathway_network = torch.Tensor(pathway_network).to(device)

    #######################
    ### Model&optimizer ###
    #######################

    #####hyperparameter
    EPOCH_NUM = epoch_num
    LEARNING_RATE_MAX = lr_max
    LEARNING_RATE_MIN = lr_min
    GRADIENT_ACCUMULATION = gradient_num
    DEPTH = depth #3
    HEAD = heads #8
    HEAD_DIM = dim_head #32

    N_PRJ = data_train.shape[2]  # length of gene embedding
    N_GENE = data_train.shape[1] # number of gene
    col_dim = gene_pathway.shape[1]
    if N_PRJ<=2:
        embeding=True
        embeding_num = 32
        row_dim=embeding_num
        classifier_input = embeding_num * gene_pathway.shape[1]
    else:
        embeding = False
        row_dim = N_PRJ
        embeding_num = N_PRJ # NOTE added this line since originally embeding_num was not defined
        classifier_input = N_PRJ * gene_pathway.shape[1]
    mask_raw = gene_pathway
    label_dim = len(set(label['y']))

    #### class weights
    n_samples = train_label.flatten()
    print(np.unique(n_samples, return_counts=True))
    class_labels, counts = np.unique(n_samples, return_counts=True)
    C = len(class_labels)
    class_weights = len(n_samples) / (C * counts)
    class_weights = class_weights.astype(np.float32)
    class_weight = class_weights

    #####Model
    model = pathformer_model(mask_raw=mask_raw,
                        row_dim=row_dim,
                        col_dim=col_dim,
                        depth=DEPTH,
                        heads=HEAD,
                        dim_head=HEAD_DIM,
                        classifier_input=classifier_input,
                        classifier_dim=classifier_dim,
                        label_dim=label_dim,
                        embeding=embeding,
                        embeding_num=embeding_num,
                        beta=beta,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        classifier_dropout=classifier_dropout).to(device)

    #####optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE_MAX)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,max_lr=LEARNING_RATE_MAX,min_lr=LEARNING_RATE_MIN,
                                                first_cycle_steps=15, cycle_mult=2,warmup_steps=5,gamma=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device, dtype=torch.float32))


    ##############################################
    ###### train, val, test ####
    ##############################################
    torch.cuda.reset_peak_memory_stats(device)

    best_model = None
    patience = patience # NOTE
    best_val_loss = 100000.0
    cnt_no_improvement = 0

    perf = None

    if bk_identification_only == 0:
        import time
        total_train_time = 0.0

        if label_dim == 2:
            for epoch in range(1, EPOCH_NUM+1):
                st_time = time.perf_counter()
                ###### Model Training ########
                model.train()
                running_loss = 0.0
                y_train = []
                predict_train = np.zeros([train_label.shape[0], label_dim])
                for index, (data, labels) in enumerate(train_loader):
                    index += 1
                    if index % 100 == 1:
                        print(index)
                    data = data.to(device)
                    labels = labels.to(device).flatten()
                    pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                    if index % GRADIENT_ACCUMULATION != 0:
                        dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(dec_logits, labels)
                        loss.backward()
                    if index % GRADIENT_ACCUMULATION == 0:
                        dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(dec_logits, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                        optimizer.step()
                        optimizer.zero_grad()
                    y_train.extend(labels.tolist())
                    if (index) * BATCH_SIZE <= train_label.shape[0]:
                        predict_train[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = dec_logits.data.cpu().numpy()
                    else:
                        predict_train[(index - 1) * BATCH_SIZE:, :] = dec_logits.data.cpu().numpy()

                    running_loss += loss.item()
                epoch_loss = running_loss / index
                ACC_train, AUC_train, f1_weighted_train, f1_macro_train = get_roc(np.array(y_train),np.array(predict_train)[:, 1])
                scheduler.step()
                lr_epoch = scheduler.get_lr()[0]

                total_train_time += time.perf_counter() - st_time

                # output
                print(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}')
                print('ACC_train:', ACC_train)
                print('auc_train:', AUC_train)
                print('f1_weighted_train:', f1_weighted_train)
                print('f1_macro_train:', f1_macro_train)
                print('lr:', lr_epoch)

                ###### Model Validation ########
                model.eval()
                running_loss = 0.0
                y_val = []
                predict_val = np.zeros([len(validation_sample), label_dim])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(val_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data.to(device)
                        labels = labels.to(device).flatten()
                        pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_val.extend(labels.tolist())
                        if (index) * BATCH_SIZE <= len(validation_sample):
                            predict_val[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                        else:
                            predict_val[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
                val_loss = running_loss / index
                ACC_val, AUC_val, f1_weighted_val, f1_macro_val = get_roc(np.array(y_val), np.array(predict_val)[:, 1])

                # output
                print()
                print(f'val Loss: {val_loss:.6f}')
                print('ACC_val:', ACC_val)
                print('auc_val:', AUC_val)
                print('f1_weighted_val:', f1_weighted_val)
                print('f1_macro_val:', f1_macro_val)

                print("=========================================================")

                ####### early_stopping ######
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    cnt_no_improvement = 0
                    print(f"Best model updated at epoch {epoch}.")
                else:
                    cnt_no_improvement += 1
                    if cnt_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break
            
            print(f"Pathformer model Training time for {epoch} epochs: {total_train_time:.2f} seconds.")
            print(f"Pathformer model number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

            ######################### perf ########################
            model = best_model # NOTE
            model.eval()
            y_test = []
            predict_test = np.zeros([len(test_sample), label_dim])
            with torch.no_grad():
                for index, (data, labels) in enumerate(test_loader):
                    index += 1
                    data = data.to(device)
                    labels = labels.to(device).flatten()
                    pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                    logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    y_test.extend(labels.tolist())
                    if (index) * BATCH_SIZE <= len(test_sample):
                        predict_test[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                    else:
                        predict_test[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
            y_true = np.array(y_test).flatten()
            y_pred = predict_test.argmax(axis=1).flatten()
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
            assert task_is_biclassif
            y_proba = predict_test[:, -1]
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
        else:
            for epoch in range(1, EPOCH_NUM+1):
                st_time = time.perf_counter()
                ###### Model Training ########
                model.train()
                running_loss = 0.0
                y_train = []
                predict_train = np.zeros([train_label.shape[0], label_dim])
                for index, (data, labels) in enumerate(train_loader):
                    index += 1
                    if index % 100 == 1:
                        print(index)
                    data = data.to(device)
                    labels = labels.to(device).flatten()
                    pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                    if index % GRADIENT_ACCUMULATION != 0:
                        dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(dec_logits, labels)
                        loss.backward()
                    if index % GRADIENT_ACCUMULATION == 0:
                        dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(dec_logits, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                        optimizer.step()
                        optimizer.zero_grad()
                    y_train.extend(labels.tolist())
                    if (index) * BATCH_SIZE <= train_label.shape[0]:
                        predict_train[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = dec_logits.data.cpu().numpy()
                    else:
                        predict_train[(index - 1) * BATCH_SIZE:, :] = dec_logits.data.cpu().numpy()

                    running_loss += loss.item()
                epoch_loss = running_loss / index
                acc_train, auc_weighted_ovr_train, auc_weighted_ovo_train, auc_macro_ovr_train, auc_macro_ovo_train, f1_weighted_train, f1_macro_train = get_roc_multi(np.array(y_train), predict_train)
                y_train_new = np.array(y_train).copy()
                y_train_new[y_train_new >= 1] = 1
                scheduler.step()
                lr_epoch = scheduler.get_lr()[0]

                total_train_time += time.perf_counter() - st_time

                # output
                print(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}')
                print('acc_train:', acc_train)
                print('auc_weighted_ovr_train:', auc_weighted_ovr_train)
                print('auc_weighted_ovo_train:', auc_weighted_ovo_train)
                print('auc_macro_ovr_train:', auc_macro_ovr_train)
                print('auc_macro_ovo_train:', auc_macro_ovo_train)
                print('f1_weighted_train:', f1_weighted_train)
                print('f1_macro_train:', f1_macro_train)
                print('lr_epoch:', lr_epoch)

                ###### Model val ########
                model.eval()
                running_loss = 0.0
                y_val = []
                predict_val = np.zeros([len(validation_sample), label_dim])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(val_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data.to(device)
                        labels = labels.to(device).flatten()
                        pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_val.extend(labels.tolist())
                        if (index) * BATCH_SIZE <= len(validation_sample):
                            predict_val[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                        else:
                            predict_val[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
                val_loss = running_loss / index
                acc_val, auc_weighted_ovr_val, auc_weighted_ovo_val, auc_macro_ovr_val, auc_macro_ovo_val, f1_weighted_val, f1_macro_val = get_roc_multi(
                    np.array(y_val), predict_val)
                y_val_new = np.array(y_val).copy()
                y_val_new[y_val_new >= 1] = 1

                # output
                print()
                print(f'val Loss: {val_loss:.6f}')
                print('acc_val:', acc_val)
                print('auc_weighted_ovr_val:', auc_weighted_ovr_val)
                print('auc_weighted_ovo_val:', auc_weighted_ovo_val)
                print('auc_macro_ovr_val:', auc_macro_ovr_val)
                print('auc_macro_ovo_val:', auc_macro_ovo_val)
                print('f1_weighted_val:', f1_weighted_val)
                print('f1_macro_val:', f1_macro_val)
                print("=========================================================")

                ####### early_stopping ######
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    cnt_no_improvement = 0
                    print(f"Best model updated at epoch {epoch}.")
                else:
                    cnt_no_improvement += 1
                    if cnt_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break
            
            print(f"Pathformer model Training time for {epoch} epochs: {total_train_time:.2f} seconds.")
            model = best_model # NOTE
            ###################### perf #####################
            model.eval()
            y_test = []
            predict_test = np.zeros([len(test_sample), label_dim])
            with torch.no_grad():
                for index, (data, labels) in enumerate(test_loader):
                    index += 1
                    data = data.to(device)
                    labels = labels.to(device).flatten()
                    pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                    logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    y_test.extend(labels.tolist())
                    if (index) * BATCH_SIZE <= len(test_sample):
                        predict_test[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                    else:
                        predict_test[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
            y_true = np.array(y_test)
            y_pred = predict_test.argmax(axis=1)
            roc_auc = None
            aucpr = None
            recall = None
            precision = None
            f1 = None
            f1_weighted = None
            f1_macro = None
            acc = None
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
            }
            #######################################################################

        model = best_model.eval().to(device)
        if (save_model_suffix is not None) and (run_bk_identification == 0):
            print("saving trained model to: ", f'../../../../result/trained_models/Pathformer_model_{save_model_suffix}.pt')
            # move model_dict to cpu
            model_dict = model.state_dict()
            torch.save(model_dict, f'../../../../result/trained_models/Pathformer_model_{save_model_suffix}.pt')

    ft_score = None
    if run_bk_identification == 1:
        if bk_identification_only == 1:
            # load model
            model_dict = torch.load(f'../../../../result/trained_models/Pathformer_model_{save_model_suffix}.pt', map_location=device)
            model = pathformer_model(mask_raw=mask_raw,
                            row_dim=row_dim,
                            col_dim=col_dim,
                            depth=DEPTH,
                            heads=HEAD,
                            dim_head=HEAD_DIM,
                            classifier_input=classifier_input,
                            classifier_dim=classifier_dim,
                            label_dim=label_dim,
                            embeding=embeding,
                            embeding_num=embeding_num,
                            beta=beta,
                            attn_dropout=attn_dropout,
                            ff_dropout=ff_dropout,
                            classifier_dropout=classifier_dropout).to(device)
            model.load_state_dict(model_dict)
            model.eval()

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")

        ############################## bk identification ##############################
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        pathway_network = pathway_network.to(device)
        data_train = torch.tensor(data_train, device=device)
        data_test = torch.tensor(data_test, device=device)
        
        class PathformerWrapper(torch.nn.Module):
            def __init__(self, model):
                super(PathformerWrapper, self).__init__()
                self.model = model
            def forward(self, x):
                pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=x.shape[0])
                logits = model(pathway_network_batch, x.permute(0, 2, 1), output_attentions=False)
                return logits
        model_wrapped = PathformerWrapper(model).to(device)
        
        # NOTE. (SHAP_BATCH_SIZE,11400,6) requires 25GB of GPU memory when SHAP_BATCH_SIZE=1
        # SHAP_BATCH_SIZE = data_test.size(0)
        SHAP_BATCH_SIZE = 1

        torch.cuda.empty_cache()
        gc.collect()

        try:
            background = data_train
            explainer = shap.GradientExplainer(model_wrapped, background, batch_size=SHAP_BATCH_SIZE)
        except:
            try:
                print("CUDA memory exceeds, using 100 random samples as background.")
                random_indices = torch.randperm(data_train.size(0))[:100]
                background = data_train[random_indices]
                explainer = shap.GradientExplainer(model_wrapped, background, batch_size=SHAP_BATCH_SIZE)
            except:
                print("CUDA memory exceeds, using 10 points as background.")
                random_indices = torch.randperm(data_train.size(0))[:10]
                background = data_train[random_indices]
                explainer = shap.GradientExplainer(model_wrapped, background, batch_size=SHAP_BATCH_SIZE)
        index_to_value = {v: k for k, v in value_to_index.items()}
        label_uni = np.unique(test_label.flatten()) # factorized
        ft_score = pd.DataFrame(
            index=np.concatenate([[mdic['mods_uni'][i] + '@' + g for g in cover_gset] for i in range(len(mdic['mods_uni']))]),
            columns=[f'score_{index_to_value[cls]}' for cls in label_uni],
            data=0.0
        ).astype(np.float32)
        modal_dim_dic = {
            'DNAm' : 4,
            'miRNA' : 4,
            'protein' : 4,
            'mRNA' : 1,
            'CNV' : 1,
            'SNV' : 1
        }
        st_time = time.perf_counter()
        shap_values_all = explainer.shap_values(data_test) # (N, G, M, C)
        print("Shap value matrix shape:", shap_values_all.shape)
        for cur_label in label_uni: # already factorized
            print("Running SHAP for label: ", index_to_value[cur_label])
            shap_values = shap_values_all[:, :, :, cur_label] # (N, G, M)
            gene_by_mods_shap_scores = np.mean(np.abs(shap_values), axis=0) # (G, M)
            cur_modal_pos = 0
            for i in range(len(mdic['mods_uni'])):
                cur_mods_mask = ft_score.index.str.split('@').str[0]==mdic['mods_uni'][i]
                cur_modal_pos_end = cur_modal_pos + modal_dim_dic[mdic['mods_uni'][i]]
                ft_score.loc[cur_mods_mask, f'score_{index_to_value[cur_label]}'] = gene_by_mods_shap_scores[:, cur_modal_pos:cur_modal_pos_end].astype(np.float32).mean(axis=1)
                cur_modal_pos = cur_modal_pos_end
        assert cur_modal_pos == gene_by_mods_shap_scores.shape[1], "modal idx did not end correctly."
        # need a further summing over the classes (perform ft_score.sum(axis=1)) to get the shap scores as in the original paper, which is performed in the analysis code.
        print(f"Pathformer BK identification runing time labels: {time.perf_counter()-st_time:.2f} seconds.")

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
