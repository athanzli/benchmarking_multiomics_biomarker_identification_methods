# """
# Script to run DeePathNet with cross validation for any task.
# E.g. python scripts/deepathnet_cv.py configs/tcga_all_cancer_types/mutation_cnv_rna/deepathnet_allgenes_mutation_cnv_rna.json
# """
###################################### 
import pandas as pd
global DEVICE

import sys
sys.path.append('/home/athan.li/eval_bk/code/')
from utils import factorize_label, mod_mol_dict, modmol_gene_set_tcga, convert_omics_to_gene_level, P2G, C2G, SPLITTER, R2G

import json
import sys
from datetime import datetime

from sympy import randMatrix
import torch.optim
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

from .model_transformer_lrp import DeePathNet
from .models import *

STAMP = datetime.today().strftime("%Y%m%d%H%M")
# proj_dir = "/home/scai/DeePathNet" 
proj_dir = '/home/athan.li/eval_bk/code/selected_models/DeePathNet' # NOTE
sys.path.extend([proj_dir])

def get_setup(genes_to_id, id_to_genes, target_dim):
    def load_pathway(random_control=False):
        pathway_dict = {}
        pathway_df = pd.read_csv(configs["pathway_file"])
        # deepathnet_pw_gset = pathway_df['genes'].str.split('|').explode().unique()
        if "min_cancer_publication" in configs:
            pathway_df = pathway_df[
                pathway_df["Cancer_Publications"] > configs["min_cancer_publication"]
            ]
            logger.info(
                f"Filtering pathway with Cancer_Publications > {configs['min_cancer_publication']}"
            )
        if "max_gene_num" in configs:
            pathway_df = pathway_df[pathway_df["GeneNumber"] < configs["max_gene_num"]]
            logger.info(
                f"Filtering pathway with GeneNumber < {configs['max_gene_num']}"
            )
        if "min_gene_num" in configs:
            pathway_df = pathway_df[pathway_df["GeneNumber"] > configs["min_gene_num"]]
            logger.info(
                f"Filtering pathway with GeneNumber > {configs['min_gene_num']}"
            )

        # NOTE removes genes not in the input multiomics dataset
        pathway_df["genes"] = pathway_df["genes"].map(
            lambda x: "|".join([gene for gene in x.split("|") if gene in genes])
        )

        for index, row in pathway_df.iterrows():
            if row["genes"]:
                pathway_dict[row["name"]] = row["genes"].split("|")
        cancer_genes = set(
            [y for x in pathway_df["genes"].values for y in x.split("|")]
        )
        non_cancer_genes = set(genes) - set(cancer_genes)
        logger.info(
            f"Cancer genes:{len(cancer_genes)}\tNon-cancer genes:{len(non_cancer_genes)}"
        )
        if random_control:
            logger.info("Randomly select genes for each pathway")
            for key in pathway_dict:
                pathway_dict[key] = np.random.choice(list(set(cancer_genes)),
                                                     len(pathway_dict[key]), replace=False)
        return pathway_dict, non_cancer_genes

    pathway_dict, non_cancer_genes = load_pathway(random_control=False)
    model = DeePathNet(
        len(omics_types),
        target_dim,
        genes_to_id,
        id_to_genes,
        pathway_dict,
        non_cancer_genes,
        embed_dim=configs["dim"],
        depth=configs["depth"],
        mlp_ratio=configs["mlp_ratio"],
        out_mlp_ratio=configs["out_mlp_ratio"],
        num_heads=configs["heads"],
        pathway_drop_rate=configs["pathway_dropout"],
        only_cancer_genes=configs["cancer_only"],
        tissues=tissues,
    )
    logger.info(
        # open("/home/scai/DeePathNet/scripts/model_transformer_lrp.py", "r").read()
        open("/home/athan.li/eval_bk/code/selected_models/DeePathNet/scripts/model_transformer_lrp.py", "r").read() # NOTE
    )

    logger.info(model)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )

    logger.info(optimizer)

    lr_scheduler = None

    return model, criterion, optimizer, lr_scheduler


def run_experiment(
    merged_df_train, merged_df_test, val_score_dict, run="test", class_name_to_id=None,
):
    train_df = merged_df_train.iloc[:, :num_of_features]
    test_df = merged_df_test.iloc[:, :num_of_features]
    train_target = merged_df_train.iloc[:, num_of_features:]
    test_target = merged_df_test.iloc[:, num_of_features:]

    X_train = train_df
    X_test = test_df

    print("Constructing dataset...")

    if configs["task"] == "classif":
        train_dataset = MultiOmicMulticlassDataset(
            X_train,
            train_target,
            mode="train",
            omics_types=omics_types,
            class_name_to_id=class_name_to_id,
            logger=logger,
        )
        test_dataset = MultiOmicMulticlassDataset(
            X_test,
            test_target,
            mode="val",
            omics_types=omics_types,
            class_name_to_id=class_name_to_id,
            logger=logger,
        )
    else:
        train_dataset = MultiOmicDataset(
            X_train,
            train_target,
            mode="train",
            omics_types=omics_types,
            logger=logger,
            with_tissue=with_tissue,
        )
        test_dataset = MultiOmicDataset(
            X_test,
            test_target,
            mode="val",
            omics_types=omics_types,
            logger=logger,
            with_tissue=with_tissue,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=configs["drop_last"],
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    if configs["task"] == "classif":
        target_dim = len(class_name_to_id)
    else:
        target_dim = train_target.shape[1]

    print("Getting setup...")

    model, criterion, optimizer, lr_scheduler = get_setup(
        train_dataset.genes_to_id, train_dataset.id_to_genes, target_dim
    )

    ### class weights
    n_samples = train_target.values.flatten()
    print(np.unique(n_samples, return_counts=True))
    # we follow the same formula as in the original implementation but extends it to multi-class
    class_labels, counts = np.unique(n_samples, return_counts=True)
    C = len(class_labels)
    class_weights = len(n_samples) / (C * counts)
    class_weights = class_weights.astype(np.float32)
    print(class_weights)

    print("Traininig started...")
    val_drug_ids = merged_df_test.columns[num_of_features:]
    
    import time
    st_time = time.perf_counter()
    val_res, model = train_loop( # NOTE added returned model
        NUM_EPOCHS,
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        logger,
        STAMP,
        configs,
        lr_scheduler,
        val_drug_ids, # NOTE useless since we set save_ckpt to false
        run=run,
        val_score_dict=val_score_dict,
        device=DEVICE,
        class_weights=class_weights # NOTE added
    )
    print(f"DeePathNet model Training time: {time.perf_counter()-st_time} (s).")

    return val_res, model # NOTE added model


def run_deepathnet(
    data_trn, label_trn, data_val, label_val, data_tst, label_tst, device, task
):
    print("Task: ", task)
    global DEVICE, configs, logger, STAMP, BATCH_SIZE, NUM_WORKERS, LOG_FREQ, NUM_EPOCHS, num_of_features, omics_types, with_tissue, genes, tissues
    DEVICE = torch.device(device)
    ################
    # task = 'classif'
    if task == 'classif': #  both biclassif and multiclassif
        config_file = '/home/athan.li/eval_bk/code/selected_models/DeePathNet/configs/deepathnet_forbk_classif.json' # NOTE
    elif task == 'regression':
        config_file = '/home/athan.li/eval_bk/code/selected_models/DeePathNet/configs/deepathnet_forbk_regression.json' # NOTE
    # config_file = '/home/athan.li/eval_bk/code/selected_models/DeePathNet/configs/tcga_all_cancer_types/mutation_cnv_rna/deepathnet_mutation_cnv_rna_example.json'
    configs = json.load(open(config_file, "r"))

    seed = configs["seed"]
    torch.manual_seed(seed)

    BATCH_SIZE = configs["batch_size"]
    NUM_WORKERS = 0
    LOG_FREQ = configs["log_freq"]
    NUM_EPOCHS = configs["num_of_epochs"]

    ###########################################################################
    ###########################################################################
    ### to dict and convert mols to gene-level, ensuring same gene features
    ###########################################################################
    ###########################################################################
    ###########################################################################
    # to dict
    ###########################################################################
    omics_gset = modmol_gene_set_tcga(data_trn.columns, op='union', c2g=C2G, p2g=P2G, r2g=R2G)
    mdic = mod_mol_dict(data_trn.columns)
    ####### filter out genes not in LCPathways first to speed up the section below
    # from matplotlib_venn import venn2
    # venn2([set(omics_gset), set(cancer_genes)])
    # pd.DataFrame(cancer_genes).to_csv("/home/athan.li/eval_bk/code/selected_models/DeePathNet/data/graph_predefined/LCPathways/LCPathways_genes.csv", index=False)
    lcpw_gset = pd.read_csv("/home/athan.li/eval_bk/code/selected_models/DeePathNet/data/graph_predefined/LCPathways/LCPathways_genes.csv", index_col=0).index.values.astype(str)  # this is from cancer_genes in load_pathway
    cover_gset = np.intersect1d(omics_gset, lcpw_gset) # auto unique and sorted

    assert 'mRNA' in mdic['mods_uni'], "mRNA must be present"
    ###########################################################################
    # convert mols to gene-level, ensuring same gene feature names
    ###########################################################################
    data_trn.columns = data_trn.columns.str.split(SPLITTER).str[1]
    data_trn = {k : data_trn.loc[:, mdic['mods']==k] for k in mdic['mods_uni']}
    data_val.columns = data_val.columns.str.split(SPLITTER).str[1]
    data_val = {k : data_val.loc[:, mdic['mods']==k] for k in mdic['mods_uni']}
    data_tst.columns = data_tst.columns.str.split(SPLITTER).str[1]
    data_tst = {k : data_tst.loc[:, mdic['mods']==k] for k in mdic['mods_uni']}
    data_trn = convert_omics_to_gene_level(data_trn, cover_gset)
    data_val = convert_omics_to_gene_level(data_val, cover_gset)
    data_tst = convert_omics_to_gene_level(data_tst, cover_gset)

    gene_names = cover_gset
    n_genes = len(gene_names)

    # concat
    data_trn = pd.concat([data_trn[mod] for mod in mdic['mods_uni']], axis=1)
    data_val = pd.concat([data_val[mod] for mod in mdic['mods_uni']], axis=1)
    data_tst = pd.concat([data_tst[mod] for mod in mdic['mods_uni']], axis=1)

    train_sample_names = data_trn.index
    val_sample_names = data_val.index
    test_sample_names = data_tst.index
    data_input_all = pd.concat([data_trn, data_val, data_tst], axis=0)
    data_input_all.columns = [f'{mol}_{mod}' for mod in mdic['mods_uni'] for mol in gene_names]
    ###########################################################################
    ###########################################################################
    ### end
    ###########################################################################
    ###########################################################################

    val_score_dict = {'run': [],
    'epoch': [],
    'top1_acc': [],
    'top3_acc': [],
    'f1': [],
    'roc_auc': []}

    num_of_features = data_input_all.shape[1]

    data_input_all.index.name = 'Cell_line'
    data_target_all = pd.concat([label_trn, label_val, label_tst], axis=0)
    data_target_all.index.name = 'Cell_line'
    data_target_all.rename(columns={"label": "Cell_type"}, inplace=True)
    val_score_dict = {'run': [],
    'epoch': [],
    'top1_acc': [],
    'top3_acc': [],
    'f1': [],
    'roc_auc': []}
    genes = np.unique([x.split('_')[0] for x in data_input_all.columns]) # this is the genes used in get_setup
    print("number of genes:", len(genes))
    #
    omics_types = mdic['mods_uni']
    #
    with_tissue = False # same as example data
    tissues = None
    from .utils.training_prepare import get_logger # . if running from another file
    logger = get_logger(config_file, STAMP)
    cell_lines_all = data_input_all.index.values
    assert all(data_target_all.index==data_input_all.index)

    class_name_to_id = None
    id_to_class_name = None
    if configs["task"] == "classif": # NOTE
        class_name_to_id = dict(
            zip(
                sorted(data_target_all.iloc[:, 0].unique()),
                list(range(data_target_all.iloc[:, 0].unique().size)),
            )
        )
        id_to_class_name = dict(
            zip(
                list(range(data_target_all.iloc[:, 0].unique().size)),
                sorted(data_target_all.iloc[:, 0].unique()),
            )
        )

    train_lines = train_sample_names
    val_lines = val_sample_names
    tst_lines = test_sample_names

    merged_df_train = pd.merge(data_input_all[data_input_all.index.isin(train_lines)], data_target_all, on=["Cell_line"])
    merged_df_val = pd.merge(data_input_all[data_input_all.index.isin(val_lines)], data_target_all, on=["Cell_line"])
    merged_df_tst = pd.merge(data_input_all[data_input_all.index.isin(tst_lines)], data_target_all, on=["Cell_line"])

    unique_train, unique_test = merged_df_train.iloc[:,-1].unique(), merged_df_val.iloc[:,-1].unique()
    if set(unique_train) != set(unique_test):
        logger.info("Missing class in validation fold")

    count = 0
    num_repeat = 1 if "num_repeat" not in configs else configs["num_repeat"]

    # device = 'cuda:2'
    torch.tensor([1], device=DEVICE) # to ensure the device is initialized correctly.
    torch.cuda.reset_peak_memory_stats(DEVICE) if torch.cuda.is_available() else None

    val_res, model = run_experiment(
        merged_df_train,
        merged_df_val,
        val_score_dict,
        run=f"cv_{count}",
        class_name_to_id=class_name_to_id,
    )

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")


    ################ get test preds and targets
    tst_df = merged_df_tst.iloc[:, :num_of_features]
    tst_target = merged_df_tst.iloc[:, num_of_features:]
    X_test = tst_df
    if configs["task"] == "classif":
        tst_dataset = MultiOmicMulticlassDataset(
            X_test,
            tst_target,
            mode="test",
            omics_types=omics_types,
            class_name_to_id=class_name_to_id,
            logger=logger,
        )
    elif configs["task"] == "regression":
        NotImplementedError
        # tst_dataset = MultiOmicDataset(
        #     X_test,
        #     tst_target,
        #     mode="val",
        #     omics_types=omics_types,
        #     logger=logger,
        #     with_tissue=with_tissue,
        # )
    tst_loader = DataLoader(
        tst_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    outputs, targets = inference(tst_loader, model, device=DEVICE) # NOTE added device
    from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
    from scipy.special import softmax
    ################################ perf ################################
    y_true = targets.flatten()
    y_pred = outputs.argmax(axis=1)
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
        y_proba = softmax(outputs, axis=1)[:, -1]
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
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
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
        'balanced_acc': balanced_acc,
    }
    #######################################################################

    ############################################################################################################################################################################
    ############################################################################################################################################################################
    ############################################################################################################################################################################
    # from transformer_shap_cancer_type import run_shap # too slow
    torch.cuda.reset_peak_memory_stats(DEVICE) if torch.cuda.is_available() else None
    import shap

    print("Running SHAP...")

    train_df = merged_df_train.iloc[:, :num_of_features]
    test_df = merged_df_tst.iloc[:, :num_of_features]
    train_target = merged_df_train.iloc[:, num_of_features:]
    # test_target = merged_df_tst.iloc[:, num_of_features:]
    X_train = train_df
    X_test = test_df
    train_input = torch.tensor(X_train.values).view(-1, n_genes, len(omics_types)).to(DEVICE).float()
    test_input = torch.tensor(X_test.values).view(-1, n_genes, len(omics_types)).to(DEVICE).float()

    label_uni = np.unique(train_target.values.flatten())
    mod_at_gene = [f"{mod}@{gene}" for mod in omics_types for gene in gene_names]
    ft_score = pd.DataFrame(index=mod_at_gene, columns=[f'score_{label_uni[i]}' for i in range(len(label_uni))], data=0.0)

    start = time.perf_counter()
    background = train_input
    
    explainer = shap.GradientExplainer(model, background, batch_size=12) # use batch_size to avoid mem issue
    shap_batch = 12
    shap_values_all = np.array([])
    for i in range(0, test_input.shape[0], shap_batch):
        shap_values = explainer.shap_values(test_input[i:i+shap_batch])
        if len(shap_values_all) == 0:
            shap_values_all = shap_values
        else:
            shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0)
    # shap_values = explainer.shap_values(test_input)
    shap_values = shap_values_all

    # class and mod specific as in the paper
    for i, c in enumerate(label_uni):
        g_by_m = shap_values[:, :, :, i].mean(axis=0)
        for k in range(len(mdic['mods_uni'])):
            ft_score.loc[mod_at_gene[len(gene_names)*k:(k+1)*len(gene_names)], f'score_{c}'] = g_by_m[:, k]
    end = time.perf_counter()
    print("DeePathNet BK identification running time (s):", end - start)

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
