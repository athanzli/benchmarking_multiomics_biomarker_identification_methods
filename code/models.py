import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score

from utils import *

SPLITTER = '@'

MODEL_NAMES = [
    'DeepKEGG',
    'DeePathNet',
    'Pathformer',
    'PNet',
    'CustOmics',
    'TMONet',
    'GENIUS',
    'MOGONET', 
    'MORE',
    'MoAGLSA',
    'MOGLAM',
    'GNNSubNet',

    ####################
    # 'mannwhitneyu',
    # 'ttest',
    'SVM_ONE',
    'SVM_RFE',
    'SVM_ONE_and_SVM_RFE',
    'RF_VI',
    'RF_VI_and_RF_RFE',
    ####################

    # ML & stats
    'DIABLO',
    'asmPLSDA',
    'Stabl',
    'MOFA',
    'GDF',
    'GAUDI',
    'MCIA',
    'DPM',
]

MOGONET_FAMILY_MODELS = ['MOGONET', 'MORE', 'MoAGLSA']

#%%
###############################################################################
# 
###############################################################################

def run_model(model_name: str, **kwargs):
    """
    Run a specified model.

    Args:
        model_name (str): The name of the model to run.
    """
    model_funcs = {
        'CustOmics': run_customics,
        'GENIUS': run_genius,
        "MOGONET": run_mogonet,
        "MORE": run_more,
        'MoAGLSA': run_moaglsa,
        'MOGLAM': run_moglam,
        'GNNSubNet': run_gnnsubnet,
        # "ttest": run_ttest,
        "TMONet": run_tmonet,
        "DeepKEGG": run_deepkegg,
        "PNet": run_pnet,
        "DeePathNet": run_deepathnet,
        "Pathformer": run_pathformer,

        "SVM_ONE_and_SVM_RFE": run_SVM_ONE_and_SVM_RFE,
        "RF_VI_and_RF_RFE": run_RF_VI_and_RF_RFE,
        # "mannwhitneyu": run_mannwhitneyu,
        # "PLSDA": run_PLSDA,
        # "IG": run_IG,
        # "ReliefF": run_ReliefF,
        # ML & stats
        'DIABLO': run_diablo,
        'asmPLSDA': run_asmplsda,
        'Stabl': run_stabl,
        'GDF': run_gdf,
        'MOFA': run_mofa,
        'GAUDI': run_gaudi,
        'MCIA': run_mcia,
        'DPM': run_dpm,
    }
    print("Running model:", model_name)
    return model_funcs[model_name](**kwargs)

#%%
###############################################################################
# DL models
###############################################################################
def run_mogonet(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:6',
    bk_identification_only=0,
    run_bk_identification=1,
    save_model_suffix=None,
):
    from selected_models.MOGONET.train_test import train_test
    from selected_models.MOGONET.feat_importance import cal_feat_imp, cal_feat_imp

    ft_names = data_trn.columns
    # divide data into a list of different omics data for training
    mods = np.array([col.split(SPLITTER)[0] for col in data_trn.columns])
    mods_uni = np.unique(mods)
    data_trn = [data_trn.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for validation data
    mods = np.array([col.split(SPLITTER)[0] for col in data_val.columns])
    mods_uni = np.unique(mods)
    data_val = [data_val.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for test data
    mods = np.array([col.split(SPLITTER)[0] for col in data_tst.columns])
    mods_uni = np.unique(mods)
    data_tst = [data_tst.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]

    num_view = len(data_trn)
    view_list = [i+1 for i in range(num_view)]
    
    all_labels = np.concatenate([
        label_trn.values.flatten(),
        label_val.values.flatten(),
        label_tst.values.flatten()
    ])
    y_fact_all, fact_dic = factorize_label(all_labels)
    n_tr = label_trn.shape[0]
    n_val = label_val.shape[0]
    y_fact_trn = y_fact_all[:n_tr]
    y_fact_val = y_fact_all[n_tr:n_tr+n_val]
    y_fact_tst = y_fact_all[n_tr+n_val:]
    num_class = len(np.unique(y_fact_all))

    # default params of MOGONET
    adj_parameter = 10
    dim_he_list = [400,400,200]
    num_epoch_pretrain = 500
    num_epoch = 2500 # NOTE
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    perf = None
    if bk_identification_only == 0:
        # ############# using train, val, and test (validation for early stopping)
        ### class weights
        n_samples = y_fact_trn
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

        model_dict, perf = train_test(
            data_trn=data_trn,
            y_trn=y_fact_trn,
            data_val=data_val,
            y_val=y_fact_val,
            data_tst=data_tst,
            y_tst=y_fact_tst,
            view_list=view_list,
            num_epoch_pretrain=num_epoch_pretrain,
            num_epoch=num_epoch,
            lr_e_pretrain=lr_e_pretrain,
            lr_e=lr_e,
            lr_c=lr_c,
            num_class=num_class,
            adj_parameter=adj_parameter,
            dim_he_list=dim_he_list,
            device=device,
            class_weights=class_weights
        )
        if (save_model_suffix is not None) and (run_bk_identification == 0):
            # move model_dict to cpu
            model_dict = {k: v.cpu() for k, v in model_dict.items()}

            # NOTE
            print("saving trained model to: ", f'/data/zhaohong/eval_bk/result/trained_models/MOGONET_model_{save_model_suffix}.pt')
            torch.save(model_dict, f'/data/zhaohong/eval_bk/result/trained_models/MOGONET_model_{save_model_suffix}.pt')

    if run_bk_identification == 1:
        # device='cpu'
        if bk_identification_only == 1:
            # load model
            model_dict = torch.load(f'/data/zhaohong/eval_bk/result/trained_models/MOGONET_model_{save_model_suffix}.pt', map_location=device, weights_only=False)
        # NOTE
        for module in model_dict.values():
            if hasattr(module, 'device'):
                module.device = device
                module.to(device)
        feat_imp_list = cal_feat_imp( # NOTE
            data_trn=data_trn,
            y_trn=y_fact_trn,
            data_tst=data_tst,
            y_tst=y_fact_tst,
            num_class=num_class,
            adj_parameter=adj_parameter,
            model_dict=model_dict,
            device=device
        )
        # feat_imp_list is a list of multiple pandas DataFrame, each containing the feature importance of the corresponding view, with columns "feat_name" and "imp"
        df_tmp = pd.concat(feat_imp_list, axis=0)
        ft_score = pd.DataFrame(index=df_tmp['feat_name'].values)
        ft_score['score'] = df_tmp['imp'].values

        assert all(ft_score.index == ft_names)
        
        print("MOGONET feature imp calc complete. #positive features:", sum(ft_score['score'].values > 0))

        return ft_score, perf
    else:
        return None, perf

def run_more(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:7',
    bk_identification_only=0,
    run_bk_identification=1,
    save_model_suffix=None
):
    r"""
    Run the MORE model.
    """
    from selected_models.MORE.Code.train_test import train_test
    from selected_models.MORE.Code.feat_importance import cal_feat_imp

    ft_names = data_trn.columns
    # divide data into a list of different omics data for training
    mods = np.array([col.split(SPLITTER)[0] for col in data_trn.columns])
    mods_uni = np.unique(mods)
    data_trn = [data_trn.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for validation data
    mods = np.array([col.split(SPLITTER)[0] for col in data_val.columns])
    mods_uni = np.unique(mods)
    data_val = [data_val.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for test data
    mods = np.array([col.split(SPLITTER)[0] for col in data_tst.columns])
    mods_uni = np.unique(mods)
    data_tst = [data_tst.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]

    num_view = len(data_trn)
    view_list = [i+1 for i in range(num_view)]
    
    all_labels = np.concatenate([
        label_trn.values.flatten(),
        label_val.values.flatten(),
        label_tst.values.flatten()
    ])
    y_fact_all, fact_dic = factorize_label(all_labels)
    n_tr = label_trn.shape[0]
    n_val = label_val.shape[0]
    y_fact_trn = y_fact_all[:n_tr]
    y_fact_val = y_fact_all[n_tr:n_tr+n_val]
    y_fact_tst = y_fact_all[n_tr+n_val:]
    num_class = len(np.unique(y_fact_all))

    # default params of MORE
    num_epoch_pretrain = 500
    num_epoch = 1500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    perf = None
    if bk_identification_only == 0:
        # ############# using train, val, and test (validation for early stopping)
        ### class weights
        n_samples = y_fact_trn
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
        model_dict, perf = train_test(
            data_trn=data_trn,
            y_trn=y_fact_trn,
            data_val=data_val,
            y_val=y_fact_val,
            data_tst=data_tst,
            y_tst=y_fact_tst,
            view_list=view_list,
            num_epoch_pretrain=num_epoch_pretrain,
            num_epoch=num_epoch,
            lr_e_pretrain=lr_e_pretrain,
            lr_e=lr_e,
            lr_c=lr_c,
            num_class=num_class,
            device=device,
            class_weights=class_weights
        )
        if (save_model_suffix is not None) and (run_bk_identification == 0):
            # move model_dict to cpu
            model_dict = {k: v.cpu() for k, v in model_dict.items()}
    
            #  NOTE
            print("saving trained model to: ", f'/data/zhaohong/eval_bk/result/trained_models/MORE_model_{save_model_suffix}.pt')
            torch.save(model_dict, f'/data/zhaohong/eval_bk/result/trained_models/MORE_model_{save_model_suffix}.pt')


    if run_bk_identification == 1:
        # device='cpu'
        device=device
        if bk_identification_only == 1:
            # load model
            model_dict = torch.load(f'/data/zhaohong/eval_bk/result/trained_models/MORE_model_{save_model_suffix}.pt', map_location=device, weights_only=False)
        for module in model_dict.values():
            if hasattr(module, 'device'):
                module.device = device
                module.to(device)
        num_views = len(mods_uni)
        for i in range(num_views):
            model_dict[f'E{i+1}'].device = device
            model_dict[f'E{i+1}'].hgc1.device = device
            model_dict[f'E{i+1}'].hgc2.device = device
            model_dict[f'E{i+1}'].hgc1.to(device)
            model_dict[f'E{i+1}'].hgc2.to(device)
            # print("POS 2 device: ", device)
        model_dict['C'].device = device
        model_dict['C'].to(device)
        model_dict['C'].InputLayer.device = device
        model_dict['C'].InputLayer.to(device)
        feat_imp_list = cal_feat_imp(
            data_trn=data_trn,
            y_trn=y_fact_trn,
            data_tst=data_tst,
            y_tst=y_fact_tst,
            num_class=num_class,
            model_dict=model_dict,
            device=device
        )
        # feat_imp_list is a list of multiple pandas DataFrame, each containing the feature importance of the corresponding view, with columns "feat_name" and "imp"
        df_tmp = pd.concat(feat_imp_list, axis=0)
        ft_score = pd.DataFrame(index=df_tmp['feat_name'].values)
        ft_score['score'] = df_tmp['imp'].values

        assert all(ft_score.index == ft_names)
        
        print("MORE feature imp calc complete. #positive features:", sum(ft_score['score'].values > 0))

        return ft_score, perf
    else:
        return None, perf

def run_moaglsa(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:7',
    bk_identification_only=0,
    run_bk_identification=1,
    save_model_suffix=None
):
    r"""
    Run the MoAGL-SA model.
    """
    from selected_models.MoAGLSA.train_test import train_test
    from selected_models.MoAGLSA.feat_importance import cal_feat_imp

    ft_names = data_trn.columns
    # divide data into a list of different omics data for training
    mods = np.array([col.split(SPLITTER)[0] for col in data_trn.columns])
    mods_uni = np.unique(mods)
    data_trn = [data_trn.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for validation data
    mods = np.array([col.split(SPLITTER)[0] for col in data_val.columns])
    mods_uni = np.unique(mods)
    data_val = [data_val.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for test data
    mods = np.array([col.split(SPLITTER)[0] for col in data_tst.columns])
    mods_uni = np.unique(mods)
    data_tst = [data_tst.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]

    num_view = len(data_trn)
    view_list = [i+1 for i in range(num_view)]
    
    all_labels = np.concatenate([
        label_trn.values.flatten(),
        label_val.values.flatten(),
        label_tst.values.flatten()
    ])
    y_fact_all, fact_dic = factorize_label(all_labels)
    n_tr = label_trn.shape[0]
    n_val = label_val.shape[0]
    y_fact_trn = y_fact_all[:n_tr]
    y_fact_val = y_fact_all[n_tr:n_tr+n_val]
    y_fact_tst = y_fact_all[n_tr+n_val:]
    num_class = len(np.unique(y_fact_all))

    perf = None
    if bk_identification_only == 0:
        # ############# using train, val, and test (validation for early stopping)
        ### class weights
        n_samples = y_fact_trn
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
        model_dict, perf = train_test(
            data_trn=data_trn,
            y_trn=y_fact_trn,
            data_val=data_val,
            y_val=y_fact_val,
            data_tst=data_tst,
            y_tst=y_fact_tst,
            num_class=num_class,
            num_epoch=2500,
            device=device,
            class_weights=class_weights
        )
        if (save_model_suffix is not None) and (run_bk_identification == 0):
            # move model_dict to cpu
            model_dict = {k: v.cpu() for k, v in model_dict.items()}

            #  NOTE
            print("saving trained model to: ", f'/data/zhaohong/eval_bk/result/trained_models/MoAGLSA_model_{save_model_suffix}.pt')
            torch.save(model_dict, f'/data/zhaohong/eval_bk/result/trained_models/MoAGLSA_model_{save_model_suffix}.pt')

    if run_bk_identification == 1:
        # device='cpu'
        if bk_identification_only == 1:
            # load model
            model_dict = torch.load(f'/data/zhaohong/eval_bk/result/trained_models/MoAGLSA_model_{save_model_suffix}.pt', map_location=device, weights_only=False)
        # NOTE 
        for module in model_dict.values():
            if hasattr(module, 'device'):
                module.device = device
                module.to(device)

        st_time = time.perf_counter()
        feat_imp_list = cal_feat_imp(
            data_trn=data_trn,
            y_trn=y_fact_trn,
            data_tst=data_tst,
            y_tst=y_fact_tst,
            num_class=num_class,
            model_dict=model_dict,
            device=device
        )
        print("BK identification running time:", (time.perf_counter() - st_time), "seconds")
        # feat_imp_list is a list of multiple pandas DataFrame, each containing the feature importance of the corresponding view, with columns "feat_name" and "imp"
        df_tmp = pd.concat(feat_imp_list, axis=0)
        ft_score = pd.DataFrame(index=df_tmp['feat_name'].values)
        ft_score['score'] = df_tmp['imp'].values

        assert all(ft_score.index == ft_names)
        
        print("MoAGLSA feature imp calc complete. #positive features:", sum(ft_score['score'].values > 0))
        return ft_score, perf
    else:
        return None, perf

def run_moglam(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:1',
):
    from selected_models.MOGLAM.train_test import train_test

    ft_names = data_trn.columns
    # divide data into a list of different omics data for training
    mods = np.array([col.split(SPLITTER)[0] for col in data_trn.columns])
    mods_uni = np.unique(mods)
    data_trn = [data_trn.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for validation data
    mods = np.array([col.split(SPLITTER)[0] for col in data_val.columns])
    mods_uni = np.unique(mods)
    data_val = [data_val.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    
    # same for test data
    mods = np.array([col.split(SPLITTER)[0] for col in data_tst.columns])
    mods_uni = np.unique(mods)
    data_tst = [data_tst.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]

    num_view = len(data_trn)
    view_list = [i+1 for i in range(num_view)]
    
    all_labels = np.concatenate([
        label_trn.values.flatten(),
        label_val.values.flatten(),
        label_tst.values.flatten()
    ])
    y_fact_all, fact_dic = factorize_label(all_labels)
    n_tr = label_trn.shape[0]
    n_val = label_val.shape[0]
    y_fact_trn = y_fact_all[:n_tr]
    y_fact_val = y_fact_all[n_tr:n_tr+n_val]
    y_fact_tst = y_fact_all[n_tr+n_val:]
    num_class = len(np.unique(y_fact_all))

    # default
    num_epoch_pretrain = 500
    num_epoch = 3000
    theta_smooth = 1
    theta_degree = 0.5
    theta_sparsity = 0.5
    lr_e_pretrain = 1e-4
    lr_e = 1e-5
    lr_c = 1e-6
    reg = 0.001
    neta = 0.1

    ### class weights
    n_samples = y_fact_trn
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

    model_dict, perf = train_test(
        data_trn=data_trn,
        y_trn=y_fact_trn,
        data_val=data_val,
        y_val=y_fact_val,
        data_tst=data_tst,
        y_tst=y_fact_tst,
        device=device,
        class_weights=class_weights,
        num_class=num_class,
        lr_e_pretrain=lr_e_pretrain,
        lr_e=lr_e,
        lr_c=lr_c, 
        num_epoch_pretrain=num_epoch_pretrain,
        num_epoch=num_epoch,
        theta_smooth=theta_smooth,
        theta_degree=theta_degree,
        theta_sparsity=theta_sparsity,
        neta=neta,
        reg=reg,
        view_list=view_list)

    ################# bk iden ###################
    def extract_feature_scores(model_dict, feature_names_dict=None):
        """
        Extract feature scores from MOGLAM's learned feature indicator matrices.
        """
        scores_list = []
        for key in model_dict: # iterate over the keys corresponding to the omic-specific encoder branches
            if key.startswith('E'):
                encoder = model_dict[key] # retrieve the omic-specific encoder module
                # extract the feature indicator matrix from the first graph convolution layer.
                WF_tensor = encoder.gc1.WF   # WF is of shape (num_input_features, featuresSelect)
                WF_param = WF_tensor.data.cpu().numpy()
                
                feature_scores = np.mean(np.abs(WF_param), axis=1) # NOTE taking absolute values here.

                view_index = int(key[1:])
                if feature_names_dict is not None and view_index in feature_names_dict:
                    feature_names = feature_names_dict[view_index]
                else:
                    raise ValueError(f"Feature names for view {view_index} not provided.")
                df_view = pd.DataFrame({
                    'view': view_index,
                    'score': feature_scores
                }, index=feature_names)
                df_view.index.name = 'feature'
                scores_list.append(df_view)
        result_df = pd.concat(scores_list, axis=0)
        return result_df
    ft_score = extract_feature_scores(model_dict, feature_names_dict={1: data_trn[0].columns, 2: data_trn[1].columns, 3: data_trn[2].columns})['score'].to_frame()
    ft_score = ft_score.loc[ft_names]
    return ft_score, perf

def run_customics(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:0',
    run_bk_identification=True,
    task='classification' # task='survival'
):
    r"""
    Args:
        data (pd.DataFrame): data for identifying biomarkers, as well as for training
        label: data label for identifying biomarkers, as well as for training
        data_tst (pd.DataFrame): test data for evaluating performance
        label_tst (pd.DataFrame): test data label for evaluating performance
    Returns:
        ft_score (pd.DataFrame): feature importance scores

    """
    ### check
    assert np.intersect1d(data_trn.index, data_tst.index).shape[0] == 0, "Data overlap between training and test sets. This will cause error later."
    assert np.intersect1d(data_trn.index, data_val.index).shape[0] == 0, "Data overlap between training and val sets. This will cause error later."

    ### import
    from selected_models.CustOmics.src.network.customics import CustOMICS
    from selected_models.CustOmics.src.tools.utils import get_sub_omics_df

    ### data
    # divide data into a list of different omics data
    mods = np.array([col.split(SPLITTER)[0] for col in data_trn.columns])
    mods_uni = np.unique(mods)
    data_trn = [data_trn.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    # same for data_val
    mods = np.array([col.split(SPLITTER)[0] for col in data_val.columns])
    mods_uni = np.unique(mods)
    data_val = [data_val.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]
    # same for data_tst
    mods = np.array([col.split(SPLITTER)[0] for col in data_tst.columns])
    mods_uni = np.unique(mods)
    data_tst = [data_tst.loc[:, mods==mods_uni[i]] for i in range(len(mods_uni))]

    mod_mapping = {
        'protein' : 'protein',
        'mRNA' : 'RNAseq',
        'DNAm' : 'methyl',
        'CNV' : 'CNV',
        'SNV' : 'SNV',
        'miRNA' : 'miRNA'
    }
    data_all = [pd.concat([data_trn[i], data_val[i], data_tst[i]], axis=0) for i in range(len(data_trn))]

    omics_df = {}
    for i, mod in enumerate(mods_uni):
        omics_df[mod_mapping[mod]] = data_all[i]

    clinical_df_trn = label_trn.copy()
    clinical_df_val = label_val.copy()
    clinical_df_tst = label_tst.copy()
    if task == 'classification':
        clinical_df_trn.loc[:, 'event'] = np.random.randint(0, 2, size=label_trn.shape[0])
        clinical_df_trn.loc[:, 'time'] = np.random.randint(1, 1000, size=label_trn.shape[0]).astype(np.float32)
        clinical_df_val.loc[:, 'event'] = np.random.randint(0, 2, size=label_val.shape[0]) # NOTE
        clinical_df_val.loc[:, 'time'] = np.random.randint(1, 1000, size=label_val.shape[0]).astype(np.float32) # NOTE
        clinical_df_tst.loc[:, 'event'] = np.random.randint(0, 2, size=label_tst.shape[0])
        clinical_df_tst.loc[:, 'time'] = np.random.randint(1, 1000, size=label_tst.shape[0]).astype(np.float32)
    elif task == 'survival':
        clinical_df_trn.rename(columns={'E': 'event', 'T': 'time'}, inplace=True)
        clinical_df_val.rename(columns={'E': 'event', 'T': 'time'}, inplace=True)
        clinical_df_tst.rename(columns={'E': 'event', 'T': 'time'}, inplace=True)
        clinical_df_trn.loc[:, 'label'] = np.random.randint(0, 2, size=label_trn.shape[0])
        clinical_df_val.loc[:, 'label'] = np.random.randint(0, 2, size=label_val.shape[0])
        clinical_df_tst.loc[:, 'label'] = np.random.randint(0, 2, size=label_tst.shape[0])
    clinical_df = pd.concat([clinical_df_trn, clinical_df_val, clinical_df_tst], axis=0)
    assert all(clinical_df.index==data_all[0].index), "Index mismatch between data and label."

    samples_train = list(clinical_df_trn.index)
    samples_val = list(clinical_df_val.index) # NOTE
    samples_test = list(clinical_df_tst.index)
    omics_train = get_sub_omics_df(omics_df, samples_train)
    omics_val = get_sub_omics_df(omics_df, samples_val)
    omics_test = get_sub_omics_df(omics_df, samples_test)
    x_dim = [omics_df[omic_source].shape[1] for omic_source in omics_df.keys()]
    device = torch.device(device) # NOTE
    sources = list(omics_df.keys())
    label = 'label'
    event = 'event'
    surv_time = 'time'

    #### hyperparams. Default as in CustOmics code.
    if task == 'classification':
        num_classes = clinical_df[label].nunique() # NOTE changed from 5 to num_classes
        lambda_classif = 1
        lambda_survival = 0
    elif task == 'survival':
        num_classes = 2 # can be arbitrary > 1 integers (cannot be 1)
        lambda_classif = 0
        lambda_survival = 1
    hidden_dim = [1024, 512, 256]
    central_dim = [2048, 1024, 512, 256]
    rep_dim = 128
    latent_dim=128
    dropout = 0.2
    beta = 1
    classifier_dim = [256, 128]
    survival_dim = [64,32]
    batch_size = 32
    n_epochs = 1000 # NOTE originally 20
    switch = 10
    lr = 0.001

    assert n_epochs > switch, "n_epochs should be greater than switch."

    source_params = {}
    central_params = {'hidden_dim': central_dim, 'latent_dim': latent_dim, 'norm': True, 'dropout': dropout, 'beta': beta}
    classif_params = {'n_class': num_classes, 'lambda': lambda_classif, 'hidden_layers': classifier_dim, 'dropout': dropout}
    surv_params = {'lambda': lambda_survival, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True, 'dropout': dropout, 'device': device}
    for i, source in enumerate(sources):
        source_params[source] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': rep_dim, 'norm': True, 'dropout': dropout}
    train_params = {'switch': switch, 'lr': lr}

    ### model
    model = CustOMICS(
        source_params=source_params,
        central_params=central_params,
        classif_params=classif_params,
        surv_params=surv_params,
        train_params=train_params,
        device=device).to(device)
    model.fit(
        omics_train=omics_train,
        clinical_df=clinical_df,
        label=label,
        event=event,
        surv_time=surv_time,
        omics_val=omics_val,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=True)
    print('CustOMICS model parameters:', model.get_number_parameters())
    # test
    perf = model.evaluate(
        omics_test=omics_test,
        clinical_df=clinical_df,
        label=label,
        event=event,
        surv_time=surv_time,
        task=task, batch_size=1024, plot_roc=False)
    print(f'Validation set performance: {perf}')
    # model.plot_loss()
    # model.get_latent_representation(omics_df)

    if run_bk_identification:
        ############# shap
        # sanity check. consistency of CustOmics's label encoding with ours
        label_uni = clinical_df[label].unique()
        y = clinical_df[label].values.flatten()
        y_fact, fact_dic = factorize_label(y)
        encoded_clinical_df_label = model.label_encoder.transform(clinical_df.loc[:, label].values)
        assert all(encoded_clinical_df_label == y_fact) and all(model.label_encoder.inverse_transform(y_fact) == clinical_df[label].values), \
            'Label encoding is not consistent.'

        ft_names = np.concatenate([omics_df[source].columns for source in sources])
        ft_score = pd.DataFrame(index=ft_names, columns=[f'score_{label_uni[i]}' for i in range(len(label_uni))])
        sample_id = clinical_df_tst.index # NOTE

        torch.cuda.reset_peak_memory_stats(device)

        st_time = time.perf_counter()
        if task == 'classification':
            for source in sources:        
                for subtype in label_uni: # class label for which you want to calculate shap values. should be original label not encoded
                    print(f'Calculating SHAP values for source {source} and label {subtype}...')
                    mod_class_ft_score = model.explain(
                        model=model,
                        sample_id=sample_id,
                        omics_df=omics_df,
                        clinical_df=clinical_df,
                        source=source,
                        subtype=subtype,
                        label=label,
                        device=device,
                        show=False,
                        background_ids=data_trn[0].index.values, # train sample ids
                        task='classification'
                    )
                    ft_score.loc[omics_df[source].columns, f'score_{subtype}'] = mod_class_ft_score

        elif task == 'survival':
            for source in sources:
                print(f'Calculating SHAP values for source {source}...')
                mod_ft_score = model.explain(
                    model=model,
                    sample_id=sample_id,
                    omics_df=omics_df,
                    clinical_df=clinical_df,
                    source=source,
                    subtype=None,
                    label=label,
                    device=device,
                    show=False,
                    background_ids=data_trn[0].index.values, # train sample ids
                    task='survival'
                )
                ft_score.loc[omics_df[source].columns, f'score'] = mod_ft_score
        print(f"CustOMICS BK identification running time: {time.perf_counter()-st_time:.2f} seconds.")

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")

        return ft_score, perf
    else:
        None, perf

def run_tmonet(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:0',

    # params. default
    pretrain_epochs = 50,
    downstream_epochs = 1000, # NOTE 
    task='classification'
):
    r"""
    """
    if 'cuda' not in device:
        raise ValueError("Device must be a cuda device.")
    device_id = int(device.split(':')[-1])

    from selected_models.TMONet.dataset.dataset import CancerDataset_forbk
    from selected_models.TMONet.train.forbk_train_tmonet import TCGA_Dataset_pretrain, TCGA_Dataset_classification, TMONET_PATH
    from captum.attr import IntegratedGradients
    
    mdic = mod_mol_dict(data_trn.columns)
    mods_uni = mdic['mods_uni']
    omics = dict(zip(mods_uni, range(len(mods_uni))))

    #----- NOTE will switch back
    data_tst0 = data_tst.copy()
    label_tst0 = label_tst.copy()
    data_tst = data_val.copy()
    label_tst = label_val.copy()
    #-----

    label_trn = label_trn.copy()
    # divide data into a list of different omics data
    omics_data_trn = [data_trn.loc[:, mdic['mods']==mods_uni[i]] for i in range(len(mods_uni))]
    # same for data_tst
    omics_data_tst = [data_tst.loc[:, mdic['mods']==mods_uni[i]] for i in range(len(mods_uni))]

    # from omics_data to data to ensure the order of mods in data
    data_trn = pd.concat(omics_data_trn, axis=1)
    data_tst = pd.concat(omics_data_tst, axis=1)

    for i in range(len(omics_data_trn)):
        # add a column called ID as the first column
        omics_data_trn[i].insert(0, 'ID', omics_data_trn[i].index)
        omics_data_tst[i].insert(0, 'ID', omics_data_tst[i].index)
    
    omics_data_trn = {mods_uni[i]: omics_data_trn[i] for i in range(len(mods_uni))}
    omics_data_tst = {mods_uni[i]: omics_data_tst[i] for i in range(len(mods_uni))}

    # NOTE currently using dummy values for surv.
    clinical_data_trn = pd.DataFrame({
        'ID': omics_data_trn[mods_uni[0]]['ID'].values,
        'Cancer_Type': label_trn['label'].values,
        'OS': np.full(omics_data_trn[mods_uni[0]].shape[0], 1), # NOTE dummy values since not used in TCGA_Dataset_pretrain func
        'OS.time': np.full(omics_data_trn[mods_uni[0]].shape[0], 1) # NOTE dummy values since not used in TCGA_Dataset_pretrain func
    })
    clinical_data_tst = pd.DataFrame({
        'ID': omics_data_tst[mods_uni[0]]['ID'].values,
        'Cancer_Type': label_tst['label'].values,
        'OS': np.full(omics_data_tst[mods_uni[0]].shape[0], 1), # NOTE dummy values since not used in TCGA_Dataset_pretrain func
        'OS.time': np.full(omics_data_tst[mods_uni[0]].shape[0], 1) # NOTE dummy values since not used in TCGA_Dataset_pretrain func
    })

    num_mods = len(mods_uni)
    mods_dim = [omics_data_trn[mod].shape[1] - 1 for mod in mods_uni] # NOTE minus 1 to exclude ID column
    omics_data_type = ['gaussian'] * num_mods # NOTE as default

    y_trn_fact, cancer_fact_mapping = factorize_label(label_trn['label'].values)

    train_dataset = CancerDataset_forbk(
        omics_data_trn,
        clinical_data_trn,
        cancer_fact_mapping=cancer_fact_mapping)
    test_dataset = CancerDataset_forbk(
        omics_data_tst,
        clinical_data_tst,
        cancer_fact_mapping=cancer_fact_mapping)

    Loss_list, test_Loss_list, pretrain_score_list, pretrain_model_dict = TCGA_Dataset_pretrain(
        omics=omics,
        num_mods=num_mods,
        mods_dim=mods_dim,
        omics_data_type=omics_data_type,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        fold=0,
        epochs=pretrain_epochs,
        device_id=device_id)

    torch.cuda.reset_peak_memory_stats(device)
    if task == 'classification':
        ### class weights
        n_samples = y_trn_fact
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

        # fold is used in this function only for recording. value will not affect prediction res.
        model, perf = TCGA_Dataset_classification(
            omics=omics,
            num_mods=num_mods,
            mods_dim=mods_dim,
            omics_data_type=omics_data_type,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            fold=0,
            epochs=downstream_epochs,
            pretrain_model_dict=pretrain_model_dict,
            fixed=False,
            device_id=device_id,
            class_weights=class_weights)
        print(f'Validation set performance: {perf}')

        ########## NOTE get test set performance. switch back.
        data_tst = data_tst0.copy()
        label_tst = label_tst0.copy()
        omics_data_tst = [data_tst.loc[:, mdic['mods']==mods_uni[i]] for i in range(len(mods_uni))]
        data_tst = pd.concat(omics_data_tst, axis=1)
        for i in range(len(omics_data_trn)):
            omics_data_tst[i].insert(0, 'ID', omics_data_tst[i].index)
        omics_data_tst = {mods_uni[i]: omics_data_tst[i] for i in range(len(mods_uni))}
        # NOTE currently using dummy values for surv.
        clinical_data_tst = pd.DataFrame({
            'ID': omics_data_tst[mods_uni[0]]['ID'].values,
            'Cancer_Type': label_tst['label'].values,
            'OS': np.full(omics_data_tst[mods_uni[0]].shape[0], 1), # NOTE dummy values since not used in TCGA_Dataset_pretrain func
            'OS.time': np.full(omics_data_tst[mods_uni[0]].shape[0], 1) # NOTE dummy values since not used in TCGA_Dataset_pretrain func
        })
        test_dataset = CancerDataset_forbk(
            omics_data_tst,
            clinical_data_tst,
            cancer_fact_mapping=cancer_fact_mapping)
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=64)
        param_groups = [
            {'params': model.cross_encoders.parameters(), 'lr': 0.00001},
            {'params': model.downstream_predictor.parameters(), 'lr': 0.0001},
        ]
        optimizer = torch.optim.Adam(param_groups)
        fold = 0
        epoch = 'N/A'
        criterion = torch.nn.CrossEntropyLoss()
        from selected_models.TMONet.train.forbk_train_tmonet import test_classification
        perf, tst_loss = test_classification( 
            test_dataloader,
            model,
            epoch,
            'pancancer', # an argument that is not used
            fold,
            optimizer,
            omics,
            criterion)
        print("Test set performance:", perf)

    elif task == 'survival':
        NotImplementedError

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")

    ### Integrated gradients for feature importance
    torch.cuda.reset_peak_memory_stats(device)
    class TMONetWrapper(torch.nn.Module):
        # this class is for running ig in our bk benchmarking
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input):
            # input must be N x (L1 + L2 + ... + Ln). Multimodal concatenated.
            inputs = {i:input[:, mdic['mods']==mods_uni[i]] for i in range(len(mods_uni))}
            return self.model.forward_for_ig(inputs)
    ig = IntegratedGradients(TMONetWrapper(model))

    cat_input = torch.tensor(data_tst.values.astype(np.float32)).to(device) # NOTE using test data

    label_uni = label_tst['label'].unique()
    ft_score = pd.DataFrame(
        index=data_tst.columns,
        columns=[f'score_{label_uni[i]}' for i in range(len(label_uni))])

    start_time = time.perf_counter()

    for c in label_uni:
        target = cancer_fact_mapping[c]
        attr = ig.attribute(
            inputs=cat_input,
            target=target,
            return_convergence_delta=False,
            internal_batch_size=cat_input.shape[0], # NOTE
            method='gausslegendre',
            n_steps=50
        )
        ft_class_score = attr.cpu().numpy().mean(axis=0) # (N_class, L) --> (L)
        ft_score.loc[data_tst.columns, f'score_{c}'] = ft_class_score
    
    print("BK identification running time (s):", time.perf_counter() - start_time)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")

    return ft_score, perf

def run_genius(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device
):
    from selected_models.GENIUS.Training.run_genius_for_bk import run_genius
    ft_score, perf = run_genius(data_trn, label_trn, data_val, label_val, data_tst, label_tst, device)
    return ft_score, perf

def run_deepathnet(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device,
    task='classif'
):
    assert task is not None, "Must correspond to the configuration file in DeePathNet folder."
    from selected_models.DeePathNet.scripts.run_deeppathnet_forbk import run_deepathnet
    ft_score, perf = run_deepathnet(
        data_trn=data_trn,
        label_trn=label_trn,
        data_val=data_val,
        label_val=label_val,
        data_tst=data_tst,
        label_tst=label_tst,
        device=device,
        task=task
    )
    return ft_score, perf

def run_pathformer(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device="cuda:7",
    run_bk_identification=1
):
    from selected_models.Pathformer.Pathformer_code.run_pathformer_for_evalbk import run_pathformer
    ft_score, perf = run_pathformer(
        data_trn=data_trn,
        label_trn=label_trn,
        data_val=data_val,
        label_val=label_val,
        data_tst=data_tst,
        label_tst=label_tst,
        device=device,
        run_bk_identification=run_bk_identification
    )
    return ft_score, perf

def run_pnet(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device,
    **kwargs
):
    from selected_models.PNet.run_pnet import run_pnet
    ft_score, perf = run_pnet(data_trn, label_trn, data_val, label_val, data_tst, label_tst, device, **kwargs)
    return ft_score, perf

def run_deepkegg(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device,
    **kwargs
):
    from selected_models.DeepKEGG.run_deepkegg_forbk_pytorch import run_deepkegg
    ft_score, perf = run_deepkegg(data_trn, label_trn, data_val, label_val, data_tst, label_tst, device, **kwargs)
    return ft_score, perf

def run_gnnsubnet(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:0'
):
    from selected_models.GNNSubNet.run_gnnsubnet import run_gnnsubnet
    ft, perf = run_gnnsubnet(
        data_trn=data_trn,
        label_trn=label_trn,
        data_val=data_val,
        label_val=label_val,
        data_tst=data_tst,
        label_tst=label_tst,
        device=device
    )
    return ft, perf


#%%
###############################################################################
# stats&ml methods
###############################################################################
# MOFA
def run_mofa(
    data: pd.DataFrame,
    save_model_suffix: str | None = None,
):
    from selected_models.MOFA.run_mofa import run_mofa
    ft_score, ft_score_only_score = run_mofa(data, save_model_suffix)
    return ft_score, ft_score_only_score

# MCIA
def run_mcia(
    data: pd.DataFrame,
):
    ###### filtering
    data = data.astype(np.float32)
    # remove 0 var features
    data = data.loc[:, data.var(axis=0)>0]
    ######
    ft_names = data.columns.copy()
    from selected_models.MCIA.run_mcia import run_mcia
    ft_score = run_mcia(data)
    assert len(ft_names)==len(ft_score), "feature number mismatch after running MCIA"
    ft_score = ft_score.loc[ft_names]
    return ft_score

# GAUDI
def run_gaudi(
    data: pd.DataFrame,
) -> pd.DataFrame:
    ##### data checks
    for m, df in data.items():
        assert np.issubdtype(df.to_numpy().dtype, np.number), f"{m} not numeric"
    for m, df in data.items():
        assert not df.isna().any().any(), f"{m} contains NA"
    ## remove samples with 0 var to avoid error in gaudi()
    mdic = mod_mol_dict(data.columns)
    new_samples = set(data.index.values.astype(str))
    for mod in mdic['mods_uni']:
        cur_data = data.loc[:, mdic['mods']==mod].copy()
        cur_data = cur_data.loc[cur_data.var(axis=1)>0, :]
        new_samples = new_samples & set(cur_data.index)
    new_samples = sorted(list(new_samples))
    data = data.loc[new_samples, :].copy()
    ##### run
    from selected_models.GAUDI.run_gaudi import run_gaudi
    ft_score = run_gaudi(data)
    assert (ft_score>=0).all().all(), "GAUDI feature importance scores contain negative values."
    return ft_score

# GDF
def run_gdf(
    data_trn: pd.DataFrame,
    label_trn: pd.DataFrame,
    data_tst: pd.DataFrame,
    label_tst: pd.DataFrame):

    from selected_models.GDF.run_gdf import run_gdf

    topo = pd.read_csv("../data/STRING_PPI_data/topology_filtered0.7.csv", index_col=0)
    ppi = topo.loc[topo['combined_score']>0.95][['protein1', 'protein2']]
    ppi_gset = set(ppi.values.flatten().astype(str))

    mdic = mod_mol_dict(data_trn.columns)
    data_trn_dic = {
        mod : data_trn.loc[:, mdic['mods'] == mod] for mod in mdic['mods_uni']
    }
    data_tst_dic = {
        mod : data_tst.loc[:, mdic['mods'] == mod] for mod in mdic['mods_uni']
    }
    # strip mod@ prefix
    for mod in data_trn_dic:
        data_trn_dic[mod].columns = [c.split(SPLITTER, 1)[1] for c in data_trn_dic[mod].columns]
        data_tst_dic[mod].columns = [c.split(SPLITTER, 1)[1] for c in data_tst_dic[mod].columns]

    cover_gset = set(modmol_gene_set_tcga(data_trn.columns, op='union')) & ppi_gset
    cover_gset = np.array(sorted(list(cover_gset))) # sorted to have a fixed order
    ppi = ppi[ppi['protein1'].isin(cover_gset) & ppi['protein2'].isin(cover_gset)].reset_index(drop=True)
    # update cover_gset after filtering ppi
    cover_gset = np.array(sorted(list(set(ppi.values.flatten().astype(str)))))
    # keep only genes in cover_gset
    mappings = {
        'protein' : P2G,
        'DNAm' : C2G,
        'miRNA': R2G}
    assert data_trn_dic.keys() == data_tst_dic.keys()
    for mod in data_trn_dic:
        if mod in ['mRNA', 'CNV', 'SNV']:
            data_trn_dic[mod] = data_trn_dic[mod].loc[:, data_trn_dic[mod].columns.isin(cover_gset)]
            data_tst_dic[mod] = data_tst_dic[mod].loc[:, data_tst_dic[mod].columns.isin(cover_gset)]
        elif mod in ['DNAm', 'protein', 'miRNA']:
            mapping_cur = mappings[mod].loc[mappings[mod]['gene'].isin(cover_gset)].copy()
            data_trn_dic[mod] = data_trn_dic[mod].loc[:, data_trn_dic[mod].columns.isin(mapping_cur.index)]
            data_tst_dic[mod] = data_tst_dic[mod].loc[:, data_tst_dic[mod].columns.isin(mapping_cur.index)]
        else: raise ValueError(f"Unexpected mod {mod}")

    data_trn_dic = convert_omics_to_gene_level(data_dict=data_trn_dic, cover_gset=cover_gset)
    data_tst_dic = convert_omics_to_gene_level(data_dict=data_tst_dic, cover_gset=cover_gset)

    # align genes
    assert np.array([(data_trn_dic[mod].columns == cover_gset).all() for mod in mdic['mods_uni']]).all() \
        and np.array([(data_tst_dic[mod].columns == cover_gset).all() for mod in mdic['mods_uni']]).all()
    for mod in mdic['mods_uni']:
        data_trn_dic[mod].columns = f'{mod}@' + data_trn_dic[mod].columns
        data_tst_dic[mod].columns = f'{mod}@' + data_tst_dic[mod].columns
    data_trn = pd.concat([data_trn_dic[mod] for mod in mdic['mods_uni']], axis=1)
    data_tst = pd.concat([data_tst_dic[mod] for mod in mdic['mods_uni']], axis=1)

    ft_score, perf = run_gdf(ppi, data_trn, label_trn, data_tst, label_tst)

    return ft_score, perf

DPM_CONSTRAINTS_VECS = {
    'CNV+DNAm+mRNA' : [1, -1, 1],
    'CNV+SNV+mRNA' : [1, 0, 1],
    'CNV+mRNA+miRNA' : [1, 1, -1],
    'DNAm+SNV+mRNA' : [-1, 0, 1],
    'DNAm+mRNA+miRNA' : [-1, 1, -1],
    'SNV+mRNA+miRNA' : [0, 1, -1],
    'DNAm+mRNA+protein' : [-1, 1, 1],
}
def run_dpm(
    data,
    label,
    cv=None
):
    for k, v in DPM_CONSTRAINTS_VECS.items():
        assert (np.sort(k.split("+"))==k.split("+")).all()

    mdic = mod_mol_dict(np.asarray(data.columns))
    assert (np.sort(mdic['mods_uni'])==mdic['mods_uni']).all()
    if cv is None:
        cv = DPM_CONSTRAINTS_VECS['+'.join(mdic['mods_uni'])]
    label['label'] = factorize_label(label.iloc[:, 0])[0]

    # unify genes
    gset = modmol_gene_set_tcga(
        data.columns,
        op='intersection',
        c2g=C2G,
        p2g=P2G,
        r2g=R2G
    )
    data.columns = data.columns.str.split(SPLITTER).str[1]
    data = {k: data.loc[:, mdic['mods'] == k] for k in mdic['mods_uni']}
    data = convert_omics_to_gene_level(data, gset)

    from selected_models.DPM.run_dpm import run_dpm
    ft_score = run_dpm(data, label, cv=cv)
    return ft_score


# Stabl
def run_stabl(
    data_trn: pd.DataFrame,
    data_tst: pd.DataFrame,
    label_trn: pd.DataFrame,
    label_tst: pd.DataFrame,
):
    """
    Note:
        modifed knockpy utilities.py line 115
        
        # NOTE modified by athan li 08142025 to avoid "AttributeError: `scipy.sparse.linalg.eigen` has no attribute `arpack`" error.
        # except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
        except:
            print("scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence")
            return np.linalg.eigh(M)[0].min()
    """
    from stabl.stabl import Stabl
    from sklearn.linear_model import LogisticRegression

    mdic = mod_mol_dict(data_trn.columns)
    Xtr_blocks = {}
    Xte_blocks = {}
    for mod in mdic['mods_uni']:
        Xtr_blocks[mod] = data_trn.loc[:, mdic['mods']==mod]
        Xte_blocks[mod] = data_tst.loc[:, mdic['mods']==mod]
    
    # Encode labels
    label_trn, _ = factorize_label(label_trn.values.flatten())
    label_tst, _ = factorize_label(label_tst.values.flatten())

    ###########################################################################
    # run stabl
    ###########################################################################
    st_time = time.perf_counter()
    per_omic = {}
    for omic, Xtr in Xtr_blocks.items():
        st = Stabl(artificial_type='knockoff') # as in the paper. this is for internally correlated data for most omics types, the authors only ever applied random_permutation to cfRNA which has low internal correlations.
        st.fit(Xtr, np.asarray(label_trn).ravel())
        # feature scores = max selection frequency across lambda (paper’s stability path score)
        scores = pd.Series(st.get_importances(), index=st.feature_names_in_, name="score") # # guaranteed to match order
        # selected names = frequency >= theta (theta chosen by FDP+ minimization inside Stabl)
        selected = list(st.get_feature_names_out())   # uses st.fdr_min_threshold_ internally
        per_omic[omic] = {
            "model": st,
            "scores": scores,                 # per-feature frequency
            "selected_names": selected,       # reliable set 
            "theta": st.fdr_min_threshold_,   # the omic-specific theta
        }
    print("Stabl running time (s):", time.perf_counter() - st_time)
    # fusion: concatenate only selected features from each omic, and train a final logistic learner
    Xtr_sel = pd.concat([Xtr_blocks[o][per_omic[o]["selected_names"]] for o in Xtr_blocks], axis=1)
    Xte_sel = pd.concat([Xte_blocks[o][per_omic[o]["selected_names"]] for o in Xte_blocks], axis=1)
    print({o: per_omic[o]["theta"] for o in per_omic})
    print({o: len(per_omic[o]["selected_names"]) for o in per_omic})
    print("Xtr_sel shape:", getattr(Xtr_sel, "shape", None))
    if Xtr_sel.shape[1]>0:
        clf = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=1000000)
        clf.fit(Xtr_sel, np.asarray(label_trn).ravel())
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ####### ft_score
    ft_score = pd.DataFrame(index=data_trn.columns, columns=['score'], dtype=float, data=0.0)
    for o, vals in per_omic.items():
        ft_score.loc[vals['scores'].index, 'score'] = vals['scores'].values.flatten().astype(float)

    if Xtr_sel.shape[1]>0:
        ####### perf
        assert np.unique(label_trn).shape[0]==2
        y_proba = clf.predict_proba(Xte_sel)
        y_pred  = clf.predict(Xte_sel)

        task_is_biclassif = (y_proba.ndim == 2 and y_proba.shape[1] == 2)
        y_pred = y_pred.astype(int)
        y_true = label_tst

        roc_auc = aucpr = recall = precision = f1 = f1_weighted = f1_macro = mcc = balanced_acc = None
        if task_is_biclassif:
            y_proba = y_proba[:, 1]
            roc_auc = roc_auc_score(y_true, y_proba)
            aucpr   = average_precision_score(y_true, y_proba)
            recall  = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            f1      = f1_score(y_true, y_pred)
            mcc     = matthews_corrcoef(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            print(f"AUC-ROC:        {roc_auc:.4f}")
            print(f"AUCPR:          {aucpr:.4f}")
            print(f"F1:             {f1:.4f}")
            print(f"Precision:      {precision:.4f}")
            print(f"Recall:         {recall:.4f}")
            print(f"MCC:            {mcc:.4f}")
        else:
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            f1_macro    = f1_score(y_true, y_pred, average='macro')
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
        print("Performance:", perf)
    else:
        perf = {
            'acc': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'f1_weighted': None,
            'f1_macro': None,
            'roc_auc': None,
            'aucpr': None,
            'mcc': None,
            'balanced_acc': None,
        }
    return ft_score, perf

# DIABLO
def run_diablo(
    data_trn,
    label_trn,
    data_tst,
    label_tst):
    from selected_models.DIABLO.run_diablo import run_diablo
    ft_score, ft_score_rank, perf = run_diablo(
        data_trn=data_trn,
        label_trn=label_trn,
        data_tst=data_tst,
        label_tst=label_tst,
    )
    return ft_score, ft_score_rank, perf # NOTE

# asmPLS-DA
def run_asmplsda(
    data_trn,
    label_trn,
    data_tst,
    label_tst):
    from selected_models.asmPLSDA.run_asmplsda import run_asmplsda
    ft_score, ft_score_rank, perf = run_asmplsda(
        data_trn=data_trn,
        label_trn=label_trn,
        data_tst=data_tst,
        label_tst=label_tst,
    )
    return ft_score, ft_score_rank, perf # NOTE

#%%
###############################################################################
# classical
###############################################################################
# def run_mannwhitneyu(data, label):
#     """
#     Performs Mann-Whitney U tests between two classes for each feature.

#     """
#     ft_score = pd.DataFrame(index=data.columns, columns=['score'])
#     pvals = []
#     label = label.values.flatten()
#     label_uni = np.unique(label)
#     assert len(label_uni) == 2, "Only two classes are supported."
#     mask1 = label==label_uni[0]
#     mask2 = label==label_uni[1]
#     st_time = time.perf_counter()
#     for i in range(len(data.columns)):
#         group1 = data.iloc[mask1, i]
#         group2 = data.iloc[mask2, i]
#         u_stat, p_val = mannwhitneyu(group1, group2)
#         pvals.append(p_val)
#     print("Mann-Whitney U test running time (s):", time.perf_counter() - st_time)
#     ft_score['score'] = 1 - np.array(pvals)
#     return ft_score

# def run_ttest(data, label):
#     """
#     Performs ttests between two classes for each feature.

#     """
#     ft_score = pd.DataFrame(index=data.columns, columns=['score'])
#     pvals = []
#     label = label.values.flatten()
#     label_uni = np.unique(label)
#     assert len(label_uni) == 2, "Only two classes are supported."
#     mask1 = label==label_uni[0]
#     mask2 = label==label_uni[1]
#     st_time = time.perf_counter()
#     for i in range(len(data.columns)):
#         group1 = data.iloc[mask1, i]
#         group2 = data.iloc[mask2, i]
#         u_stat, p_val = ttest_ind(group1, group2, equal_var=False)
#         pvals.append(p_val)
#     print("t-test running time (s):", time.perf_counter() - st_time)
#     ft_score['score'] = 1 - np.array(pvals)
#     return ft_score

def run_SVM_ONE_and_SVM_RFE(data, label, data_tst=None, label_tst=None,
                            n_features_to_select=0.001, C=1.0, max_iter=10000,
                            step=0.001):
    from sklearn.feature_selection import RFE

    X = data.values
    y = label.values.flatten() if hasattr(label, "values") else np.array(label)
    y, v2i = factorize_label(y)
    y_tst, _ = factorize_label(label_tst.values.flatten())

    from sklearn.svm import LinearSVC

    estimator = LinearSVC(C=C, max_iter=max_iter)
    st_time = time.perf_counter()
    estimator.fit(X, y)
    print("SVM training time (s):", time.perf_counter() - st_time)

    ## ONE
    st_time = time.perf_counter()
    label_uni = np.unique(label.values.flatten())
    if len(label_uni) > 2:
        ft_score = pd.DataFrame({f'score_{l}': 0 for l in label_uni}, index=data.columns)
        for i, l in enumerate(label_uni):
            ft_score.loc[:, f'score_{l}'] = estimator.coef_[v2i[l]]
    elif len(label_uni) == 2:
        ft_score = pd.DataFrame({f'score': 0}, index=data.columns)
        ft_score.loc[:, f'score'] = estimator.coef_[0]
    else:
        raise ValueError("requires at least 2 unique labels.")
    print("SVM_ONE running time (s):", time.perf_counter() - st_time)
    ft_score_ONE = ft_score.copy()
    ## RFE
    st_time = time.perf_counter()
    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(data_tst.values, y_tst)
    ranking = rfe.ranking_
    ft_score = pd.DataFrame({'score': ranking}, index=data.columns)
    print("SVM_RFE BK identification running time (s):", time.perf_counter() - st_time)
    ft_score_RFE = ft_score.copy()

    perf = None
    if data_tst is not None and label_tst is not None:
        try:
            y_prob = estimator.predict_proba(data_tst.values)
        except AttributeError:
            decision = estimator.decision_function(data_tst.values)
            if decision.ndim == 1:
                prob = 1 / (1 + np.exp(-decision))
                y_prob = np.vstack([1 - prob, prob]).T
            else:
                exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                y_prob = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
        
        y_true = y_tst
        y_pred = np.argmax(y_prob, axis=1)
        task_is_biclassif = np.unique(y_true).shape[0] == 2
        
        from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, accuracy_score
        roc_auc = aucpr = recall = precision = f1 = f1_weighted = f1_macro = None
        if task_is_biclassif:
            y_proba = y_prob[:, -1]
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
            'mcc' : mcc if task_is_biclassif else None,
            'balanced_acc': balanced_acc if task_is_biclassif else None,
        }
        print("Performance:", perf)
    return [ft_score_ONE, ft_score_RFE], perf

def run_RF_VI_and_RF_RFE(
    data, label, data_tst=None, label_tst=None, 
    n_features_to_select=0.001, step=0.001,
    n_estimators=1000,
    random_state=42):
    """
    Uses RF-based Recursive Feature Elimination (RF-RFE) to rank features.
    The estimator used in RFE is a RandomForestClassifier.
    """
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier

    X = data.values
    y = label.values.flatten() if hasattr(label, "values") else np.array(label)
    y, _ = factorize_label(y)
    y_tst, _ = factorize_label(label_tst.values.flatten())

    estimator = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # VI
    st_time = time.perf_counter()
    estimator.fit(X, y)
    importances = estimator.feature_importances_
    ft_score_VI = pd.DataFrame({'score': importances}, index=data.columns)
    print("RF-VI running time (s):", time.perf_counter() - st_time)

    # RFE
    st_time = time.perf_counter()
    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(data_tst.values, y_tst)
    ft_score_RFE = pd.DataFrame({'score': rfe.ranking_}, index=data.columns)
    print("RF_RFE running time (s):", time.perf_counter() - st_time)

    perf = None
    if data_tst is not None and label_tst is not None:
        try:
            y_prob = estimator.predict_proba(data_tst.values)
        except AttributeError:
            decision = estimator.decision_function(data_tst.values)
            if decision.ndim == 1:  # binary classification
                prob = 1 / (1 + np.exp(-decision))
                y_prob = np.vstack([1 - prob, prob]).T
            else:
                exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                y_prob = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
        
        y_true = y_tst
        label_uni = np.unique(y_true)
        y_pred = np.argmax(y_prob, axis=1)
        task_is_biclassif = (label_uni.shape[0] == 2)
        
        roc_auc = aucpr = recall = precision = f1 = f1_weighted = f1_macro = None
        
        if task_is_biclassif:
            y_proba = y_prob[:, -1]
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
            'mcc' : mcc if task_is_biclassif else None,
            'balanced_acc': balanced_acc if task_is_biclassif else None,
        }
        print("Performance:", perf)
    
    return [ft_score_VI, ft_score_RFE], perf

