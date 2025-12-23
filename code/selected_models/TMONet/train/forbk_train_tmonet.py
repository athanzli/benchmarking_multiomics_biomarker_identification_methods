from torch.nn.utils import clip_grad_norm_
CLIPPING_MAX_NORM = 10.0 # TODO NOTE
import torch

# sys.path.append('/home/athan.li/eval_bk/code/selected_models/TMO-Net')

from ..dataset.dataset import CancerDataset
from ..model.TMO_Net_model import TMO_Net, DownStream_predictor, dfs_freeze, un_dfs_freeze
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from ..util.loss_function import cox_loss
from lifelines.utils import concordance_index
import pickle
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,balanced_accuracy_score # NOTE
from .LogME import LogME
logme = LogME(regression=False)
import copy

def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(66)

# pretrain
def train_pretrain(train_dataloader, model, epoch, cancer, optimizer, dsc_optimizer, fold, pretrain_omics):
    model.train()
    # model = model.state_dict()
    print(f'-----start epoch {epoch} training-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_ad_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()

    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time,  omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.flatten()
            all_label = torch.concat((all_label, cancer_label), dim=0)

            input_x = []
            for key in omics_data.keys():
                omic = omics_data[key]
                omic = omic.cuda()
                # print(omic)
                input_x.append(omic)
            un_dfs_freeze(model.discriminator)
            un_dfs_freeze(model.infer_discriminator)
            cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0), pretrain_omics)
            # recon_omics = model.cross_modal_generation(input_x[:2], incomplete_omics)
            ad_loss = cross_infer_dsc_loss + dsc_loss
            total_ad_loss += dsc_loss.item()

            dsc_optimizer.zero_grad()
            ad_loss.backward(retain_graph=True)
            disc_params = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
            clip_grad_norm_(disc_params, max_norm=CLIPPING_MAX_NORM) 
            dsc_optimizer.step()

            dfs_freeze(model.discriminator)
            dfs_freeze(model.infer_discriminator)
            loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x, os_event.size(0), pretrain_omics)

            total_self_elbo += self_elbo.item()
            total_cross_elbo += cross_elbo.item()
            total_cross_infer_loss += cross_infer_loss.item()
            multi_embedding = model.get_embedding(input_x, os_event.size(0), pretrain_omics)

            pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)
            contrastive_loss = model.contrastive_loss(multi_embedding, cancer_label)
            loss += contrastive_loss
            # loss = ce_loss
            total_dsc_loss += dsc_loss.item()
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=CLIPPING_MAX_NORM) 
            optimizer.step()
            
            
            tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                               cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

        # print('total loss: ', total_loss / len(train_dataloader))
        Loss.append(total_loss / len(train_dataloader))
        # print('self elbo loss: ', total_self_elbo / len(train_dataloader))
        Loss.append(total_self_elbo / len(train_dataloader))
        # print('cross elbo loss: ', total_cross_elbo / len(train_dataloader))
        Loss.append(total_cross_elbo / len(train_dataloader))
        # print('cross infer loss: ', total_cross_infer_loss / len(train_dataloader))
        Loss.append(total_cross_infer_loss / len(train_dataloader))
        # print('ad loss', total_ad_loss / len(train_dataloader))
        # print('dsc loss', total_dsc_loss / len(train_dataloader))
        Loss.append(total_dsc_loss / len(train_dataloader))

        pretrain_score = logme.fit(pancancer_embedding.detach().cpu().numpy(), all_label.cpu().numpy())
        # print('pretrain score:', pretrain_score)
        return Loss, pretrain_score


def val_pretrain(test_dataloader, model, epoch, cancer, fold, pretrain_omics):
    model.eval()
    # model = model.state_dict()
    print(f'-----start epoch {epoch} val-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()
    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_event, os_time, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.flatten()
                all_label = torch.concat((all_label, cancer_label), dim=0)
                input_x = []
                for key in omics_data.keys():
                    omic = omics_data[key]
                    omic = omic.cuda()
                    input_x.append(omic)

                cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0), pretrain_omics)

                total_cross_infer_dsc_loss += cross_infer_dsc_loss.item()

                loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x, os_event.size(0), pretrain_omics)
                multi_embedding = model.get_embedding(input_x, os_event.size(0), pretrain_omics)

                pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)
                total_self_elbo += self_elbo.item()
                total_cross_elbo += cross_elbo.item()
                total_cross_infer_loss += cross_infer_loss.item()

                total_dsc_loss += dsc_loss.item()
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                                   cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

            # print('test total loss: ', total_loss / len(test_dataloader))
            Loss.append(total_loss / len(test_dataloader))
            # print('test self elbo loss: ', total_self_elbo / len(test_dataloader))
            Loss.append(total_self_elbo / len(test_dataloader))
            # print('test cross elbo loss: ', total_cross_elbo / len(test_dataloader))
            Loss.append(total_cross_elbo / len(test_dataloader))
            # print('test cross infer loss: ', total_cross_infer_loss / len(test_dataloader))
            Loss.append(total_cross_infer_loss / len(test_dataloader))
            # print('test ad loss', total_cross_infer_dsc_loss / len(test_dataloader))
            # print('test dsc loss', total_dsc_loss / len(test_dataloader))
            Loss.append(total_dsc_loss / len(test_dataloader))

            pretrain_score = logme.fit(pancancer_embedding.detach().cpu().numpy(), all_label.cpu().numpy())
            # print('pretrain score:', pretrain_score)
    return Loss

#%%
TMONET_PATH = "/home/athan.li/eval_bk/code/selected_models/TMONet/"

def TCGA_Dataset_pretrain(
    omics,
    num_mods, mods_dim, omics_data_type,
    train_dataset, test_dataset, # NOTE added
    fold, epochs, device_id
):
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = TMO_Net(num_mods, mods_dim, 64,
                    [2048, 512], [512, 2048], omics_data_type, 0.01)
    torch.cuda.set_device(device_id)
    model.cuda()
    print(len(train_dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    dsc_parameters = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
    dsc_optimizer = torch.optim.Adam(dsc_parameters, lr=1e-4, weight_decay=5e-4)

    Loss_list = []
    test_Loss_list = []
    pretrain_score_list = []

    total_train_time = 0.0

    for epoch in range(epochs):
        prev_model_dict = model.state_dict()
        st_time = time.perf_counter()
        loss, pretrain_score = train_pretrain(
            train_dataloader,
            model,
            epoch,
            'PanCancer',
            optimizer,
            dsc_optimizer,
            fold,
            omics)
        total_train_time += (time.perf_counter() - st_time)

        Loss_list.append(loss)
        pretrain_score_list.append(pretrain_score)

        test_Loss_list.append(0) # NOTE
    model_dict = model.state_dict()

    print(f"TMO-Net pretraining phase Training time for {epoch} epochs: {total_train_time:.2f}s")

    Loss_list = torch.Tensor(Loss_list)
    test_Loss_list = torch.Tensor(test_Loss_list)
    pretrain_score_list = pd.DataFrame(pretrain_score_list, columns=['pretrain_score'])
    
    return Loss_list, test_Loss_list, pretrain_score_list, model_dict

def train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics):
    model.train()
    print(f'-----start {cancer} epoch {epoch} training-----')
    total_loss = 0
    train_risk_score = torch.Tensor([]).cuda()
    train_censors = torch.Tensor([]).cuda()
    train_event_times = torch.Tensor([]).cuda()
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time, omics_data, _ = data
            os_event = os_event.cuda()
            os_time = os_time.cuda()
            train_censors = torch.concat((train_censors, os_event))
            train_event_times = torch.concat((train_event_times, os_time))

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            risk_score = model(input_x, os_event.size(0), omics)
            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x, os_event.size(0), omics)
            train_risk_score = torch.concat((train_risk_score, risk_score))
            CoxLoss = cox_loss(os_time, os_event, risk_score)
            loss = CoxLoss
            total_loss += CoxLoss.item()
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=CLIPPING_MAX_NORM) 
            optimizer.step()
            
            tepoch.set_postfix(loss=CoxLoss.item())

        print('cox loss: ', total_loss / len(train_dataloader))
        train_c_index = concordance_index(train_event_times.detach().cpu().numpy(),
                                          -train_risk_score.detach().cpu().numpy(),
                                          train_censors.detach().cpu().numpy())

        print(f'{cancer} train survival c-index: ', train_c_index)


def cal_c_index(dataloader, model, best_c_index, best_c_indices, omics):
    model.eval()
    best_c_index_list = best_c_indices
    device = next(model.parameters()).device

    with torch.no_grad():
        risk_scores = {
            'all': torch.zeros((len(dataloader)), device=device),
        }

        censors = torch.zeros((len(dataloader)))
        event_times = torch.zeros((len(dataloader)))

        for i, data in enumerate(dataloader):
            os_event, os_time, omics_data, cancer_label = data
            input_x = [omics_data[key].cuda() for key in omics_data.keys()]
            os_event = os_event.cuda()
            os_time = os_time.cuda()

            survival_risk = model(input_x, os_event.size(0), omics)

            risk_scores['all'][i] = survival_risk
            censors[i] = os_event
            event_times[i] = os_time

        c_indices = {}
        for key in risk_scores.keys():
            c_indices[key] = concordance_index(event_times.cpu().numpy(), -risk_scores[key].cpu().numpy(),
                                               censors.cpu().numpy())

        if c_indices['all'] > best_c_index:
            best_c_index = c_indices['all']
            best_c_index_list = [c_indices[key] for key in c_indices.keys()]

        print(f'test survival c-index: ', c_indices)

    return best_c_index_list, best_c_index


def TCGA_Dataset_survival_prediction(fold, epochs, cancer_types, pretrain_model_path, fixed, device_id):
    pancancer_c_index = []
    for cancer in cancer_types:
        # print(cancer)
        train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                      fold + 1, [cancer])
        test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path,
                                     fold + 1, [cancer])

        train_dataloader = DataLoader(train_dataset, batch_size=32, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        task = {'output_dim': 1}
        model = DownStream_predictor(4, [6016, 6617, 4539, 7460], 64, [2048, 512],
                                     [512, 2048], pretrain_model_path, task, omics_data_type, fixed, omics, 0.01)
        torch.cuda.set_device(device_id)
        model.cuda()
        param_groups = [
            {'params': model.cross_encoders.parameters(), 'lr': 0.0001},
            {'params': model.downstream_predictor.parameters(), 'lr': 0.0001},

        ]
        optimizer = torch.optim.Adam(param_groups)
        best_c_index = 0
        best_c_indices = []
        for epoch in range(epochs):
            # adjust_learning_rate(optimizer, epoch, 0.0001)
            start_time = time.time()
            train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics)
            best_c_indices, best_c_index = cal_c_index(test_dataloader, model, best_c_index, best_c_indices, omics)
            # print(f'{fold} time used: ', time.time() - start_time)
        best_c_indices.insert(0, cancer)
        pancancer_c_index.append(best_c_indices)

        #   clean memory of gpu cuda
        del model
        del optimizer
        del train_dataloader
        del test_dataloader
        torch.cuda.empty_cache()

    pancancer_c_index = pd.DataFrame(pancancer_c_index,
                                     columns=['cancer', 'multiomics'])

    pancancer_c_index.to_csv(f'all_pancancer_pretrain_cross_encoders_c_index_fold{fold}_all_omics.csv',
                             encoding='utf-8')


#   cancer classification
def train_classification(dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.train()
    pancancer_embedding = torch.Tensor([]).cuda()
    with tqdm(dataloader, unit='batch') as tepoch:
        total_samples = 0
        all_labels = []
        all_predictions = []

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time, omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.flatten()

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            classification_pred = model(
                input_x,
                os_event.size(0),
                omics)
            _, labels_pred = torch.max(classification_pred, 1)

            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(
                input_x,
                os_event.size(0),
                omics)
            pred_loss = criterion(classification_pred, cancer_label)
            total_loss += pred_loss.item()
            loss = pred_loss

            embedding_tensor = model.cross_encoders.get_embedding(input_x, os_event.size(0), omics)
            pancancer_embedding = torch.concat((pancancer_embedding, embedding_tensor), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_samples += cancer_label.size(0)
            all_labels.extend(cancer_label.tolist())
            all_predictions.extend(labels_pred.tolist())

            tepoch.set_postfix(loss=pred_loss.item())

        f1weighted = f1_score(all_labels, all_predictions, average='weighted')

        all_labels = torch.Tensor(all_labels)


def test_classification(dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.eval()
    pancancer_embedding = torch.Tensor([]).cuda()
    with torch.no_grad():
        with tqdm(dataloader, unit='batch') as tepoch:
            total_samples = 0
            all_labels = []
            all_predictions = []
            y_pred_probs = []

            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_event, os_time, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.flatten()

                input_x = [omics_data[key].cuda() for key in omics_data.keys()]

                embedding_tensor = model.cross_encoders.get_embedding(
                    input_x,
                    os_event.size(0),
                    omics)
                pancancer_embedding = torch.concat((pancancer_embedding, embedding_tensor), dim=0)

                classification_pred = model(
                    input_x,
                    os_event.size(0),
                    omics)
                _, labels_pred = torch.max(classification_pred, 1)
                y_pred_proba = classification_pred[:, -1]

                pred_loss = criterion(classification_pred, cancer_label)
                total_loss += pred_loss.item()

                total_samples += cancer_label.size(0)
                all_labels.extend(cancer_label.tolist())
                # print('test pred label', labels_pred)
                all_predictions.extend(labels_pred.tolist())
                y_pred_probs.extend(y_pred_proba.tolist()) # NOTE

                tepoch.set_postfix(loss=pred_loss.item())


            f1weighted = f1_score(all_labels, all_predictions, average='weighted')
            # print('fold {} test:, F1 weighted: {:.4f}'.format(fold, f1weighted))
            perf = f1weighted

            all_labels = torch.Tensor(all_labels)

            ################################ perf ################################
            y_true = all_labels.cpu().numpy()
            y_pred = all_predictions
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
                y_proba = y_pred_probs
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

            return perf, total_loss # NOTE added loss

def TCGA_Dataset_classification(
    omics,
    num_mods, mods_dim, omics_data_type,
    train_dataset, test_dataset, # NOTE added   
    fold, epochs, pretrain_model_dict, fixed, device_id,
    class_weights=None # NOTE added
):

    # class weights

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    
    num_classes = len(np.unique(train_dataset.clinical_data['Cancer_Type']))
    num_mods = len(mods_dim)
    task = {'output_dim': num_classes} # NOTE changed

    model = DownStream_predictor(
        num_mods,
        mods_dim,
        64,
        [2048, 512],
        [512, 2048],
        pretrain_model_dict,
        task,
        omics_data_type,
        fixed,
        omics,
        0.01)

    torch.cuda.set_device(device_id)
    model.cuda()
    param_groups = [
        {'params': model.cross_encoders.parameters(), 'lr': 0.00001},
        {'params': model.downstream_predictor.parameters(), 'lr': 0.0001},
    ]

    optimizer = torch.optim.Adam(param_groups)
    best_perf = 0
    classification_score = []

    # ------ early stopping ------
    best_loss = float('inf')
    patience = 100 # NOTE
    patience_counter = 0
    best_model_dict = None
    # ------ early stopping ------


    # ------ running time ------
    total_train_time = 0.0
    # --------------------------

    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch, 0.0001)

        start_time = time.time()
        st_time = time.perf_counter()
        train_classification(
            train_dataloader,
            model,
            epoch,
            'pancancer',
            fold,
            optimizer,
            omics,
            criterion)
        
        # NOTE
        total_train_time += (time.perf_counter() - st_time)

        perf, val_loss = test_classification( # NOTE added loss
            test_dataloader,
            model,
            epoch,
            'pancancer',
            fold,
            optimizer,
            omics,
            criterion)
        print(f"epoch {epoch} val_loss: {val_loss:.4f}") # NOTE added


        # ------ early stopping ------
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), TMONET_PATH + f'model/model_dict/forearlystopping_trainingphase_best_model_fold{fold}_{early_stopping_best_model_saving_suffix}.pt')
            best_model_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        # ------ early stopping ------

    # ------ running time ------
    print(f"TMO-Net training phase Training time for {epochs} epochs: {total_train_time:.2f}s")
    model.load_state_dict(best_model_dict)
    # model.load_state_dict(torch.load(TMONET_PATH + f'model/model_dict/forearlystopping_trainingphase_best_model_fold{fold}_{early_stopping_best_model_saving_suffix}.pt')) # NOTE

    #   clean memory of gpu cuda
    del optimizer
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    
    model.eval()
    return model, perf

#   multiprocessing pretrain_fold
def multiprocessing_train_fold(folds, function, func_args_list):
    processes = []
    for i in range(folds):
        p = mp.Process(target=function, args=func_args_list[i])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


