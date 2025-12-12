# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the CustOMICS module.
"""
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd # NOTE

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

from torch.optim import Adam

import shap

from sklearn.metrics import f1_score # NOTE added

from ..loss.survival_loss import CoxLoss # NOTE changed to relative imports. originally: src.
from ..datasets.multi_omics_dataset import MultiOmicsDataset # NOTE changed to relative imports. originally: src.
from ..models.autoencoder import AutoEncoder # NOTE changed to relative imports. originally: src.
from ..encoders.encoder import Encoder # NOTE changed to relative imports. originally: src.
from ..decoders.decoder import Decoder # NOTE changed to relative imports. originally: src.
from ..encoders.probabilistic_encoder import ProbabilisticEncoder # NOTE changed to relative imports. originally: src.
from ..decoders.probabilistic_decoder import ProbabilisticDecoder # NOTE changed to relative imports. originally: src.
from ..tasks.classification import MultiClassifier # NOTE changed to relative imports. originally: src.
from ..tasks.survival import SurvivalNet # NOTE changed to relative imports. originally: src.
from ..models.vae import VAE # NOTE changed to relative imports. originally: src.
from ..loss.classification_loss import classification_loss # NOTE changed to relative imports. originally: src.
from ..metrics.classification import multi_classification_evaluation, plot_roc_multiclass # NOTE changed to relative imports. originally: src.
from ..metrics.survival import CIndex_lifeline, cox_log_rank # NOTE changed to relative imports. originally: src.
from ..tools.utils import save_plot_score # NOTE changed to relative imports. originally: src.
from ..tools.utils import get_common_samples # NOTE changed to relative imports. originally: src.
from ..ex_vae.shap_vae import processPhenotypeDataForSamples, randomTrainingSample, splitExprandSample, ModelWrapper, addToTensor # NOTE changed to relative imports. originally: src.
from lifelines import KaplanMeierFitter

# NOTE
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, matthews_corrcoef, recall_score, precision_score, f1_score, balanced_accuracy_score

# NOTE
def chunked_run(
        func,
        input,
        output_size,
        batch_size,
        **kwargs
):
    r"""
    NOTE assuming first argument is the data input. may vary depending on the function arguments.

    Args:
        - output_size (Tuple)
    """
    assert batch_size > 0
    assert input is not None
    assert func is not None
    assert output_size is not None

    print("Running batches...")
    output = torch.zeros(size=output_size).to(input.device)
    batch_size = min(batch_size, input.shape[0]) # NOTE
    n_chunks = input.shape[0] // batch_size
    for i in range(n_chunks):
        chunk = input[i*batch_size:(i+1)*batch_size]
        output_chunk = func(chunk, **kwargs)
        if not isinstance(output_chunk, torch.Tensor):
            output_chunk = torch.tensor(output_chunk).to(input.device)
        output[i*batch_size:(i+1)*batch_size] = output_chunk
    return output

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})


class CustOMICS(nn.Module):
    """
    The main CustOMICS object that represents the main network for dealing with multi-source integration and multi-task learning
    """
    def __init__(self, source_params, central_params, classif_params, surv_params, train_params, device):
        """
        Construct the whole architecture with intermediate autoencoders, central layer and eventually downstream predictors
        Parameters:
            source_params (dict)      -- parameters related to the different sources to integrate
            central_params (dict)     -- parameters of the central autoencoder
            classif_params (dict)     -- classifier parameters
            surv_params (dict)        -- parameters of the survival network
            train_params (dict)       -- training hyperparameters
            device (pytorch)          -- the device in which the computation is done
        """
        super(CustOMICS, self).__init__()
        self.n_source = len(list(source_params.keys()))
        self.device = device
        self.lt_encoders = [Encoder(input_dim=source_params[source]['input_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             latent_dim=source_params[source]['latent_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.lt_decoders = [Decoder(latent_dim=source_params[source]['latent_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             output_dim=source_params[source]['input_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.rep_dim = sum([source_params[source]['latent_dim'] for source in source_params])
        self.central_encoder = ProbabilisticEncoder(input_dim=self.rep_dim, hidden_dim=central_params['hidden_dim'], 
                                                    latent_dim=central_params['latent_dim'], norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(latent_dim=central_params['latent_dim'], hidden_dim=central_params['hidden_dim'], 
                                                    output_dim=self.rep_dim, norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.beta = central_params['beta']
        self.num_classes = classif_params['n_class']
        self.lambda_classif = classif_params['lambda']
        self.classifier =  MultiClassifier(n_class=self.num_classes, latent_dim=central_params['latent_dim'], dropout=classif_params['dropout'],
            class_dim = classif_params['hidden_layers']).to(self.device)
        self.lambda_survival = surv_params['lambda']
        surv_param = {'drop': surv_params['dropout'], 'norm': surv_params['norm'], 'dims': [central_params['latent_dim']] + surv_params['dims'] + [1], 
                    'activation': surv_params['activation'], 'l2_reg': surv_params['l2_reg'], 'device': self.device}
        self.survival_predictor = SurvivalNet(surv_param).to(self.device) # NOTE added to device
        self.phase = 1
        self.switch_epoch = train_params['switch']
        self.lr = train_params['lr']
        self.autoencoders = []
        self.central_layer = None
        self._set_autoencoders()
        self._set_central_layer()
        self._relocate()
        self.optimizer = self._get_optimizer(self.lr)
        self.vae_history = []
        self.survival_history = []
        self.label_encoder = None
        self.one_hot_encoder = None
        self.source_names = list(source_params.keys()) # NOTE added

    def _get_optimizer(self, lr):
        """
        Initilizes the optimizer
        Parameters:
            lr (float)      -- learning rate for the CustOmics network
        """
        lt_params = []
        for autoencoder in self.autoencoders:
            lt_params += list(autoencoder.parameters())
        lt_params += list(self.central_layer.parameters())
        lt_params += list(self.survival_predictor.parameters())
        lt_params += list(self.classifier.parameters())        
        optimizer = Adam(lt_params, lr=lr)
        return optimizer

    def _set_autoencoders(self):
        """
        Initializes the autoencoders
        """
        for i in range(self.n_source):
            self.autoencoders.append(AutoEncoder(self.lt_encoders[i], self.lt_decoders[i], self.device))

    def _set_central_layer(self):
        """
        Initializes the central variational autoencoder
        """
        self.central_layer = VAE(self.central_encoder, self.central_decoder, self.device)

    def _relocate(self):
        """
        Relocates the network to specified device
        """
        for i in range(self.n_source):
            self.autoencoders[i].to(self.device)
        self.central_layer.to(self.device)

    def _switch_phase(self, epoch):
        """
        Switches phases during training
        Parameters:
            epoch (int)      -- epoch starting which to switch phases
        """
        if epoch < self.switch_epoch:
            self.phase = 1
        else:
            self.phase = 2

    def _compute_baseline(self, clinical_df, lt_samples, event, surv_time):
        kmf = KaplanMeierFitter()
        kmf.fit(clinical_df.loc[lt_samples, surv_time], clinical_df.loc[lt_samples, event])
        return kmf.survival_function_


    def per_source_forward(self, x):
        lt_forward = []
        for i in range(self.n_source):
            lt_forward.append(self.autoencoders[i](x[i]))
        return lt_forward

    def get_per_source_representation(self, x):
        lt_rep = []
        for i in range(self.n_source):
            lt_rep.append(self.autoencoders[i](x[i])[1])
        return lt_rep

    def decode_per_source_representation(self, lt_rep):
        lt_hat = []
        for i in range(self.n_source):
            lt_hat.append(self.autoencoders[i].decode(lt_rep[i]))
        return lt_hat



    def forward(self, x):
        lt_forward = self.per_source_forward(x)
        lt_hat = [element[0] for element in lt_forward]
        lt_rep = [element[1] for element in lt_forward]
        central_concat = torch.cat(lt_rep, dim=1)
        mean, logvar = self.central_encoder(central_concat)
        return lt_hat, lt_rep ,mean

    def _compute_loss(self, x):
        if self.phase == 1:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            return lt_rep, loss
        elif self.phase == 2:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            central_concat = torch.cat(lt_rep, dim=1)
            loss += self.central_layer.loss(central_concat, self.beta)
            mean, logvar = self.central_encoder(central_concat)
            z = mean
            return z, loss

    def _train_loop(self, x, labels, os_time, os_event, class_weights): # NOTE added class_weights
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
        # NOTE 
        labels = labels.to(self.device)

        loss = 0
        self.optimizer.zero_grad()
        if self.phase == 1:
            lt_rep, loss = self._compute_loss(x)
            for z in lt_rep:
                hazard_pred = self.survival_predictor(z)
                survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                loss += self.lambda_survival * survival_loss
                y_pred_proba = self.classifier(z)
                classification = classification_loss('CE', y_pred_proba, labels, class_weights=class_weights) # NOTE added class_weights
                loss += self.lambda_classif * classification

        elif self.phase == 2:
            z, loss = self._compute_loss(x)
            hazard_pred = self.survival_predictor(z)
            survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
            loss += self.lambda_survival * survival_loss
            y_pred_proba = self.classifier(z)
            classification = classification_loss('CE', y_pred_proba, labels, class_weights=class_weights) # NOTE added class_weights
            loss += self.lambda_classif * classification

        return loss


    def fit(self, omics_train, clinical_df, label, event, surv_time, omics_val=None, batch_size=32, n_epochs=30, verbose=False):
        
        encoded_clinical_df = clinical_df.copy()
        self.label_encoder = LabelEncoder().fit(encoded_clinical_df.loc[:, label].values)
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)
        self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(encoded_clinical_df.loc[:, label].values.reshape(-1,1)) # NOTE modified code.


        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_train.values()] + [clinical_df])
        self.baseline = self._compute_baseline(clinical_df, lt_samples_train, event, surv_time)
        dataset_train = MultiOmicsDataset(omics_df=omics_train, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
                                            label=label, event=event, surv_time=surv_time)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, **kwargs)
        if omics_val:
            lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])
            dataset_val = MultiOmicsDataset(omics_df=omics_val, clinical_df=encoded_clinical_df, lt_samples=lt_samples_val,
                                            label=label, event=event, surv_time=surv_time)
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

        self.history = []

        ### class weights
        n_samples = encoded_clinical_df.loc[lt_samples_train, label].values
        print(np.unique(n_samples, return_counts=True))
        # we follow the same formula as in the original implementation but extends it to multi-class
        class_labels, counts = np.unique(n_samples, return_counts=True)
        C = len(class_labels)
        class_weights = len(n_samples) / (C * counts)
        class_weights = class_weights.astype(np.float32)
        print("Class weights:")
        print(class_weights)
        class_weights = torch.tensor(class_weights).to(self.device)

        # ----------------- early stopping -----------------
        import copy
        best_loss = np.inf
        patience = 100 # NOTE
        best_model = None
        # -------------------------------------------------

        #### timing
        import time
        total_train_time = 0.0

        torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(n_epochs):

            st_time = time.perf_counter()

            overall_loss = 0
            self._switch_phase(epoch)
            for batch_idx, (x,labels,os_time,os_event) in enumerate(train_loader):
                self.train_all()
                loss_train = self._train_loop(x, labels, os_time,os_event, class_weights) # NOTE added class_weights
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx+1)*batch_size)

            #### timing
            total_train_time += time.perf_counter() - st_time

            overall_loss = 0
            if omics_val != None:
                for batch_idx, (x,labels, os_time,os_event) in enumerate(val_loader):
                    self.eval_all()
                    loss_val = self._train_loop(x, labels, os_time,os_event, class_weights) # NOTE added class_weights
                    overall_loss += loss_val.item()
                average_loss_val = overall_loss / ((batch_idx+1)*batch_size)
                self.history.append((average_loss_train, average_loss_val))
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train, "\tAverage Loss Val : ", average_loss_val)
                # ----------------- early stopping -----------------
                if average_loss_val < best_loss:
                    best_loss = average_loss_val
                    best_model = copy.deepcopy(self.state_dict())
                    no_imprv_counter = 0
                else:
                    no_imprv_counter += 1
                    if no_imprv_counter >= patience:
                        print("Early stopping at epoch", epoch + 1)
                        self.load_state_dict(best_model)
                        break
                # --------------------------------------------------
            else:
                self.history.append(average_loss_train)
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train)

        self.load_state_dict(best_model)
        #### timing
        print(f"CustOMICS Training time for {epoch+1} epochs: {total_train_time:.2f} seconds")

        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")

    def get_latent_representation(self, omics_df, tensor=False):
        self.eval_all()
        if tensor == False:
            x = [torch.Tensor(omics_df[source].values) for source in omics_df.keys()]
        else:
            x = [omics for omics in omics_df]
        with torch.no_grad():
            for i in range(len(x)):
                x[i] = x[i].to(self.device)
            z, loss = self._compute_loss(x)
        return z.cpu().detach().numpy()

    def reconstruct(self, x):
        x = torch.Tensor(x)
        z = self.autoencoders[0](x)[1]
        return self.autoencoders[0].decode(z).cpu().detach().numpy()


    def plot_representation(self, omics_df, clinical_df, labels, filename, title, show=True):
        labels_df = clinical_df.loc[:, labels]
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df=omics_df)
        save_plot_score(filename, z, labels_df[lt_samples].values, title, show=True)


    # def source_predict(self, expr_df, source):
    #     #tensor_expr = torch.Tensor(expr_df.values)
    #     tensor_expr = expr_df
    #     if source == 'CNV' or source == 'protein':
    #         z = self.lt_encoders[0](tensor_expr)
    #     elif source == 'RNAseq' or source == 'gene_exp':
    #         z = self.lt_encoders[1](tensor_expr)
    #     elif source == 'methyl':
    #         z = self.lt_encoders[2](tensor_expr)
    #     y_pred_proba = self.classifier(z)
    #     return y_pred_proba

    def source_predict(self, expr_df, source):
        tensor_expr = expr_df  # assuming expr_df is already a torch.Tensor
        if source not in self.source_names:
            raise ValueError(f"Source '{source}' is not recognized. Expected one of: {self.source_names}")
        idx = self.source_names.index(source)
        z = self.lt_encoders[idx](tensor_expr)
        y_pred_proba = self.classifier(z)
        return y_pred_proba

    def predict_risk(self, omics_df):
        z = self.get_latent_representation(omics_df)
        return self.survival_predictor(torch.Tensor(z))

    def predict_survival(self, omics_df, t=None):
        lt_samples = get_common_samples([df for df in omics_df.values()])
        dt_surv = {}
        risk_score = self.predict_risk(omics_df).cpu().detach().numpy()
        for sample, risk in zip(lt_samples, risk_score):
            dt_surv[sample] = self.baseline*np.exp(risk[0])
        return dt_surv

    def evaluate_latent(self, omics_test, clinical_df, label, event, surv_time, task, batch_size=32, plot_roc=False):
        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
                                            label=label, event=event, surv_time=surv_time)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (x, labels, os_time, os_event) in enumerate(test_loader):
                z, loss  = self._compute_loss(x)
                if task == 'survival':
                    predicted_survival_hazard = self.survival_predictor(z)
                    predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    c_index.append(CIndex_lifeline(predicted_survival_hazard, os_event, os_time))
                    return np.mean(c_index)
                elif task == 'classification':
                    svc_model = SVC()
                    z = self.get_latent_representation(x, tensor=True)
                    svc_model.fit(z.cpu().detach().numpy(), labels.cpu().detach().numpy())
                    y_pred_proba = svc_model.predict_proba()
                    y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    y_pred_proba = y_pred_proba.cpu().detach().numpy()
                    y_true = labels.cpu().detach().numpy()
                    classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder))
                    if plot_roc:
                        plot_roc_multiclass(y_test=y_true, y_pred_proba=y_pred_proba, filename='test', n_classes=self.num_classes,
                                            var_names=np.unique(clinical_df.loc[:, label].values.tolist()))

                
                    return classif_metrics


    # def evaluate(self, omics_test, clinical_df, label, event, surv_time, task, batch_size=32, plot_roc=False):

    #     encoded_clinical_df = clinical_df.copy()
    #     encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

    #     kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

    #     lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
    #     dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
    #                                         label=label, event=event, surv_time=surv_time)
    #     test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

    #     self.eval_all()
    #     classif_metrics = []
    #     c_index = []
    #     with torch.no_grad():
    #         for batch_idx, (x, labels, os_time, os_event) in enumerate(test_loader):
                
    #             # NOTE move to device
    #             for i in range(len(x)):
    #                 x[i] = x[i].to(self.device)
    #             labels = labels.to(self.device)

    #             z, loss  = self._compute_loss(x)
    #             if task == 'survival':
    #                 predicted_survival_hazard = self.survival_predictor(z)
    #                 predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
    #                 os_time = os_time.cpu().detach().numpy()
    #                 os_event = os_event.cpu().detach().numpy()
    #                 c_index.append(CIndex_lifeline(predicted_survival_hazard, os_event, os_time))
    #                 return np.mean(c_index)
    #             elif task == 'classification':
    #                 y_pred_proba = self.classifier(z)
    #                 y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
    #                 y_pred_proba = y_pred_proba.cpu().detach().numpy()
    #                 y_true = labels.cpu().detach().numpy()
    #                 classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder))
    #                 if plot_roc:
    #                     plot_roc_multiclass(y_test=y_true, y_pred_proba=y_pred_proba, filename='test', n_classes=self.num_classes,
    #                                         var_names=np.unique(clinical_df.loc[:, label].values.tolist()))


    # NOTE modified
    def evaluate(self, omics_test, clinical_df, label, event, surv_time, task, batch_size=32, plot_roc=False):

        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_test = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_test,
                                            label=label, event=event, surv_time=surv_time)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            
            # NOTE
            y_preds = np.array([])
            y_trues = np.array([])
            y_probs = np.array([])
            os_times = np.array([])
            os_events = np.array([])
            predicted_survival_hazards = np.array([])

            for batch_idx, (x, labels, os_time, os_event) in enumerate(test_loader):
                
                # NOTE move to device
                for i in range(len(x)):
                    x[i] = x[i].to(self.device)
                labels = labels.to(self.device)

                z, loss  = self._compute_loss(x)
                if task == 'survival':
                    predicted_survival_hazard = self.survival_predictor(z)
                    predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    os_times = np.append(os_times, os_time) # NOTE
                    os_events = np.append(os_events, os_event) # NOTE
                    predicted_survival_hazards = np.append(predicted_survival_hazards, predicted_survival_hazard) # NOTE
                elif task == 'classification':
                    y_pred_proba = self.classifier(z)
                    y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    y_true = labels.cpu().detach().numpy()
                    y_preds = np.append(y_preds, y_pred)
                    y_trues = np.append(y_trues, y_true)
                    y_pred_proba_batch = y_pred_proba.cpu().detach().numpy()
                    if y_probs.size == 0:
                        y_probs = y_pred_proba_batch
                    else:
                        y_probs = np.vstack((y_probs, y_pred_proba_batch))
            print("====================================================================")
            if task == 'classification':
                ################################ perf ################################
                y_true = y_trues
                y_pred = y_preds
                task_is_biclassif = len(np.unique(y_true)) == 2 # NOTE
                roc_auc = None
                aucpr = None
                recall = None
                precision = None
                f1 = None
                f1_weighted = None
                f1_macro = None
                acc = None
                mcc = None
                balanced_acc =  None
                if task_is_biclassif:
                    y_proba = y_probs[:, -1]
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
                    print(f"MCC:          {mcc:.4f}")
                    balanced_acc = balanced_accuracy_score(y_true, y_pred)
                    print(f"Balanced Acc: {balanced_acc:.4f}")
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
            elif task == 'survival':
                c_index = CIndex_lifeline(predicted_survival_hazards, os_events, os_times)
                perf = {
                    'c_index': c_index
                }
            return perf # NOTE

    def stratify(self, omics_df, clinical_df, event, surv_time, treshold=0.5, 
                    save_plot=False, plot_title="", filename=''):
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df)
        hazard_pred = self.survival_predictor(torch.Tensor(z)).cpu().detach().numpy()
        dt_strat = {'high': [], 'low': []}
        for i in range(len(lt_samples)):
            if hazard_pred[i] <= np.mean(hazard_pred):
                dt_strat['low'].append(lt_samples[i])
            else:
                dt_strat['high'].append(lt_samples[i])
        kmf_low = KaplanMeierFitter(label='low risk')
        kmf_high = KaplanMeierFitter(label='high risk')
        kmf_low.fit(clinical_df.loc[dt_strat['low'], surv_time], clinical_df.loc[dt_strat['low'], event])
        kmf_high.fit(clinical_df.loc[dt_strat['high'], surv_time], clinical_df.loc[dt_strat['high'], event])
        p_value = cox_log_rank(hazard_pred.reshape(1,-1)[0], np.array(clinical_df.loc[lt_samples, event].values, dtype=float), np.array(clinical_df.loc[lt_samples, surv_time].values, dtype=float))

        kmf_low.plot()
        kmf_high.plot()
        plt.title(plot_title + " (p-value = {:.3g})".format(p_value))
        plt.xlim((0,2500))
        if save_plot:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()


    # def explain(self, sample_id, omics_df, clinical_df, source, subtype,label='PAM50', device='cpu', show=False):
    #     """
    #     :param sample_id: List of samplesid to consider for explanation.
    #     :param vae_model: The CustOmics model to explain, the output should be a 1xn_class tensor.
    #     :param expr_df: DataFrame of data to explain, the input has to match the source selected for the explanation.
    #     :param clinical_df: DataFrame containing the clinical data.
    #     :param source: Omic source to explain.
    #     :param subtype: Subtype that needs explanation, shap values will be computed for this subtypes against the others.
    #     :param le: LabelEncoder to revert back from numerical encoding to original names.
    #     :return:None, plots of the top 10 genes and their shap values
    #     """
    #     #This class has combined the different analysis' of the Deep SHAP values we conducted.
    #     #SHAP reference: Lundberg et al., 2017: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

    #     expr_df = omics_df[source]
    #     sample_id = list(set(sample_id).intersection(set(expr_df.index)))
    #     phenotype = processPhenotypeDataForSamples(clinical_df, sample_id, self.label_encoder)

    #     conditionaltumour=phenotype.loc[:, label] == subtype

        
    #     expr_df = expr_df.loc[sample_id,:]
    #     normal_expr = randomTrainingSample(expr_df, 10) # NOTE sample 10 data points as bg
    #     tumour_expr = splitExprandSample(condition=conditionaltumour, sampleSize=10, expr=expr_df)
    #     # put on device as correct datatype
    #     background = addToTensor(expr_selection=normal_expr, device=device)
    #     male_expr_tensor = addToTensor(expr_selection=tumour_expr, device=device)

    #     e = shap.DeepExplainer(ModelWrapper(self, source=source), background)
    #     shap_values_female = e.shap_values(male_expr_tensor, ranked_outputs=None)

    #     shap.summary_plot(shap_values_female[0],features=tumour_expr,feature_names=list(tumour_expr.columns), show=False, plot_type="violin", max_display=10, plot_size=[4,6])
    #     plt.savefig('shap_{}_{}.png'.format(source, subtype), bbox_inches='tight')
    #     if show:
    #         plt.show()
    #     plt.clf()

    # NOTE adjusted to fit the current code
    def explain(
        self,
        model,
        sample_id,
        omics_df,
        clinical_df,
        source,
        subtype,
        label='PAM50',
        device='cpu',
        show=False,
        background_ids=None, # NOTE added
        task=None): # NOTE added task
        """
        :param sample_id: List of samplesid to consider for explanation.
        :param vae_model: The CustOmics model to explain, the output should be a 1xn_class tensor.
        :param expr_df: DataFrame of data to explain, the input has to match the source selected for the explanation.
        :param clinical_df: DataFrame containing the clinical data.
        :param source: Omic source to explain.
        :param subtype: Subtype that needs explanation, shap values will be computed for this subtypes against the others.
        :param le: LabelEncoder to revert back from numerical encoding to original names.
        :return:None, plots of the top 10 genes and their shap values
        """
        #This class has combined the different analysis' of the Deep SHAP values we conducted.
        #SHAP reference: Lundberg et al., 2017: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

        expr_df = omics_df[source]
        sample_id = list(set(sample_id).intersection(set(expr_df.index)))
        phenotype = processPhenotypeDataForSamples(clinical_df, sample_id, model.label_encoder) # NOTE this is just clinical_df.loc[sample_id]

        if task == 'classification':
            conditionaltumour=phenotype.loc[:, label] == subtype # mask for the class. NOTE this does not align with what we did for other approaches, but we keep this as it is since it's customics original code.
        elif task == 'survival':
            conditionaltumour = pd.Series([True]*phenotype.shape[0], index=phenotype.index)
        
        # NOTE to enable using training samples as background
        background = torch.tensor(expr_df.loc[background_ids, :].values, dtype=torch.float32).to(device)

        expr_df = expr_df.loc[sample_id,:] # take test

        tumour_expr = splitExprandSample(condition=conditionaltumour, sampleSize=sum(conditionaltumour), expr=expr_df) # NOTE sampleSize changed from 10 to all, as 10 was for demonstraing example in the original code.
        male_expr_tensor = addToTensor(expr_selection=tumour_expr, device=device)
        e = shap.DeepExplainer(ModelWrapper(model, source=source), background)
        X = male_expr_tensor
        shap_values = chunked_run(
            func = e.shap_values,
            input = X,
            output_size = (X.shape[0], X.shape[1], model.label_encoder.classes_.shape[0]),
            batch_size = min(1024, X.shape[0]),
            ranked_outputs=None,
            check_additivity=False # NOTE
        ).cpu().numpy()

        if task == 'classification':
            mod_class_ft_score = shap_values[:, :, model.label_encoder.transform([subtype])[0]].mean(axis=0) # (N, L, C) --> (N_c, L) --> (L) 
            return mod_class_ft_score
        elif task == 'survival':
            return shap_values

    def plot_loss(self):
        n_epochs = len(self.history)
        plt.title('Evolution of the loss function with respect to the epochs')
        plt.vlines(x=self.switch_epoch, ymin=0, ymax=2.5, colors='purple', ls='--', lw=2, label='phase 2 switch')
        plt.plot(range(0, n_epochs), [loss[0] for loss in self.history], label = 'train loss')
        plt.plot(range(0, n_epochs), [loss[1] for loss in self.history], label = 'val loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def print_parameters(self):
        lt_params = []
        lt_names = []
        for autoencoder in self.autoencoders:
            for name, param in autoencoder.named_parameters():
                lt_params.append(param.data)
                lt_names.append(name)
        for name, param in self.central_layer.named_parameters():
            lt_params.append(param.data)
            lt_names.append(name)
        print(len(lt_params))

    def get_number_parameters(self):
        sum_params = 0
        for autoencoder in self.autoencoders:
            sum_params += sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
        sum_params += sum(p.numel() for p in self.central_layer.parameters() if p.requires_grad)
        return sum_params


    def train_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.train()
            decoder.train()
        for autoencoder in self.autoencoders:
            autoencoder.train()
        self.central_layer.train()
        if self.survival_predictor:
            self.survival_predictor.train()
        if self.classifier:
            self.classifier.train()
    def eval_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.eval()
            decoder.eval()
        for autoencoder in self.autoencoders:
            autoencoder.eval()
        self.central_layer.eval()
        if self.survival_predictor:
            self.survival_predictor.eval()
        if self.classifier:
            self.classifier.eval()