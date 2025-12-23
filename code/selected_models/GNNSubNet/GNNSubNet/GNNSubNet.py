# GNNSubNet.py
# Authors: Bastian Pfeifer <https://github.com/pievos101>, Marcus D. Bloice <https://github.com/mdbloice>
from urllib.parse import _NetlocResultMixinStr
import numpy as np
import random
#from scipy.sparse.extract import find
from scipy.sparse import find
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn.modules import conv
from torch_geometric import data
from torch_geometric.data import DataLoader, Batch
from pathlib import Path
import copy
from tqdm import tqdm
import os
import requests
import pandas as pd
import io
#from collections.abc import Mapping
import time

# NOTE added
from collections import defaultdict
import random

from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from .gnn_training_utils import check_if_graph_is_connected, pass_data_iteratively
from .dataset import convert_to_s2vgraph
from .gnn_explainer import GNNExplainer
from .graphcnn  import GraphCNN
from .graphcheb import GraphCheb, ChebConvNet, test_model_acc, test_model

from .community_detection import find_communities
from .edge_importance import calc_edge_importance

from torch_geometric.nn.conv.cheb_conv import ChebConv

class GNNSubNet(object):
    """
    The class GNNSubSet represents the main user API for the
    GNN-SubNet package.
    """
    def __init__(
        self,
        dataset,
        gene_names,
        # location=None, ppi=None, features=None, target=None,
        # cutoff=950, normalize=True,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        device='cpu') -> None: # NOTE added device

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        # self.location = location
        # self.ppi = ppi
        # self.features = features
        # self.target = target
        # self.dataset = None
        self.model_status = None
        self.model = None
        # self.gene_names = None
        self.accuracy = None
        self.confusion_matrix = None
        self.test_loss = None
        self.device = device

        # Flags for internal use (hidden from user)
        self._explainer_run = False

        # if ppi == None:
        #     return None

        # NOTE commented out
        # assert ppi is not None
        # dataset, gene_names = load_OMICS_dataset(self.ppi, self.features, self.target, True, cutoff, normalize)
        #  # Check whether graph is connected
        # check = check_if_graph_is_connected(dataset[0].edge_index)
        # print("Graph is connected ", check)
        # if check == False:
        #     print("Calculate subgraph ...")
        #     dataset, gene_names = load_OMICS_dataset(self.ppi, self.features, self.target, False, cutoff, normalize)
        # check = check_if_graph_is_connected(dataset[0].edge_index)
        # print("Graph is connected ", check)
        #print('\n')
        # print('##################')
        # print("# DATASET LOADED #")
        # print('##################')
        #print('\n')

        self.dataset = dataset
        self.true_class = None
        self.gene_names = gene_names
        self.s2v_test_dataset = None
        self.edges =  np.transpose(np.array(dataset[0].edge_index))

        self.edge_mask = None
        self.node_mask = None
        self.node_mask_matrix = None
        self.modules = None
        self.module_importances = None

    def summary(self):
        """
        Print a summary for the GNNSubSet object's current state.
        """
        print("")
        print("Number of nodes:", len(self.dataset[0].x))
        print("Number of edges:", self.edges.shape[0])
        print("Number of modalities:",self.dataset[0].x.shape[1])

    def train(self, epoch_nr = 1000, 
        learning_rate=0.01): # NOTE TODO original default was 0.01
        print("graphcnn for training ...")
        perf = self.train_graphcnn(epoch_nr = epoch_nr, learning_rate=learning_rate)
        self.classifier="graphcnn"
        return perf

    def explain(self, n_runs=1, classifier="graphcnn", communities=True):
        if self.classifier=="graphcnn":
            self.explain_graphcnn(n_runs=n_runs, communities=communities)      

    def predict(self, gnnsubnet_test, classifier="graphcnn"):
        if self.classifier=="graphcnn":
            pred = self.predict_graphcnn(gnnsubnet_test=gnnsubnet_test)      
        pred = np.array(pred)
        pred = pred.reshape(1, pred.size)
        return pred

    # NOTE 
    #model = GraphCNN(5, 2, input_dim, 32, n_classes, 0.5, True, 'sum1', 'sum', 0)
    def train_graphcnn(self, num_layers=2, num_mlp_layers=2, 
                        epoch_nr = 100, graph_pooling_type='sum1', neighbor_pooling_type ='sum',
                        learning_rate=0.01):
        """
        Train the GNN model on the data provided during initialisation.
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
        neighbor_pooling_type: *sum*! how to aggregate neighbors (mean, average, or max)
        """
        use_weights = False

        dataset = self.dataset

        ########################################################################################################################
        # Downsampling of the class that contains more elements ===========================================================
        # ########################################################################################################################

        # NOTE commented out
        # graphs_class_0_list = []
        # graphs_class_1_list = []
        # for graph in dataset:
        #     if graph.y.numpy() == 0:
        #         graphs_class_0_list.append(graph)
        #     else:
        #         graphs_class_1_list.append(graph)

        # if graphs_class_0_len >= graphs_class_1_len:
        #     random_graphs_class_0_list = random.sample(graphs_class_0_list, graphs_class_1_len)
        #     balanced_dataset_list = graphs_class_1_list + random_graphs_class_0_list

        # if graphs_class_0_len < graphs_class_1_len:
        #     random_graphs_class_1_list = random.sample(graphs_class_1_list, graphs_class_0_len)
        #     balanced_dataset_list = graphs_class_0_list + random_graphs_class_1_list

        # random.shuffle(balanced_dataset_list)

        # list_len = len(balanced_dataset_list)
        # #print(list_len)
        # train_set_len = int(list_len * 4 / 5)
        # train_dataset_list = balanced_dataset_list[:train_set_len]
        # test_dataset_list  = balanced_dataset_list[train_set_len:]

        # train_graph_class_0_nr = 0
        # train_graph_class_1_nr = 0
        # for graph in train_dataset_list:
        #     if graph.y.numpy() == 0:
        #         train_graph_class_0_nr += 1
        #     else:
        #         train_graph_class_1_nr += 1
        # print(f"Train graph class 0: {train_graph_class_0_nr}, train graph class 1: {train_graph_class_1_nr}")

        # test_graph_class_0_nr = 0
        # test_graph_class_1_nr = 0
        # for graph in test_dataset_list:
        #     if graph.y.numpy() == 0:
        #         test_graph_class_0_nr += 1
        #     else:
        #         test_graph_class_1_nr += 1
        # print(f"Validation graph class 0: {test_graph_class_0_nr}, validation graph class 1: {test_graph_class_1_nr}")

        # s2v_train_dataset = convert_to_s2vgraph(train_dataset_list)
        # s2v_test_dataset  = convert_to_s2vgraph(test_dataset_list)

        # NOTE for both multi and bi class
        # print("graphs by class....")
        graphs_by_class = {}
        for graph in dataset:
            label = int(graph.y.numpy())  # supports any number of classes
            if label not in graphs_by_class:
                graphs_by_class[label] = []
            graphs_by_class[label].append(graph)

        # print('train, val, test ...')
        train_dataset_list = [dataset[i] for i in self.train_idx]
        val_dataset_list   = [dataset[i] for i in self.val_idx]
        test_dataset_list  = [dataset[i] for i in self.test_idx]

        # def count_graphs(dataset_list):
        #     counts = {}
        #     for graph in dataset_list:
        #         label = int(graph.y.numpy())
        #         counts[label] = counts.get(label, 0) + 1
        #     return counts

        # train_counts = count_graphs(train_dataset_list)
        # val_counts   = count_graphs(val_dataset_list)
        # test_counts  = count_graphs(test_dataset_list)

        # print("Train set counts per class:", train_counts)
        # print("Validation set counts per class:", val_counts)
        # print("Test set counts per class:", test_counts)

        print("converting s2vgraph for train set..")
        s2v_train_dataset = convert_to_s2vgraph(train_dataset_list, self.device)
        print("converting s2vgraph for val set..")
        s2v_val_dataset   = convert_to_s2vgraph(val_dataset_list, self.device)
        print("converting s2vgraph for test set..")
        s2v_test_dataset  = convert_to_s2vgraph(test_dataset_list, self.device)

        # TRAIN GNN -------------------------------------------------- #
        ### class weights
        trn_labels = np.array([int(graph.y.numpy()) for graph in train_dataset_list])
        # we follow the same formula as in the original implementation but extends it to multi-class
        class_labels, counts = np.unique(trn_labels, return_counts=True)
        print("Class labels:", class_labels)
        print("Class counts:", counts)
        C = len(class_labels)
        class_weights = len(trn_labels) / (C * counts)
        class_weights = class_weights.astype(np.float32)
        print("Class weights:")
        print(class_weights)
        class_weights = torch.tensor(class_weights).to(self.device)

        #print(count/len(dataset), 1-count/len(dataset))

        # model_path = 'omics_model.pth'
        no_of_features = dataset[0].x.shape[1]
        nodes_per_graph_nr = dataset[0].x.shape[0]

        #print(len(dataset), len(dataset)*0.2)
        #s2v_dataset = convert_to_s2vgraph(dataset)
        #train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=123)
        #s2v_train_dataset = convert_to_s2vgraph(train_dataset)
        #s2v_test_dataset = convert_to_s2vgraph(test_dataset)
        #s2v_train_dataset, s2v_test_dataset = train_test_split(s2v_dataset, test_size=0.2, random_state=123)

        input_dim = no_of_features
        hidden_dim = 32 # NOTE 32 is original default
        n_classes = len(np.unique(trn_labels)) # NOTE

        model = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, n_classes, 0.5, True, graph_pooling_type, neighbor_pooling_type, device=self.device)

        # NOTE
        model = model.to(self.device)

        opt = torch.optim.Adam(model.parameters(), lr = learning_rate)

        # load_model = False
        # if load_model:
        #     checkpoint = torch.load(model_path)
        #     model.load_state_dict(checkpoint['state_dict'])
        #     opt = checkpoint['optimizer']

        model.train()
        min_loss = 50
        best_model = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, n_classes, 0.5, True, graph_pooling_type, neighbor_pooling_type, device=self.device)
        min_val_loss = 1000000
        n_epochs_stop = 100 # NOTE
        epochs_no_improve = 0
        steps_per_epoch = 35

        total_train_time = 0.0

        torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(epoch_nr):
            st_time = time.perf_counter()
            model.train()
            # pbar = tqdm(range(steps_per_epoch), unit='batch')
            epoch_loss = 0
            for pos in range(steps_per_epoch): # NOTE
                selected_idx = np.random.permutation(len(s2v_train_dataset))[:32]

                batch_graph = [s2v_train_dataset[idx] for idx in selected_idx]

                logits = model(batch_graph)
                
                # NOTE
                # labels = torch.LongTensor([graph.label for graph in batch_graph])
                labels = torch.tensor([graph.label for graph in batch_graph], dtype=torch.long, device=self.device)

                # NOTE
                loss = nn.CrossEntropyLoss(weight=class_weights)(logits,labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.detach().item()

            total_train_time += time.perf_counter() - st_time

            epoch_loss /= steps_per_epoch
            model.eval()
            output = pass_data_iteratively(model, s2v_train_dataset)
            predicted_class = output.max(1, keepdim=True)[1]
            labels = torch.tensor([graph.label for graph in s2v_train_dataset], device=self.device, dtype=torch.long) # NOTE added device
            correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
            acc_train = correct / float(len(s2v_train_dataset))
            print('Epoch {}, train loss {:.4f}'.format(epoch, epoch_loss))
            print(f"Train Acc {acc_train:.4f}")

            # pbar.set_description('epoch: %d' % (epoch))
            val_loss = 0
            output = pass_data_iteratively(model, s2v_val_dataset)

            pred = output.max(1, keepdim=True)[1]
            labels = torch.tensor([graph.label for graph in s2v_val_dataset], device=self.device, dtype=torch.long) # NOTE added device
            if use_weights:
                    loss = nn.CrossEntropyLoss(weight=class_weights)(output,labels)
            else:
                loss = nn.CrossEntropyLoss()(output,labels)
            val_loss += loss

            print('Epoch {}, val_loss {:.4f}'.format(epoch, val_loss))
            if val_loss < min_val_loss:
                # print(f"Saving best model with validation loss {val_loss}")
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    model.load_state_dict(best_model.state_dict())
                    break

        print(f"GNNSubNet model Training time {total_train_time} (s) for {epoch} epochs.")
        print(f"GNNSubNet model total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")

        confusion_array = []
        true_class_array = []
        predicted_class_array = []
        model.eval()
        correct = 0
        true_class_array = []
        predicted_class_array = []

        test_loss = 0

        model.load_state_dict(best_model.state_dict())

        output = pass_data_iteratively(model, s2v_test_dataset)
        predicted_class = output.max(1, keepdim=True)[1]
        labels = torch.tensor([graph.label for graph in s2v_test_dataset], device=self.device, dtype=torch.long) # NOTE added device
        correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
        acc_test = correct / float(len(s2v_test_dataset))

        loss = nn.CrossEntropyLoss(weight=class_weights)(output,labels)
        test_loss = loss

        # NOTE
        # predicted_class_array = np.append(predicted_class_array, predicted_class)
        # true_class_array = np.append(true_class_array, labels)
        predicted_class_array = np.append(predicted_class_array, predicted_class.cpu().numpy())
        true_class_array = np.append(true_class_array, labels.cpu().numpy())

        # NOTE
        # confusion_matrix_gnn = confusion_matrix(true_class_array, predicted_class_array)
        # print("\nConfusion matrix (Validation set):\n")
        # print(confusion_matrix_gnn)
        # counter = 0
        # for it, i in zip(predicted_class_array, range(len(predicted_class_array))):
        #     if it == true_class_array[i]:
        #         counter += 1
        # accuracy = counter/len(true_class_array) * 100
        # print("Validation accuracy: {}%".format(accuracy))
        # print("Validation loss {}".format(test_loss))

        # NOTE
        # checkpoint = {
        #     'state_dict': best_model.state_dict(),
        #     'optimizer': opt.state_dict()
        # }
        # torch.save(checkpoint, model_path)

        ################################ perf ################################
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
        y_prob = output
        y_true = torch.tensor(true_class_array)
        y_pred = torch.tensor(predicted_class_array)
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
            print(f"MCC: {mcc:.4f}")
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            print(f"Balanced accuracy: {balanced_acc:.4f}")
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


        model.train()

        self.model_status = 'Trained'
        self.model = copy.deepcopy(model)
        # self.accuracy = accuracy
        # self.confusion_matrix = confusion_matrix_gnn
        self.test_loss = test_loss
        self.s2v_test_dataset = s2v_test_dataset
        self.predictions = predicted_class_array
        self.true_class  = true_class_array

        return perf

    def explain_graphcnn(self, n_runs=10, explainer_lambda=0.8, communities=True, save_to_disk=False):
        """
        Explain the model's results.
        """

        ############################################
        # Run the Explainer
        ############################################

        model = self.model
        s2v_test_dataset = self.s2v_test_dataset
        dataset = self.dataset

        print("")
        print("------- Run the Explainer -------")
        print("")

        no_of_runs = n_runs
        lamda = 0.8 # not used!
        ems = []
        NODE_MASK = list()

        torch.tensor([1],device=self.device) # NOTE to ensure peak memory stats can be initialized
        torch.cuda.reset_peak_memory_stats(self.device)

        for idx in range(no_of_runs):
            print(f'Explainer::Iteration {idx+1} of {no_of_runs}')
            exp = GNNExplainer(model, epochs=300)

            # NOTE
            # em = exp.explain_graph_modified_s2v(s2v_test_dataset, lamda)
            # #Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
            # gnn_feature_masks = np.reshape(em, (len(em), -1))
            em = exp.explain_graph_modified_s2v(s2v_test_dataset, lamda).cpu()
            gnn_feature_masks = em.reshape(len(em), -1)

            NODE_MASK.append(np.array(gnn_feature_masks.sigmoid()))
            # np.savetxt(f'{LOC}/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            #np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            gnn_edge_masks = calc_edge_importance(gnn_feature_masks, dataset[0].edge_index)
            # np.savetxt(f'{LOC}/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            #np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            ems.append(gnn_edge_masks.sigmoid().cpu().numpy())

        ems     = np.array(ems)
        mean_em = ems.mean(0)

        # OUTPUT -- Save Edge Masks
        # np.savetxt(f'{LOC}/edge_masks.txt', mean_em, delimiter=',', fmt='%.5f')
        self.edge_mask = mean_em
        self.node_mask_matrix = np.concatenate(NODE_MASK,1)
        self.node_mask = np.concatenate(NODE_MASK,1).mean(1)

        self._explainer_run = True

        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")

        # NOTE commmented out.
        # ###############################################
        # # Perform Community Detection
        # ###############################################
        # print('Performing community detection...')
        # if communities:
        #     avg_mask, coms = find_communities(f'{LOC}/edge_index.txt', f'{LOC}/edge_masks.txt')
        #     self.modules = coms
        #     self.module_importances = avg_mask

        #     np.savetxt(f'{LOC}/communities_scores.txt', avg_mask, delimiter=',', fmt='%.3f')

        #     filePath = f'{LOC}/communities.txt'

        #     if os.path.exists(filePath):
        #         os.remove(filePath)

        #     f = open(f'{LOC}/communities.txt', "a")
        #     for idx in range(len(avg_mask)):
        #         s_com = ','.join(str(e) for e in coms[idx])
        #         f.write(s_com + '\n')

        #     f.close()

        #     # Write gene_names to file
        #     textfile = open(f'{LOC}/gene_names.txt', "w")
        #     for element in gene_names:
        #         listToStr = ''.join(map(str, element))
        #         textfile.write(listToStr + "\n")

        #     textfile.close()

        # self._explainer_run = True
    
    def predict_graphcnn(self, gnnsubnet_test):

        confusion_array = []
        true_class_array = []
        predicted_class_array = []

        s2v_test_dataset  = convert_to_s2vgraph(gnnsubnet_test.dataset)
        model = self.model
        model.eval()
        output = pass_data_iteratively(model, s2v_test_dataset)
        predicted_class = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
        correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
        acc_test = correct / float(len(s2v_test_dataset))

        #if use_weights:
        #    loss = nn.CrossEntropyLoss(weight=weight)(output,labels)
        #else:
        #    loss = nn.CrossEntropyLoss()(output,labels)
        #test_loss = loss
        ### class weights
        weight = torch.tensor([0.5, 0.5], device=self.device) # NOTE TODO


        predicted_class_array = np.append(predicted_class_array, predicted_class)
        true_class_array = np.append(true_class_array, labels)

        confusion_matrix_gnn = confusion_matrix(true_class_array, predicted_class_array)
        print("\nConfusion matrix:\n")
        print(confusion_matrix_gnn)

        counter = 0
        for it, i in zip(predicted_class_array, range(len(predicted_class_array))):
            if it == true_class_array[i]:
                counter += 1

        accuracy = counter/len(true_class_array) * 100
        print("Accuracy: {}%".format(accuracy))
        
        self.predictions_test = predicted_class_array
        self.true_class_test  = true_class_array
        self.accuracy_test = accuracy
        self.confusion_matrix_test = confusion_matrix_gnn
        
        return predicted_class_array
