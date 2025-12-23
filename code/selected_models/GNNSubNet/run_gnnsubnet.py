#%%
from .GNNSubNet import GNNSubNet as gnn
from .GNNSubNet.gnn_training_utils import check_if_graph_is_connected

import time
import pandas as pd
import networkx as nx
import numpy as np
import copy
import torch
from torch_geometric.data import Data
import sys
sys.path.append('/home/athan.li/eval_bk/code/')
from utils import factorize_label, mod_mol_dict, modmol_gene_set_tcga, convert_omics_to_gene_level, P2G, C2G, R2G, SPLITTER

def load_OMICS_dataset_for_evalbk(
        feats=None,
        label=None,
        connected=True,
        threshold=950,
        normalize=True):
    """
    Loads OMICS dataset with given edge, features, and survival paths. Returns formatted dataset for further usage
    :param edge_path: String with path to file with edges
    :param feat_paths: List of strings with paths to node features
    :param survival_path: String with path to file with graph classes
    return 
    :graphs: formatted dataset
    :row_pairs: mapping between integers and proteins
    :col_pairs: mapping between integers and proteins
    """

    ############ ppi
    ppi = pd.read_csv("/home/athan.li/eval_bk/code/selected_models/GNNSubNet/topology_filtered0.0.csv", delimiter=",").iloc[:, 1:]
    ppi = ppi[ppi.combined_score >= (threshold*1.0/1000)]
    protein1 = list(set(ppi[ppi.columns.values[0]]))
    protein2 = list(set(ppi[ppi.columns.values[1]]))
    protein1.extend(protein2)
    proteins = list(set(protein1))
    ppi_gset = proteins

    ##### Read in the feature matrices
    data = pd.concat(feats, axis=0)
    omics_gset = modmol_gene_set_tcga(data.columns, op='union') # TODO NOTE
    mdic = mod_mol_dict(data.columns)
    data = [data.loc[:, mdic['mods']==mdic['mods_uni'][k]] for k in range(len(mdic['mods_uni']))]
    data = {mdic['mods_uni'][k]: data[k] for k in range(len(mdic['mods_uni']))}

    ###########################################################################
    ### convert mols to gene-level
    ###########################################################################
    cover_gset = np.intersect1d(omics_gset, ppi_gset) # auto unique and sorted
    assert 'mRNA' in mdic['mods_uni'], "mRNA must be present"
    for k in mdic['mods_uni']:
        data[k].columns = data[k].columns.str.split(SPLITTER).str[1]
    
    data = convert_omics_to_gene_level(data, cover_gset)
    ###########################################################################

    # NOTE commented out
    # #print(3)
    # #print(proteins)
    # # find feature columns with NA values
    # nans = []
    # for feat in feats:
    #     nans.extend(feat.columns[feat.isna().any()].tolist())
    # nans = list(set(nans))
    # # nans contains the genes with NA entries

    #print(4)
    #print(nans)

    # NOTE commented out
    # for i in range(len(feats)):
    #     # get feature columns which are withon the PPI
    #     feats[i] = feats[i][feats[i].columns.intersection(proteins)]
        # # exclude the NA columns
        # feats[i] = feats[i][feats[i].columns.difference(nans)]

    # feats is a harmonized feature matrix
    #print(5)
    #print(feats)
    # Now harmonize the PPI network
         
    # NOTE commented out
    # proteins = list(set(proteins) & set(feats[0].columns.values))

    #print(6)
    #print(proteins)
    # proteins are the proteins which are in feat and ppi

    feats = [data[mdic['mods_uni'][k]] for k in range(len(mdic['mods_uni']))]

    # old_cols are gene names
    old_cols = feats[0].columns.values
    # old rows are patient names    
    old_rows = feats[0].index.values
    # new_cols are ids 0:n.genes
    new_cols = pd.factorize(old_cols)[0]
    # new_rows are ids 0:n.patients
    new_rows = pd.factorize(old_rows)[0]

    # Mapping between genes and ids 
    col_pairs = {name: no for name,no in zip(old_cols, new_cols)}
    # Mapping between patient names and ids
    row_pairs = {name: no for name,no in zip(old_rows, new_rows)}

    # Harmonize/Reduce PPI with feature matrix
    ppi = ppi[ppi[ppi.columns.values[0]].isin(old_cols)]
    ppi = ppi[ppi[ppi.columns.values[1]].isin(old_cols)]

    #print(ppi)
    # convert genes to node ids
    ppi[ppi.columns.values[0]] = ppi[ppi.columns.values[0]].map(col_pairs)
    ppi[ppi.columns.values[1]] = ppi[ppi.columns.values[1]].map(col_pairs)    

    # col_pairs --> node ids + gene names!

    graphs = []
    edge_index = ppi[[ppi.columns.values[0], ppi.columns.values[1]]].to_numpy()
    #print(edge_index)
    # convert to a proper format and sort
    edge_index = np.array(sorted(edge_index, key = lambda x: (x[0], x[1]))).T

    s = list(copy.copy(edge_index[0]))
    t = list(copy.copy(edge_index[1]))
    s.extend(t)

    nodes = list(col_pairs.values())
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = np.array(edge_index)

    # convert to proper format
    edges = [(row[0].item(), row[1].item()) for row in edges.T]
    graph.add_edges_from(edges)

    
    # Start of subgraph extraction (largest component) ------------------------------------------------- #
    if connected==False:
        col_pairs_for_iso = {no: name for name,no in zip(old_cols, new_cols)}

        #iso_nodes = [col_pairs_for_iso[x] for x in isolated_nodes]
        #for feat in feats:
        #    feat.drop(columns = iso_nodes, inplace=True)
        
        # Get largest component
        print('Number of subgraphs: ',nx.number_connected_components(graph))
        COMPONENTS = list(nx.connected_components(graph))
        L = []        
        for component in COMPONENTS:
            #print(component)
            L.append(len(component))

        max_id = L.index(max(L))    
        nodes  = COMPONENTS[max_id]
        print('Size of subgraph: ', len(nodes))
        
        ppi = ppi.drop(ppi[(~ppi.protein1.isin(nodes)) | (~ppi.protein2.isin(nodes))].index)
        
        drop_nodes = [col_pairs_for_iso[x] for x in nodes]
        for feat in feats:
            # feat.drop(feat.columns.difference(drop_nodes), 1, inplace=True) # NOTE
            feat.drop(feat.columns.difference(drop_nodes), axis=1, inplace=True)

        new_nodes = list(range(len(nodes)))
        new_nodes_dict = {old: new for old,new in zip(nodes, new_nodes)}
        ppi[ppi.columns.values[0]] = ppi[ppi.columns.values[0]].map(new_nodes_dict)
        ppi[ppi.columns.values[1]] = ppi[ppi.columns.values[1]].map(new_nodes_dict)   
        for feat in feats:
            feat.rename(columns=new_nodes_dict, inplace=True)

        edge_index = ppi[[ppi.columns.values[0], ppi.columns.values[1]]].to_numpy()
        edge_index = np.array(sorted(edge_index, key = lambda x: (x[0], x[1]))).T

    # End of subgraph extraction ------------------------------------------------- #

    temp = np.stack(feats, axis=-1)
    
    # NOTE commented out
    # if normalize ==True:
    #     new_temp = []
    #     for item in temp:
    #         new_temp.append(minmax_scale(item))
    #     temp = np.array(new_temp)

    # NOTE
    # survival = pd.read_csv(survival_path, delimiter=' ')
    # survival_values = survival.to_numpy()
    label_all_flattened = pd.concat(label, axis=0).values.flatten()
    label_all_flattened_fact, _ = factorize_label(label_all_flattened)
    
    for idx in range(temp.shape[0]):
        graphs.append(Data(x=torch.tensor(temp[idx]).float(),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        y=torch.tensor(label_all_flattened_fact[idx], dtype=torch.long)))
        #graphs.append(Data(node_features=torch.tensor(temp[idx]).float(),
        #                edge_mat=torch.tensor(edge_index, dtype=torch.long),
        #                y=torch.tensor(survival_values[0][idx], dtype=torch.long)))
    
    gene_names = feats[0].columns.values

    return graphs, gene_names

def run_gnnsubnet(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device
):
    print('Loading dataset...')
    dataset, gene_names = load_OMICS_dataset_for_evalbk([data_trn,data_val,data_tst], [label_trn,label_val,label_tst], False)
    check = check_if_graph_is_connected(dataset[0].edge_index)
    print("Graph is connected ", check)
    print('##################')
    print("# DATASET LOADED #")
    print('##################')

    train_idx = np.arange(data_trn.shape[0])
    val_idx = np.arange(data_trn.shape[0], data_trn.shape[0] + data_val.shape[0])
    test_idx = np.arange(data_trn.shape[0] + data_val.shape[0], data_trn.shape[0] + data_val.shape[0] + data_tst.shape[0])

    g = gnn(dataset, gene_names, device=device,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx)

    perf = g.train(epoch_nr=1000, learning_rate=0.01)

    st_time = time.perf_counter()
    g.explain(10)
    print("GNNSubNet BK identification time:", time.perf_counter() - st_time)

    # g.edge_mask # g.edges order. 
     # corresponding to g.gene_names order
    ft_score = pd.DataFrame(
        g.node_mask,
        index=g.gene_names,
        columns=['score']
    )

    return ft_score, perf
