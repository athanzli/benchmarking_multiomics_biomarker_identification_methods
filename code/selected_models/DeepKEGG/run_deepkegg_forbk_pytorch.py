# %%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef

torch.manual_seed(1029)
np.random.seed(1029)

import pandas as pd
import numpy as np
from captum.attr import DeepLift

MODS = ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA', 'protein']
DATA_PATH = '/home/athan.li/eval_bk/data/'

C2G = pd.read_csv("/data/zhaohong/TCGA_data/data/processed/TCGA_cpg2gene_mapping.csv", index_col=0)
P2G = pd.read_csv("/data/zhaohong/TCGA_data/data/processed/TCGA_protein2gene_mapping.csv", index_col=0)

SPLITTER = '@'
def mod_mol_dict(mol_ids):
    r"""
    
    Args:
        mol_ids (np.array): an array of molecule ids. e.g., array(['DNAm@cg22832044', 'DNAm@cg19580810', 'DNAm@cg14217534',
            'mRNA@CDH1', 'mRNA@RAB25', 'mRNA@TUBA1B', 'protein@E.Cadherin',
            'protein@Rab.25', 'protein@Acetyl.a.Tubulin.Lys40'], dtype=object)
    Returns:
        dict: a dictionary of mods, mol names, and mods uni.
    """
    mods = np.array([mod_id.split(SPLITTER)[0] for mod_id in mol_ids])
    mods_uni = np.unique(mods) # auto sorting in ascending order
    mols = np.array([mod_id.split(SPLITTER)[1] for mod_id in mol_ids])
    return {'mods': mods, 'mols': mols, 'mods_uni': mods_uni}

def factorize_label(y):
    r"""
    Factorize the label.
    """
    y_uni = np.unique(y)
    value_to_index = {value: idx for idx, value in enumerate(y_uni)}
    y_fac = np.array([value_to_index[item] for item in y]).astype(np.int64)
    return y_fac, value_to_index

###############################################################################
###############################################################################
class BiologicalModule(nn.Module):
    def __init__(self, input_dim, units, mapp=None, nonzero_ind=None, 
                 kernel_initializer='glorot_uniform', W_regularizer=None,
                 activation='tanh', use_bias=True, bias_initializer='zeros', 
                 bias_regularizer=None, **kwargs):
        """
        Biological Module.
        """
        super(BiologicalModule, self).__init__()
        self.units = units
        self.activation = activation
        self.mapp = mapp
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias

        if activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        else:
            self.activation_fn = None

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        if self.mapp is not None:
            self.mapp = self.mapp.astype(np.float32)
        if self.nonzero_ind is None and self.mapp is not None:
            self.nonzero_ind = np.array(np.nonzero(self.mapp)).T

        if self.nonzero_ind is None:
            raise ValueError("Either 'mapp' or 'nonzero_ind' must be provided.")

        nonzero_count = self.nonzero_ind.shape[0]
        if self.kernel_initializer == 'glorot_uniform':
            self.kernel_vector = nn.Parameter(
                torch.empty(nonzero_count, dtype=torch.float32)
            )
            nn.init.xavier_uniform_(self.kernel_vector.unsqueeze(1))
        else:
            raise NotImplementedError("Only 'glorot_uniform' initializer is implemented.")

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.units, dtype=torch.float32))
            if self.bias_initializer == 'zeros':
                nn.init.zeros_(self.bias)
            else:
                raise NotImplementedError("Only 'zeros' bias initializer is implemented.")
        else:
            self.register_parameter('bias', None)

        self.register_buffer('nonzero_ind_tensor', torch.tensor(self.nonzero_ind, dtype=torch.long))

        self.W_regularizer = W_regularizer
        self.bias_regularizer = bias_regularizer

    def forward(self, inputs):
        """
        Forward pass through the BiologicalModule.
        """
        input_dim = inputs.shape[1]
        trans = torch.zeros((input_dim, self.units), device=inputs.device)
        trans[self.nonzero_ind_tensor[:,0], self.nonzero_ind_tensor[:,1]] = self.kernel_vector

        output = torch.matmul(inputs, trans)

        if self.use_bias:
            output = output + self.bias

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Self-Attention mechanism.
        """
        super(SelfAttention, self).__init__()
        self.output_dim = output_dim

        self.WQ = nn.Linear(input_dim, output_dim, bias=False)
        self.WK = nn.Linear(input_dim, output_dim, bias=False)
        self.WV = nn.Linear(input_dim, output_dim, bias=False)

        nn.init.uniform_(self.WQ.weight)
        nn.init.uniform_(self.WK.weight)
        nn.init.uniform_(self.WV.weight)

    def forward(self, x):
        """
        Forward pass through the SelfAttention layer.
        """
        assert x.shape[0] == 1, "DeepKEGG requires attention on sample-level"
        L = x.shape[1]
        # x shape: (batch_size, seq_length, input_dim)
        WQ = self.WQ(x)  # (batch_size, seq_length, output_dim)
        WK = self.WK(x)  # (batch_size, seq_length, output_dim)
        WV = self.WV(x)  # (batch_size, seq_length, output_dim)

        QK = torch.matmul(WQ, WK.transpose(-2, -1))  # (batch_size, seq_length, seq_length)
        QK = QK / (self.output_dim ** 0.5)
        QK = F.softmax(QK, dim=-1)

        assert (QK.shape[1]==L) and (QK.shape[2]==L)

        V = torch.matmul(QK, WV)  # (batch_size, seq_length, output_dim)

        return V

class MultiInputModel(nn.Module):
    def __init__(self, mod1_input_dim, mod2_input_dim, mod3_input_dim, 
                 gene_pathway_bp_dfs,
                 attention_dim=64, # parameter k. optimal is 64 as in the paper.
                 num_classes=None):
        """
        Multi-input model integrating mod1, mod2, and mod3 data with Biological modules and self-attention
        """
        super(MultiInputModel, self).__init__()
        self.gene_pathway_bp_dfs = gene_pathway_bp_dfs

        self.biomodule_mod1 = BiologicalModule(
            input_dim=mod1_input_dim,
            units=gene_pathway_bp_dfs[0].shape[0], # units is the num of pathways
            mapp=gene_pathway_bp_dfs[0].values.T,
            W_regularizer=0.001,
            activation='tanh',
            use_bias=True
        )
        
        self.biomodule_mod2 = BiologicalModule(
            input_dim=mod2_input_dim,
            units=gene_pathway_bp_dfs[1].shape[0],
            mapp=gene_pathway_bp_dfs[1].values.T,
            W_regularizer=0.001,
            activation='tanh',
            use_bias=True
        )
        
        self.biomodule_mod3 = BiologicalModule(
            input_dim=mod3_input_dim,
            units=gene_pathway_bp_dfs[2].shape[0],
            mapp=gene_pathway_bp_dfs[2].values.T,
            W_regularizer=0.001,
            activation='tanh',
            use_bias=True
        )

        self.attention1 = SelfAttention(
            input_dim=gene_pathway_bp_dfs[0].shape[0], 
            output_dim=attention_dim
        )
        self.attention2 = SelfAttention(
            input_dim=gene_pathway_bp_dfs[1].shape[0], 
            output_dim=attention_dim
        )
        self.attention3 = SelfAttention(
            input_dim=gene_pathway_bp_dfs[2].shape[0], 
            output_dim=attention_dim
        )

        self.fc1 = nn.Linear(attention_dim * 3, 32)
        self.fc2 = nn.Linear(32, num_classes)

        self.activation_fc1 = nn.Tanh()

    def forward(self, mod1, mod2, mod3):
        """
        Args:
            mod1: input of shape (batch_size, mod1_input_dim)
        Returns:
            output of shape (batch_size, num_classes)
        """
        h0_mod1 = self.biomodule_mod1(mod1)      # (batch_size, units)
        h0_mod2 = self.biomodule_mod2(mod2)   # (batch_size, units)
        h0_mod3 = self.biomodule_mod3(mod3) # (batch_size, units)

        h0_mod1 = h0_mod1.unsqueeze(0)   # (1, batch_size, units)
        h0_mod2 = h0_mod2.unsqueeze(0) # (1, batch_size, units)
        h0_mod3 = h0_mod3.unsqueeze(0) # (1, batch_size, units)

        atten1 = self.attention1(h0_mod1)   # (1, batch_size, attention_dim)
        atten2 = self.attention2(h0_mod2)  # (1, batch_size, attention_dim)
        atten3 = self.attention3(h0_mod3) # (1, batch_size, attention_dim)

        atten1 = atten1.squeeze(0) # (batch_size, attention_dim)
        atten2 = atten2.squeeze(0) # (batch_size, attention_dim)
        atten3 = atten3.squeeze(0) # (batch_size, attention_dim)

        feature_total = torch.cat([atten1, atten2, atten3], dim=-1)  # (batch_size, attention_dim*3)

        h4 = self.activation_fc1(self.fc1(feature_total))  # (batch_size, 32)
        h5 = self.fc2(h4)                                   # (batch_size, num_classes)

        return h5  # (batch_size, num_classes)

def get_metrics(y_true, y_pred, y_scores, num_classes):
    f1 = f1_score(y_true, y_pred, average='micro')
    return f1

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def get_weights(model, data_loader, device, data_pd_s, data_pd_name, n_class):
    """
    Args:
        data_pd_s (list of dataframes): original dfs.
        data_pd_name: corresponding to data_pd_s.
    """

    deeplift = DeepLift(model, multiply_by_inputs=True)

    model.eval()

    gene_pds_all_classes = []
    for target_label in range(n_class):
        feature_importances = {name : np.zeros(data_pd_s[i].shape[1]) for i, name in enumerate(data_pd_name)}
        for batch in data_loader:
            mod1, mod2, mod3, y_batch = batch
            mod1 = mod1.to(device).requires_grad_()
            mod2 = mod2.to(device).requires_grad_()
            mod3 = mod3.to(device).requires_grad_()
            # labels = model(mod1, mod2, mod3).argmax(1)
            inputs = (mod1, mod2, mod3)
            baselines = (
                torch.zeros_like(mod1),
                torch.zeros_like(mod2),
                torch.zeros_like(mod3)
            )
            attributions = deeplift.attribute(
                inputs,
                baselines=baselines,
                # target=labels,
                target=target_label,
                additional_forward_args=None
            )
            keys_tmp = list(feature_importances.keys())
            feature_importances[keys_tmp[0]] += attributions[0].detach().cpu().numpy().sum(axis=0)
            feature_importances[keys_tmp[1]] += attributions[1].detach().cpu().numpy().sum(axis=0) 
            feature_importances[keys_tmp[2]] += attributions[2].detach().cpu().numpy().sum(axis=0)
        gene_pds = []
        for i, name in enumerate(data_pd_name):
            gene_pd = pd.DataFrame({
                'genes': data_pd_s[i].columns,
                'values': feature_importances[name]
            })
            gene_pds.append(gene_pd)
            assert list(gene_pd['genes']) == list(data_pd_s[i].columns)
            assert len(gene_pd['values']) == data_pd_s[i].shape[1]

        gene_pds_all_classes.append(gene_pds)
    return gene_pds_all_classes

def create_model_instance(mod1_input_dim, mod2_input_dim, mod3_input_dim, gene_pathway_bp_dfs, attention_dim=64,num_classes=None):
    """
    """
    model = MultiInputModel(
        mod1_input_dim=mod1_input_dim,
        mod2_input_dim=mod2_input_dim,
        mod3_input_dim=mod3_input_dim,
        gene_pathway_bp_dfs=gene_pathway_bp_dfs,
        attention_dim=attention_dim,
        num_classes=num_classes
    )
    model.apply(initialize_weights)
    return model

#%%
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################

def run_deepkegg(
    data_trn,
    label_trn,
    data_val,
    label_val,
    data_tst,
    label_tst,
    device='cuda:7',
    epochs=1000, # originally 500
    batch_size=64,
    # n_splits=5
    learning_rate=1e-4,
):
    print(f'Using device: {device}')

    ###########################################################################
    # PW
    ###########################################################################
    ###########################################################################
    # using deepkegg's own pathway data
    # genes-pathways annotation
    # path = './KEGG_pathways/20230205_kegg_hsa.gmt'
    path = '/home/athan.li/eval_bk/code/selected_models/DeepKEGG/KEGG_pathways/20230205_kegg_hsa.gmt'
    files = open(path, encoding='utf-8')
    files = files.readlines()
    paways_genes_dict = {}
    for i in files: 
        paways_genes_dict[i.split('\t')[0].split('_')[0]] = i.replace('\n','').split('\t')[2:] 
    # mirna-pathways annotation
    path = './KEGG_pathways/kegg_anano.txt'
    path = '/home/athan.li/eval_bk/code/selected_models/DeepKEGG/KEGG_pathways/kegg_anano.txt'
    files = open(path,encoding='utf-8')
    files = files.readlines()
    paways_mirna_dict = {}
    for i in files:
        keys = i.split(',')[0].split('|')[1]
        values1 = i.split(',')[1:-1]
        values2 =  i.split(',')[-1].replace('\n','')
        values1.append(values2)
        values1 =list(set(values1)) 
        paways_mirna_dict[keys] = values1
    # intersection of pathways
    union_kegg = list(set(paways_genes_dict.keys()).intersection(set(paways_mirna_dict.keys())))
    print("Number of pathways:", len(union_kegg))
    paways_genes_dicts ={}
    paways_mirna_dicts ={}
    for i in union_kegg:
        paways_genes_dicts[i] = paways_genes_dict[i]
    for i in union_kegg:
        paways_mirna_dicts[i] = paways_mirna_dict[i]    
    # mols in pathway
    genes_existed_pathway = []
    mirna_existed_pathway = []
    for index in paways_genes_dicts.keys():
        genes_existed_pathway = genes_existed_pathway+ list(paways_genes_dicts[index])
    genes_existed_pathway = set(genes_existed_pathway)
    for index in paways_mirna_dicts.keys():
        mirna_existed_pathway = mirna_existed_pathway+ list(paways_mirna_dicts[index])
    mirna_existed_pathway = set(mirna_existed_pathway)
    #
    pathway_union = list(paways_genes_dicts.keys()) # this is the intersection of pathways presented in both gene and mirna
    ###########################################################################
    mod_mollists = []
    mod_molinpw_lists = []
    mdic = mod_mol_dict(data_trn.columns)
    for mod in mdic['mods_uni']:
        mod_mollists.append(mdic['mols'][mdic['mods']==mod])
    for i in range(mdic['mods_uni'].shape[0]):
        if mdic['mods_uni'][i] in ['SNV', 'mRNA', 'CNV']:
            mols = np.intersect1d(mod_mollists[i].astype(str), list(genes_existed_pathway))
        elif mdic['mods_uni'][i] in ['DNAm']:
            mols = np.intersect1d(C2G.loc[C2G['gene'].isin(genes_existed_pathway)].index, mdic['mols'][mdic['mods']=='DNAm'])
        elif mdic['mods_uni'][i] in ['protein']:
            mols = np.intersect1d(P2G.loc[P2G['gene'].isin(genes_existed_pathway)].index, mdic['mols'][mdic['mods']=='protein'])
        elif mdic['mods_uni'][i] in ['miRNA']:
            mols = np.intersect1d(list(mirna_existed_pathway), mod_mollists[i].astype(str))
        else:
            raise ValueError("Mod not supported.")
        mod_molinpw_lists.append(mols)
    new_modmols = [f"{mod}@{mol}" for i, mod in enumerate(mdic['mods_uni']) for mol in mod_molinpw_lists[i]]
    data_trn = data_trn.loc[:, data_trn.columns.isin(new_modmols)]
    data_val = data_val.loc[:, data_val.columns.isin(new_modmols)]
    data_tst = data_tst.loc[:, data_tst.columns.isin(new_modmols)]
    ###########################################################################
    print("Constructing pathway-gene presence indicator matrix...")
    mdic = mod_mol_dict(data_trn.columns)
    mod_mollists = []
    for mod in mdic['mods_uni']:
        mod_mollists.append(mdic['mols'][mdic['mods']==mod])
    gene_pathway_bp_dfs = []
    for i in range(mdic['mods_uni'].shape[0]):
        pathways_genes = np.zeros((len(pathway_union), len(mod_mollists[i]))) 
        for p  in pathway_union:
            gs = paways_genes_dicts[p]
            if mdic['mods_uni'][i] in ['SNV', 'mRNA', 'CNV']:
                m_inds = [list(mod_mollists[i]).index(m) for m in gs if m in mod_mollists[i]]
            elif mdic['mods_uni'][i] in ['DNAm']:
                cpgs = np.intersect1d(C2G.loc[C2G['gene'].isin(gs)].index, mdic['mols'][mdic['mods']=='DNAm'])
                m_inds = [list(mod_mollists[i]).index(m) for m in cpgs if m in mod_mollists[i]]
            elif mdic['mods_uni'][i] in ['protein']:
                proteins = np.intersect1d(P2G.loc[P2G['gene'].isin(gs)].index, mdic['mols'][mdic['mods']=='protein'])
                m_inds = [list(mod_mollists[i]).index(m) for m in proteins if m in mod_mollists[i]]
            elif mdic['mods_uni'][i] in ['miRNA']:
                mirna_set = paways_mirna_dicts[p]
                m_inds = [list(mod_mollists[i]).index(m) for m in mirna_set if m in mod_mollists[i]]
            else:
                raise ValueError("Mod not supported.")
            p_ind = pathway_union.index(p)
            pathways_genes[p_ind, m_inds] = 1
        gene_pathway_bp = pd.DataFrame(pathways_genes, index=pathway_union, columns=mod_mollists[i])
        gene_pathway_bp_dfs.append(gene_pathway_bp)
    print('Done indicator matrix construction.')
    ###########################################################################
    ###########################################################################

    ###################
    assert mdic['mods_uni'].shape[0] == 3, "DeepKEGG currently only implements 3 modalities."
    print("Number of features in each modality:", np.unique(mdic['mods'], return_counts=True))

    mod1_train_x = data_trn.loc[:, mdic['mods'] == mdic['mods_uni'][0]].values.astype(np.float32)
    mod2_train_x = data_trn.loc[:, mdic['mods'] == mdic['mods_uni'][1]].values.astype(np.float32)
    mod3_train_x = data_trn.loc[:, mdic['mods'] == mdic['mods_uni'][2]].values.astype(np.float32)
    mod1_val_x = data_val.loc[:, mdic['mods'] == mdic['mods_uni'][0]].values.astype(np.float32)
    mod2_val_x = data_val.loc[:, mdic['mods'] == mdic['mods_uni'][1]].values.astype(np.float32)
    mod3_val_x = data_val.loc[:, mdic['mods'] == mdic['mods_uni'][2]].values.astype(np.float32)
    mod1_test_x = data_tst.loc[:, mdic['mods'] == mdic['mods_uni'][0]].values.astype(np.float32)
    mod2_test_x = data_tst.loc[:, mdic['mods'] == mdic['mods_uni'][1]].values.astype(np.float32)
    mod3_test_x = data_tst.loc[:, mdic['mods'] == mdic['mods_uni'][2]].values.astype(np.float32)
    # print(mod1_train_x.shape, mod2_train_x.shape, mod3_train_x.shape)

    train_y, fact_dic = factorize_label(label_trn.values.flatten())
    val_y, _ = factorize_label(label_val.values.flatten())
    test_y, _ = factorize_label(label_tst.values.flatten())
    n_class = len(fact_dic)

    ### class weights
    n_samples = train_y
    print(np.unique(n_samples, return_counts=True))
    # we follow the same formula as in the original implementation but extends it to multi-class
    class_labels, counts = np.unique(n_samples, return_counts=True)
    C = len(class_labels)
    class_weights = len(n_samples) / (C * counts)
    class_weights = class_weights.astype(np.float32)
    print(class_weights)

    num_classes = np.unique(train_y).shape[0]

    model = create_model_instance(
        mod1_input_dim=mod1_train_x.shape[1],
        mod2_input_dim=mod2_train_x.shape[1],
        mod3_input_dim=mod3_train_x.shape[1],
        gene_pathway_bp_dfs=gene_pathway_bp_dfs,
        attention_dim=64, # default optimal
        num_classes=num_classes
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = LambdaLR(optimizer, lambda step: 1.0 / (1.0 + (1e-4) * step))


    # criterion = nn.BCELoss()
    if class_weights is not None:
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    else:
        loss_function = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.tensor(mod1_train_x),
        torch.tensor(mod2_train_x),
        torch.tensor(mod3_train_x),
        torch.tensor(train_y).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(
        torch.tensor(mod1_val_x),
        torch.tensor(mod2_val_x),
        torch.tensor(mod3_val_x),
        torch.tensor(val_y).long()
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(
        torch.tensor(mod1_test_x),
        torch.tensor(mod2_test_x),
        torch.tensor(mod3_test_x),
        torch.tensor(test_y).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train
    torch.cuda.reset_peak_memory_stats(device)

    import time
    training_time = 0.0
    # es
    best_val_loss = np.inf
    best_model = None
    early_stopping_counter = 0
    patience = 100 # NOTE TODO
    # 
    for epoch in range(epochs):
        st_time = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            mod1_batch, mod2_batch, mod3_batch, y_batch = batch
            mod1_batch = mod1_batch.to(device)
            mod2_batch = mod2_batch.to(device)
            mod3_batch = mod3_batch.to(device)
            labels = y_batch.long().to(device)
            optimizer.zero_grad()
            outputs = model(mod1_batch, mod2_batch, mod3_batch)

            ### calculate loss
            base_loss = loss_function(outputs, labels)
            l2_bio = 0.0
            for m in (model.biomodule_mod1,
                      model.biomodule_mod2,
                      model.biomodule_mod3):
                l2_bio += (m.kernel_vector ** 2).sum()
            l2_bio = 0.001 * l2_bio
            l2_attn = 0.0
            for attn in (model.attention1,
                         model.attention2,
                         model.attention3):
                l2_attn += (attn.WQ.weight ** 2).sum()
                l2_attn += (attn.WK.weight ** 2).sum()
                l2_attn += (attn.WV.weight ** 2).sum()
            l2_attn = 0.003 * l2_attn
            loss = base_loss + l2_bio + l2_attn
            ###

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * mod1_batch.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)

        training_time += time.perf_counter() - st_time

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            y_preds = []
            y_labels = []
            outputss = []
            for batch in val_loader:
                mod1_batch, mod2_batch, mod3_batch, y_batch = batch
                mod1_batch = mod1_batch.to(device)
                mod2_batch = mod2_batch.to(device)
                mod3_batch = mod3_batch.to(device)
                outputs = model(mod1_batch, mod2_batch, mod3_batch)
                outputs = outputs.cpu().numpy()
                outputss.extend(outputs)
                y_preds.extend((outputs.argmax(1)).astype(int))
                y_labels.extend(y_batch.cpu().numpy())

            ### calculate loss
            base_loss = loss_function(torch.tensor(outputss).to(device), torch.tensor(y_labels).long().to(device)).item()
            l2_bio = 0.0
            for m in (model.biomodule_mod1,
                      model.biomodule_mod2,
                      model.biomodule_mod3):
                l2_bio += (m.kernel_vector ** 2).sum()
            l2_bio = 0.001 * l2_bio.item()
            l2_attn = 0.0
            for attn in (model.attention1,
                         model.attention2,
                         model.attention3):
                l2_attn += (attn.WQ.weight ** 2).sum()
                l2_attn += (attn.WK.weight ** 2).sum()
                l2_attn += (attn.WV.weight ** 2).sum()
            l2_attn = 0.003 * l2_attn.item()
            loss_val = base_loss + l2_bio + l2_attn
            ###

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_model = deepcopy(model)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break

            accuracy = accuracy_score(y_labels, y_preds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train avg. Loss: {avg_loss:.4f}, Val Loss: {loss_val:.4f}, val_accuracy: {accuracy:.4f}')

    print(f"DeepKEGG Training time for {epoch} epochs: {training_time:.2f} seconds.")
    
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during training: {peak_mb:.1f} MB")

    model = deepcopy(best_model)
    model.eval()
    with torch.no_grad():
        y_preds = []
        y_probs = []
        y_trues = []
        for batch in test_loader:
            mod1_batch, mod2_batch, mod3_batch, y_batch = batch
            mod1_batch = mod1_batch.to(device)
            mod2_batch = mod2_batch.to(device)
            mod3_batch = mod3_batch.to(device)
            outputs = model(mod1_batch, mod2_batch, mod3_batch)
            probabilities = F.softmax(outputs, dim=1)
            y_probs.append(probabilities.cpu().numpy())
            y_preds.extend((probabilities.cpu().numpy().argmax(1)).astype(int))
            y_trues.extend(y_batch.numpy())
    y_probs = np.vstack(y_probs)

    ################################ perf ################################
    y_true = y_trues
    y_pred = y_preds
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
        y_proba = y_probs[:, 1]
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
        'balanced_acc': balanced_acc,
    }
    #######################################################################

    ############################################################################################################################################################################
    ############################################################################################################################################################################
    ############################################################################################################################################################################
    torch.cuda.reset_peak_memory_stats(device)
    
    st_time = time.perf_counter()
    
    mod1_data = data_trn.loc[:, mdic['mods'] == mdic['mods_uni'][0]].astype(np.float32)
    mod2_data = data_trn.loc[:, mdic['mods'] == mdic['mods_uni'][1]].astype(np.float32)
    mod3_data = data_trn.loc[:, mdic['mods'] == mdic['mods_uni'][2]].astype(np.float32)
    gene_pds_all_classes = get_weights(
        model=model,
        data_loader=train_loader,
        device=device,
        data_pd_s=[mod1_data, mod2_data, mod3_data], # need to ensure order
        data_pd_name=mdic['mods_uni'], # need to ensure order
        n_class=n_class
    )
    ft_score = pd.DataFrame(index=data_trn.columns)
    for i, target_class in enumerate(range(n_class)):
        target_class_name = list(fact_dic.keys())[i]
        gene_pds = gene_pds_all_classes[target_class]
        assert len(gene_pds) == len(mdic['mods_uni'])
        for k in range(len(gene_pds)):
            ft_score.loc[gene_pds[k]['genes'].values, f'score_{target_class_name}'] = gene_pds[k]['values'].values
    assert ~ft_score.isna().any().any()

    print(f"DeepKEGG BK identification running time: {time.perf_counter() - st_time:.2f} seconds.")
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n Peak GPU memory during BK identification: {peak_mb:.1f} MB")

    return ft_score, perf

