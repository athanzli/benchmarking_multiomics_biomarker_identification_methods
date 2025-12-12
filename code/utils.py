import pandas as pd
import numpy as np
import time

SPLITTER = '@'
# TCGA
try:
    P2G = pd.read_csv("../data/TCGA/TCGA_protein2gene_mapping.csv", index_col=0)
    C2G = pd.read_csv("../data/TCGA/TCGA_cpg2gene_mapping.csv", index_col=0)
    R2G = pd.read_csv("../data/TCGA/TCGA_miRNA2gene_mapping.csv", index_col=0)
except:
    try:
        P2G = pd.read_csv("/home/athan.li/eval_bk/data/TCGA/TCGA_protein2gene_mapping.csv", index_col=0)
        C2G = pd.read_csv("/home/athan.li/eval_bk/data/TCGA/TCGA_cpg2gene_mapping.csv", index_col=0)
        R2G = pd.read_csv("/home/athan.li/eval_bk/data/TCGA/TCGA_miRNA2gene_mapping.csv", index_col=0)
    except:
        P2G = pd.read_csv("D:/Projects/eval_bk/data/TCGA/TCGA_protein2gene_mapping.csv", index_col=0)
        C2G = pd.read_csv("D:/Projects/eval_bk/data/TCGA/TCGA_cpg2gene_mapping.csv", index_col=0)
        R2G = pd.read_csv("D:/Projects/eval_bk/data/TCGA/TCGA_miRNA2gene_mapping.csv", index_col=0)

TCGA_DATA_PATH = '/data/zhaohong/TCGA_data/data/'

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

def factorize_label(y):
    r"""
    Factorize the label.
    """
    y_uni = np.unique(y)
    value_to_index = {value: idx for idx, value in enumerate(y_uni)}
    y_fac = np.array([value_to_index[item] for item in y]).astype(np.int64)
    return y_fac, value_to_index

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

def explode_df_col(
    df,
    col_name
):
    df[col_name] = df[col_name].str.split(r'[;,/.]')
    df = df.explode(col_name).reset_index(drop=True)
    # remove '[', ']', and ' ' in the beginning and end of the string
    df[col_name] = df[col_name].str.strip('[]').str.strip()
    return df

def count_parameters(model):
    """
    Counts the total and trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        total_params (int): Total number of parameters.
        trainable_params (int): Number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model.__class__.__name__}: {total_params:,} total parameters, {trainable_params:,} trainable.")


def load_TCGA(
        mods=None,
        load_only_rowcol_names=False,
):
    r"""
    NOTE: change DATA_PATH accordingly.

    Args:
        mods: list of str, e.g. ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA', 'protein']

    """
    assert mods is not None
    mods = np.unique(mods) # sorted
    if not load_only_rowcol_names:
        data = [pd.read_csv(TCGA_DATA_PATH + f"processed/{mod}_mat.csv", index_col=0) for mod in mods]
        return data
    else:
        cols_all = []
        rows_all = []
        for mod in mods:
            cols = pd.read_csv(TCGA_DATA_PATH + f"processed/{mod}_mat.csv", index_col=0, nrows=0).columns.values
            rows = pd.read_csv(TCGA_DATA_PATH + f"processed/{mod}_mat.csv", index_col=0, usecols=[0]).index.values
            cols_all.append(cols)
            rows_all.append(rows)
        return rows_all, cols_all

def print_nan_positions(df):
    nan_mask = df.isnull()
    nan_locations = nan_mask.stack()
    nan_positions = nan_locations[nan_locations]
    print(nan_positions.index.tolist())

def find_non_alphanumeric_characters_from_array_of_strings(arr):
    import re
    non_alphanumerics = set()
    for s in arr:
        matches = re.findall(r'[^a-zA-Z0-9]', s)
        non_alphanumerics.update(matches)
    print("Non-alphanumeric characters found:", non_alphanumerics)

def timeit(func):
    def wrapper(*args, **kwargs):
        print(f"Running '{func.__name__}'...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"'{func.__name__}' runtime: {elapsed_time:.2f} seconds.")
        return result
    return wrapper

def modmol_gene_set_tcga(
    mod_mol_ids,
    op='union',
    c2g=C2G,
    p2g=P2G,
    r2g=R2G
):
    r"""
    Args:
        mod_mol_ids: a 1d np array with mod@mol 
    Returns:
        np.array: a 1d array of mapped and intersected gene names
    """
    mappings = {
        'protein' : p2g,
        'DNAm' : c2g,
        'miRNA': r2g
    }
    gss = []
    mmdic = mod_mol_dict(mod_mol_ids)
    for mod in mmdic['mods_uni']:
        if mod in ['mRNA', 'CNV', 'SNV']:
            gss.append(mmdic['mols'][mmdic['mods']==mod])
            continue
        elif mod in ['DNAm', 'protein', 'miRNA']:
            mols = mmdic['mols'][mmdic['mods']==mod]
            try:
                gss.append(mappings[mod].loc[mols, 'gene'].unique())
            except:
                mols = np.intersect1d(mols, mappings[mod].index.values)
                gss.append(mappings[mod].loc[mols, 'gene'].unique())
        else: raise ValueError(f"Unexpected mod {mod}")
    if op == 'union':
        gs = np.unique(np.concatenate(gss))
    elif op == 'intersection':
        gs = np.unique(list(set.intersection(*map(set, gss))))
    return gs

def beta_to_m(beta, offset=1e-10):
    """
    convert DNA methylation beta value(s) to M value(s)
    """
    b = np.asarray(beta, dtype=float)
    b = np.clip(b, offset, 1 - offset)
    return np.log2(b / (1 - b))

def convert_omics_to_gene_level(data_dict, cover_gset):
    r""" for BK eval project, TCGA multi-omics data only.
    Requiring special formatiing.

    Args:
        data_dict (dict): a dictionary of omics data. Each has samples as rows and features as columns.
            feature names must be without mod@ prefix.
        cover_gset (np.ndarray): a 1d array of gene names to cover, with ordering.
    """
    mods_uni = np.unique(list(data_dict.keys()))
    assert '@' not in data_dict[mods_uni[0]].columns[0], "Feature names must be without mod@ prefix."
    assert np.all([(data_dict[mods_uni[0]].index == data_dict[mods_uni[i]].index).all() for i in range(len(mods_uni))]), "Samples in different omics data do not match."
    assert type(cover_gset) == np.ndarray, "cover_gset must be a np.ndarray."
    original_index = data_dict[mods_uni[0]].index.values.copy()
    print("convert molecules to gene-level...")
    for mod in ['SNV', 'CNV', 'mRNA']:
        if mod in mods_uni:
            data_dict[mod] = data_dict[mod].reindex(columns=cover_gset, fill_value=0).astype(np.float32)
    if 'miRNA' in mods_uni:
        r2g_filtered = (R2G.loc[R2G['gene'].isin(cover_gset), ['gene', 'miRNA']]
                            .drop_duplicates())
        r2g_filtered.index = np.arange(len(r2g_filtered))
        valid_mirnas = data_dict['miRNA'].columns.intersection(r2g_filtered['miRNA'].unique())
        assert len(valid_mirnas) > 0
        dsub = data_dict['miRNA'][valid_mirnas]
        dsub_stacked = dsub.stack().reset_index()
        dsub_stacked.columns = ['sample', 'miRNA', 'value']
        merged = pd.merge(dsub_stacked, r2g_filtered, on='miRNA', how='inner')
        grouped = merged.groupby(['sample', 'gene'])['value'].mean().unstack(fill_value=0)
        data_dict['miRNA'] = grouped.reindex(columns=cover_gset, fill_value=0).astype(np.float32)
    if 'DNAm' in mods_uni:
        c2g_filtered = (C2G.loc[C2G['gene'].isin(cover_gset), ['gene', 'cpg.1']]
                            .drop_duplicates()
                            .rename(columns={'cpg.1': 'cpg'}))
        c2g_filtered.index = np.arange(len(c2g_filtered)) # to avoid both index and column are AGID, which raises error.
        valid_cpgs = data_dict['DNAm'].columns.intersection(c2g_filtered['cpg'].unique())
        assert len(valid_cpgs) > 0
        dsub = data_dict['DNAm'][valid_cpgs]
        dsub_stacked = dsub.stack().reset_index()
        dsub_stacked.columns = ['sample', 'cpg', 'value']
        merged = pd.merge(dsub_stacked, c2g_filtered, on='cpg', how='inner')
        grouped = merged.groupby(['sample', 'gene'])['value'].mean().unstack(fill_value=0)
        data_dict['DNAm'] = grouped.reindex(columns=cover_gset, fill_value=0).astype(np.float32)
    if 'protein' in mods_uni:
        p2g_filtered = (P2G.loc[P2G['gene'].isin(cover_gset), ['gene', 'AGID.1']]
                            .drop_duplicates()
                            .rename(columns={'AGID.1': 'AGID'}))
        p2g_filtered.index = np.arange(len(p2g_filtered)) # to avoid both index and column are AGID, which raises error.
        valid_agids = data_dict['protein'].columns.intersection(p2g_filtered['AGID'].unique())
        assert len(valid_agids) > 0
        dsub = data_dict['protein'][valid_agids]
        dsub_stacked = dsub.stack().reset_index()
        dsub_stacked.columns = ['sample', 'AGID', 'value']
        merged = pd.merge(dsub_stacked, p2g_filtered, on='AGID', how='inner')
        grouped = merged.groupby(['sample', 'gene'])['value'].mean().unstack(fill_value=0)
        data_dict['protein'] = grouped.reindex(columns=cover_gset, fill_value=0).astype(np.float32)
    for mod in mods_uni:
        assert (data_dict[mod].columns == cover_gset).all(), f"{mod} columns do not match cover_gset."
        data_dict[mod] = data_dict[mod].loc[original_index] # NOTE Restore order.
    return data_dict

def get_ppi(thres=None):
    assert thres is not None, "Please provide a threshold for combined_score."
    topo = pd.read_csv("../../../data/STRING_PPI_data/topology_filtered0.0.csv", index_col=0)
    ppi = topo.loc[topo['combined_score']>thres][['protein1', 'protein2']]
    return ppi
