import numpy as np
import pandas as pd
import time
SPLITTER = '@'
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

# library(omicade4)
run_mcia_r = """
run_mcia_r <- function(X1,X2,X3,mods_names) {
  # X1, X2, X3 are omics matrices. rows=features, cols=samples (same order)
  dfs <- setNames(list(t(X1), t(X2), t(X3)), mods_names)
  # check samples
  cn <- sapply(dfs, colnames)
  stopifnot(all(apply(cn[,-1, drop = FALSE], 2, function(y) identical(y, cn[,1]))))
  # MCIA
  mcoin <- mcia(dfs)
  # get feature scores
  feat_coords <- mcoin$mcoa$Tco # rows = features (all datasets), cols = axes
  ds_id       <- mcoin$mcoa$TC$T # factor telling which dataset each row came from
  loadings <- data.frame(
    feature = rownames(feat_coords),
    dataset = names(dfs)[as.integer(ds_id)],
    axis1   = feat_coords[, 1],
    axis2   = feat_coords[, 2],
    stringsAsFactors = FALSE
  )
  
  loadings$abs_axis1 <- abs(loadings$axis1)
  loadings$abs_axis2 <- abs(loadings$axis2)
  ft_score <- loadings[,5:6]
  rownames(ft_score) <- loadings[,1]
  
  ft_score$score <- pmax(ft_score$abs_axis1, ft_score$abs_axis2)
  ft_score <- ft_score["score"]
}
"""
def run_mcia(data: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        data (pd.DataFrame): rows are samples, columns are "mod@molecule"
    Returns:
        ft_score (pd.DataFrame): index restored to original "mod@molecule" names
    """
    mdic = mod_mol_dict(data.columns)

    # 
    data_by_mod = {}
    for m in mdic['mods_uni']:
        mask = (mdic['mods'] == m)
        df = data.loc[:, mask].copy()
        # df.columns = mdic['mols'][mask]
        data_by_mod[m] = df

    import rpy2.robjects as ro
    from rpy2.robjects import r as R
    from rpy2.robjects import pandas2ri, numpy2ri, vectors as rvec
    from rpy2.robjects.conversion import localconverter

    try:
        R('library(omicade4)')
    except Exception as e:
        raise RuntimeError(f"Failed to load required R packages: {e}")
    ro.r(run_mcia_r)

    # map R/ade4-sanitized rownames back to originals (.->@)
    def r_make_names(strings):
        return list(R['make.names'](rvec.StrVector(list(strings))))

    san_mods_per_col = r_make_names(mdic['mods'])
    san_mols_per_col = r_make_names(mdic['mols'])
    rname_to_orig = {
        f"{sm}.{sn}": f"{m}{SPLITTER}{n}" # NOTE
        for sm, sn, m, n in zip(san_mods_per_col, san_mols_per_col, mdic['mods'], mdic['mols'])
    }

    t0 = time.perf_counter()
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        X1, X2, X3 = [pandas2ri.py2rpy(data_by_mod[m]) for m in mdic['mods_uni']]
        r_mods   = rvec.StrVector(mdic['mods_uni'].tolist())
    r_res = R['run_mcia_r'](X1, X2, X3, r_mods)
    ft_score = pandas2ri.rpy2py(r_res)
    print(f"MCIA (training time + BK identification time) running time: {time.perf_counter() - t0:.2f} s")

    # restore original feature names
    idx = ft_score.index.astype(str)
    new_idx = [rname_to_orig.get(k, k) for k in idx]
    # check
    if any(k not in list(rname_to_orig.keys()) for k in idx):
        unmapped = [k for k in idx if k not in rname_to_orig]
        raise ValueError(f"Unmapped ft names: {unmapped}")
    if ft_score.index.has_duplicates:
        raise ValueError("ft name collision after restoring")
    ft_score.index = pd.Index(new_idx, name=None)
    return ft_score
