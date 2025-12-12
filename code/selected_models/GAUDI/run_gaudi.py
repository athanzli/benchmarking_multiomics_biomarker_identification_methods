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

# library(gaudi)
run_gaudi_r = """
run_gaudi_r <- function(
  X1, # samples as rows, features as columns
  X2,
  X3,
  mods_names
) {
  omics_list <- setNames(list(as.matrix(X1),
                            as.matrix(X2),
                            as.matrix(X3)), mods_names)
  ##### data checks
  ids <- Reduce(intersect, lapply(omics_list, rownames))
  omics_list <- lapply(omics_list, function(M) M[ids, , drop = FALSE])
  # numeric and no NA
  omics_list <- lapply(omics_list, function(M) { storage.mode(M) <- "double"; M })
  stopifnot(all(vapply(omics_list, function(M) !anyNA(M), logical(1))))
  # drop 0-var or all-NA cols
  nzv <- function(M) M[, apply(M, 2, function(v) var(v, na.rm = TRUE) > 0 && !all(is.na(v))), drop = FALSE]
  omics_list <- lapply(omics_list, nzv)
  #####

  if (nrow(X1) < 100) {
    result <- gaudi(omics = omics_list,
                     umap_params = list(n_neighbors = 5, n_components = 4, pca = min(nrow(omics_list[[1]]),
                                                                                     ncol(omics_list[[1]]))),
                     umap_params_conc = list(n_neighbors = 5, n_components = 2),
                     method = 'rf'
    )
  } else {
    result <- gaudi(omics = omics_list,
                     umap_params = list(n_neighbors = 15, n_components = 4, pca = min(nrow(omics_list[[1]]),
                                                                                      ncol(omics_list[[1]]))),
                     umap_params_conc = list(n_neighbors = 15, n_components = 2),
                     method = 'rf'
    )
  } 
  
  # check clustering quality
  print("silhouette_score:")
  print(result@silhouette_score)
  
  ###### for result1
  # stack the three tables
  cat_df <- do.call(rbind, lapply(result@metagenes, function(x) {
    x <- as.data.frame(x, stringsAsFactors = FALSE)
    x$contrib1 <- as.numeric(x$contrib1)
    x$contrib2 <- as.numeric(x$contrib2)
    x
  }))
  # collapse to a single score
  # max-pool across the contrib columns to get one score
  score_vec <- pmax(cat_df$contrib1, cat_df$contrib2, na.rm = TRUE)
  # 
  tmp <- data.frame(feature = rownames(cat_df), score = score_vec)
  score_df <- aggregate(score ~ feature, data = tmp, FUN = max)
  # 1 col
  row.names(score_df) <- score_df$feature
  score_df$feature <- NULL
  ft_score <- score_df[order(score_df$score, decreasing = TRUE), , drop = FALSE]
  ft_score
}
"""
def run_gaudi(data: pd.DataFrame):
    r"""
    Args:
        data (pd.DataFrame): rows are samples, columns are "mod@gene"
    """
    mdic = mod_mol_dict(data.columns)
    data = {m: data.loc[:, mdic['mods'] == m].copy() for m in mdic['mods_uni']}

    import rpy2.robjects as ro
    from rpy2.robjects import r as R
    from rpy2.robjects import pandas2ri, numpy2ri, vectors as rvec
    from rpy2.robjects.conversion import localconverter

    try:
        R('library(gaudi)')
    except Exception as e:
        raise RuntimeError(f"Failed to load required R packages: {e}")
    ro.r(run_gaudi_r)

    # 
    st_time = time.perf_counter()
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        X1, X2, X3 = [pandas2ri.py2rpy(df) for df in list(data.values())]
        r_mods   = rvec.StrVector(mdic['mods_uni'].tolist())

    r_res = R['run_gaudi_r'](
        X1, X2, X3,
        r_mods
    )
    print(f"GAUDI (training time + BK identification time) running time: {time.perf_counter() - st_time:.2f} s")

    ft_score = pandas2ri.rpy2py(r_res)

    return ft_score
