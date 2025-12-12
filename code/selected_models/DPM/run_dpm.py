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


# library('ActivePathways')
run_dpm_r = """
run_dpm <- function(data, label, cv=NULL) {
  # check data
  if (!is.list(data) || length(data) < 2) stop("'data' must be a list of >=2 omics matrices")
  if (!is.data.frame(label) || ncol(label) != 1) stop("'label' must be a 1-column data.frame of 0, 1")
  y <- as.integer(label[[1]])
  if (!all(y %in% c(0L,1L))) stop("label must contain only 0/1")
  nms <- names(data); if (is.null(nms)) nms <- paste0("ds", seq_along(data))
  
  # sample alignment
  smp <- Reduce(intersect, c(list(rownames(label)), lapply(data, rownames)))
  if (length(smp) < 2) stop("no overlapping samples")
  y <- y[match(smp, rownames(label))]
  
  # gene alignment
  genes <- Reduce(intersect, lapply(data, colnames))
  if (length(genes) < 1) stop("no shared genes")
  
  mats <- lapply(data, function(X) {
    X <- as.matrix(X[match(smp, rownames(X)), genes, drop = FALSE])
    storage.mode(X) <- "double"; X
  })
  names(mats) <- nms
  
  # binary group only
  # unadjusted Wilcoxon P; directions are unit signs from group difference.
  wilcox_by_gene <- function(M, y) {
    g1 <- which(y == 1L); g0 <- which(y == 0L)
    if (length(g1) < 1 || length(g0) < 1) stop("both groups need >=1 sample")
    p <- dir <- numeric(ncol(M)); names(p) <- names(dir) <- colnames(M)
    for (j in seq_len(ncol(M))) {
      x1 <- M[g1, j]; x0 <- M[g0, j]
      # nan or zerovars
      if (all(!is.finite(x1)) || all(!is.finite(x0)) || (sd(x1, na.rm=TRUE)==0 && sd(x0, na.rm=TRUE)==0)) {
        p[j] <- 1; dir[j] <- 0
      } else {
        suppressWarnings({
          wt <- try(stats::wilcox.test(x1, x0, alternative="two.sided", exact=FALSE), silent=TRUE)
        })
        p[j]   <- if (inherits(wt, "try-error") || is.null(wt$p.value) || !is.finite(wt$p.value)) 1 else wt$p.value
        dbar   <- stats::median(x1, na.rm=TRUE) - stats::median(x0, na.rm=TRUE)
        dir[j] <- ifelse(!is.finite(dbar) || dbar == 0, 0, ifelse(dbar > 0,  +1, -1))
      }
    }
    list(p = p, dir = dir)
  }
  
  stats_list <- lapply(mats, wilcox_by_gene, y = y)
  
  # gene by dataset matrix of p and directions
  pmat  <- do.call(cbind, lapply(stats_list, function(s) s$p))
  dirm  <- do.call(cbind, lapply(stats_list, function(s) s$dir))
  rownames(pmat) <- rownames(dirm) <- colnames(mats[[1]])
  colnames(pmat) <- colnames(dirm) <- nms
  
  # replace missing with 1
  pmat[!is.finite(pmat)]  <- 1
  dirm[!is.finite(dirm)]  <- 0
  
  # constraints vector
  guess_constraints <- function(nms) {
    cv <- rep(0, length(nms)); names(cv) <- nms
    # mRNA as baseline
    is_rna <- grepl("(^|[^a-z])rna($|[^a-z])|mrna|expr|transcr", nms, ignore.case=TRUE)
    if (any(is_rna)) cv[which(is_rna)[1]] <- +1 else cv[1] <- +1
    # 
    cv[grepl("protein|proteom", nms, ignore.case=TRUE)] <- +1
    cv[grepl("cnv|copy",        nms, ignore.case=TRUE)] <- +1
    cv[grepl("meth|dnam|methyl|methy",       nms, ignore.case=TRUE)] <- -1
    cv[grepl("mirna|miRNA|\\bmir\\b", nms, ignore.case=TRUE)] <- -1
    cv[grepl("snv|mut|variant", nms, ignore.case=TRUE)] <-  0
    unname(cv)
  }
  if (is.null(cv)) {
    cv <- guess_constraints(nms)
  }
    
  # enforce DPM rule for non-direc omic (e.g. snv)
  cv <- as.integer(sign(cv))
  if (length(cv) != ncol(dirm)) stop("cv length should equal number of datasets")
  idx_nondir <- which(cv == 0L)
  if (length(idx_nondir)) {
    dirm[, idx_nondir] <- 0L
  }

  # DPM merge
  merged <- ActivePathways::merge_p_values(
    scores = pmat,
    method = "DPM",
    scores_direction  = dirm,
    constraints_vector = cv
  )
  #
  merged <- if (is.list(merged)) unlist(merged, use.names = TRUE) else merged
  merged <- as.numeric(merged); names(merged) <- rownames(pmat)
  
  ft_score <- data.frame(score = merged[rownames(pmat)], row.names = rownames(pmat))
  ft_score <- ft_score[order(ft_score$score, decreasing = FALSE), , drop = FALSE]
  ft_score
}
"""
def run_dpm(data: pd.DataFrame, label: pd.DataFrame, cv=None) -> pd.DataFrame:
    data_by_mod = data
    mods_uni = np.unique(list(data_by_mod.keys()))
    # assert gset in all mods are the same
    gset = set(data_by_mod[mods_uni[0]].columns)
    for m in mods_uni[1:]:
        assert gset == set(data_by_mod[m].columns), "genes are not the same across omics"

    if isinstance(label, pd.Series):
        lab_df = pd.DataFrame(label)
    else:
        lab_df = label.copy()
    if lab_df.shape[1] != 1:
        raise ValueError("")
    lab_df = lab_df.copy()
    lab_df.iloc[:, 0] = pd.to_numeric(lab_df.iloc[:, 0], errors='raise').astype(int)

    import rpy2.robjects as ro
    from rpy2.robjects import r as R
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter

    R('library(ActivePathways)')
    ro.r(run_dpm_r)

    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        r_data = ro.ListVector({k: pandas2ri.py2rpy(v) for k, v in data_by_mod.items()})
        r_label = pandas2ri.py2rpy(pd.DataFrame(lab_df.iloc[:, 0]))
        if cv is None:
            r_cv = ro.NULL
        else:
            r_cv = ro.IntVector(cv)

    st_time = time.perf_counter()
    r_fun = R['run_dpm']
    r_res = r_fun(r_data, r_label, r_cv)
    print(f"DPM running time: {time.perf_counter() - st_time:.2f} s")

    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        ft_score = pandas2ri.rpy2py(r_res)

    ft_score = pd.DataFrame(ft_score).copy()
    ft_score.columns = ['score']
    ft_score.index.name = None
    return ft_score
