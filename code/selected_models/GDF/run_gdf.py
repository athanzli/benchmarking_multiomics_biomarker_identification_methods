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

run_gdf_r = """
run_gdf_r <- function(
    ppi,
    X1_trn, X2_trn, X3_trn,  # column names for features should be gene names without any prefix or suffix. Same gene sets should be used for all 3 mods.
    y_trn, y_tst, # ensure Y has been encoded into integer 0 and 1
    X1_tst, X2_tst, X3_tst,
    mods_names
) {
  set.seed(42)

  # get the induced graph structure
  graph <- graph_from_edgelist(as.matrix(ppi), directed = FALSE)
  
  #
  features <- setNames(list(as.matrix(X1_trn),
                            as.matrix(X2_trn),
                            as.matrix(X3_trn)), mods_names)
  
  dfnet_graph <- launder(graph, features, threshold = NaN)
  
  # train the GDF
  dfnet_forest <- train(,
                        dfnet_graph$graph,
                        dfnet_graph$features, y_trn,
                        # to align with the paper's setting for TCGA data
                        importance="impurity",
                        splitrule = "gini",
                        min.walk.depth = 2,
                        ntrees = 500,
                        niter = 100,
                        initial.walk.depth = 30)
  last_gen <- tail(dfnet_forest, 1)
  
  # feature importance
  feat_imp <- feature_importance(last_gen, dfnet_graph$features)
  
  # Prepare test data
  test_features <- setNames(list(as.matrix(X1_tst), as.matrix(X2_tst), as.matrix(X3_tst)), mods_names)
  DATA_test <- DFNET:::flatten2ranger(common_features(test_features))
  
  # use the ensemble as in the paper
  out <- predict(last_gen, DATA_test)
  # P(class=1) among participating trees
  prob <- ifelse(out$predictions == 1, out$approval.rate, 1 - out$approval.rate) * out$participation.rate
  
  # binary classif metrics
  to_binary <- function(y) {
    if (is.factor(y) || is.character(y)) {
      yf <- as.factor(y)
      as.integer(yf == tail(levels(yf), 1L))  # last level treated as positive (1)
    } else {
      yv <- as.integer(y)
      u  <- sort(unique(yv))
      if (length(u) == 2L && !all(u %in% c(0,1))) as.integer(yv == max(u)) else yv
    }
  }
  
  compute_metrics <- function(prob, y_true) {
    y_true <- to_binary(y_true)
    prob   <- as.numeric(prob)
    prob   <- pmin(pmax(prob, 0), 1) # clip
    y_hat  <- as.integer(prob >= 0.5)
    
    sens <- ModelMetrics::sensitivity(y_true, y_hat)
    spec <- ModelMetrics::specificity(y_true, y_hat)
    
    ##### accuracy and AUPR
    acc_score <- mean(y_hat == y_true, na.rm = TRUE)
    pr <- pr.curve(scores.class0 = prob[y_true==1], scores.class1 = prob[y_true==0])
    aupr_score <- pr$auc.integral
    #####
    
    data.frame(
      roc_auc          = ModelMetrics::auc(y_true, prob),
      aucpr           = aupr_score,
      acc         = acc_score,
      balanced_acc = 0.5 * (sens + spec),
      recall           = ModelMetrics::recall(y_true, y_hat),
      precision        = ModelMetrics::precision(y_true, y_hat),
      f1               = ModelMetrics::f1Score(y_true, y_hat),
      mcc              = ModelMetrics::mcc(y_true, prob, 0.5),
      check.names = FALSE
    )
  }
  
  perf <- compute_metrics(prob, y_tst)
  
  flat_feat_imp <- {
    m <- as.matrix(feat_imp)
    stopifnot(!is.null(rownames(m)), !is.null(colnames(m)))
    tmp <- as.data.frame(as.table(m), stringsAsFactors = FALSE)
    rownames(tmp) <- paste0(tmp$Var2, "@", tmp$Var1)# mod@gene
    flat_feat_imp <- tmp["Freq"]
    colnames(flat_feat_imp) <- "importance"
    flat_feat_imp
  }  
  
  # return ft_imp and perf
  list(
    flat_feat_imp,
    perf
  )
}
"""
def run_gdf(
    ppi: pd.DataFrame,
    data_trn: pd.DataFrame,
    label_trn: pd.Series,
    data_tst: pd.DataFrame,
    label_tst: pd.Series):
    # assertions. Ensure
    #    1) ppi (two columns) have the same gene set as data_trn and data_tst (columns)
    #    2) data_trn, data_tst, label_trn, label_tst have aligned indices
    #    3) data_trn and data_tst have the same columns (order aligned)
    mdic = mod_mol_dict(data_trn.columns)
    assert np.array([(data_trn.columns[mdic['mods']==mod]==data_tst.columns[mdic['mods']==mod]).all() for mod in mdic['mods_uni']]).all()
    gset = np.array(sorted(list(set(mdic['mols']))))
    ppi_genes = np.array(sorted(list(set(ppi.values.flatten().astype(str)))))
    assert (gset == ppi_genes).all()
    assert np.array([gset==gset_cur_mod for gset_cur_mod in [np.array(sorted(list(set(data_trn.columns[mdic['mods']==mod].str.split(SPLITTER).str[1])))) for mod in mdic['mods_uni']]]).all()
    assert (data_trn.index == label_trn.index).all()
    assert (data_tst.index == label_tst.index).all()
    assert data_trn.columns.equals(data_tst.columns)

    import rpy2.robjects as ro
    from rpy2.robjects import r as R
    from rpy2.robjects import pandas2ri, numpy2ri, vectors as rvec
    from rpy2.robjects.conversion import localconverter

    try:
        R('suppressMessages({library(ranger); library(igraph); library(pROC); '
          'library(DFNET); library(ModelMetrics); library(PRROC)})')
    except Exception as e:
        raise RuntimeError(f"Failed to load required R packages: {e}")
    ro.r(run_gdf_r)

    # encode labels to 0 1
    y_trn, _ = factorize_label(label_trn.values.flatten())
    y_tst, _ = factorize_label(label_tst.values.flatten())

    # 
    mods = np.array([c.split(SPLITTER)[0] for c in data_trn.columns])
    mods_uni = np.unique(mods)
    if len(mods_uni) not in [2,3]:
        raise ValueError(f"Expected 2 or 3 modalities; got {len(mods_uni)}: {mods_uni}")

    trn_by_mod = [data_trn.loc[:, mods == m].copy() for m in mods_uni]
    tst_by_mod = [data_tst.loc[:, mods == m].copy() for m in mods_uni]

    # 
    def _strip_to_gene(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [c.split(SPLITTER, 1)[1] for c in out.columns]
        return out

    trn_by_mod = [_strip_to_gene(df) for df in trn_by_mod]
    tst_by_mod = [_strip_to_gene(df) for df in tst_by_mod]

    genes = trn_by_mod[0].columns
    assert np.array([(df.columns == genes).all() for df in trn_by_mod]).all()
    trn_by_mod = [df.loc[:, genes].astype(float) for df in trn_by_mod]
    tst_by_mod = [df.loc[:, genes].astype(float) for df in tst_by_mod]

    st_time = time.perf_counter()

    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        r_ppi     = pandas2ri.py2rpy(ppi.iloc[:, :2])
        X1_trn, X2_trn, X3_trn = [pandas2ri.py2rpy(df) for df in trn_by_mod]
        X1_tst, X2_tst, X3_tst = [pandas2ri.py2rpy(df) for df in tst_by_mod]
        r_y_trn  = rvec.IntVector(y_trn.tolist())
        r_y_tst  = rvec.IntVector(y_tst.tolist())
        r_mods   = rvec.StrVector(mods_uni.tolist())

    r_res = R['run_gdf_r'](
        r_ppi,
        X1_trn, X2_trn, X3_trn,
        r_y_trn, r_y_tst,
        X1_tst, X2_tst, X3_tst,
        r_mods
    )
    print(f"GDF (training time + BK identification time) running time: {time.perf_counter() - st_time:.2f} s")

    flat_feat_imp = pandas2ri.rpy2py(r_res.rx2(1))
    ft_score = flat_feat_imp.rename(columns={'importance': 'score'})
    perf          = pandas2ri.rpy2py(r_res.rx2(2))
    perf = {col_name : perf.iloc[0][col_name] for col_name in perf.columns}

    return ft_score, perf
