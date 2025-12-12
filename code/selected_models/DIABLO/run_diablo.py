import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score
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

# code v2
run_diablo_r = """
run_diablo <- function (
    data,
    label,
    data_tst,
    label_tst
) {
  Y     <- factor(as.character(label),     levels = c("0","1"))
  Y_tst <- factor(as.character(label_tst), levels = levels(Y))

  ncomp <- 3

  data_filtered <- lapply(data, function(df) {
    nzv <- caret::nearZeroVar(df)
    if (length(nzv) > 0) df <- df[, -nzv, drop = FALSE]
    df
  })
  data_tst_filtered <- mapply(function(train_df, test_df) {
    if (ncol(train_df) > 0) test_df[, colnames(train_df), drop = FALSE] else test_df
  }, data_filtered, data_tst, SIMPLIFY = FALSE)

  pooled_abs_loadings <- function(fit, ncomp) {
    Lall <- fit$loadings
    loadings_X <- if (!is.null(Lall$X)) Lall$X else Lall[-fit$indY]  # robust to versions
    w_list <- lapply(loadings_X, function(L) {
      L <- as.matrix(L[, seq_len(ncomp), drop = FALSE])              # p x ncomp
      v <- if (ncol(L) == 1L) abs(L[, 1]) else apply(abs(L), 1, max)
      stats::setNames(v, rownames(L))                                 # keep feature names
    })
    w_vec <- unlist(w_list, use.names = FALSE)
    names(w_vec) <- unlist(lapply(w_list, names), use.names = FALSE)
    w_vec
  }

  # master feature index (same rows for both columns)
  all_features <- unique(unlist(lapply(data_filtered, colnames)))
  ft_score <- data.frame(sparse = numeric(length(all_features)),
                         all     = numeric(length(all_features)))
  rownames(ft_score) <- all_features

  # 
  keep1pct <- lapply(data_filtered, function(df) {
    p <- ncol(df)
    k <- max(1L, floor(0.01 * p))      # 1% per component, at least 1
    rep(k, ncomp)
  })
  names(keep1pct) <- names(data_filtered)

  opt_model <- mixOmics::block.splsda(
    X = data_filtered, Y = Y,
    ncomp = ncomp,
    design = "full",
    keepX  = keep1pct
  )

  w_opt <- pooled_abs_loadings(opt_model, ncomp)
  ft_score[names(w_opt), "sparse"] <- w_opt

  # predictions
  pred_opt <- stats::predict(opt_model, newdata = data_tst_filtered, dist = "all")
  if (!is.null(pred_opt$WeightedPredict)) {
    score3d <- pred_opt$WeightedPredict 
  } else {
    arrs <- lapply(pred_opt$predict, function(a) a[, , seq_len(ncomp), drop = FALSE])
    score3d <- Reduce(`+`, arrs) / length(arrs)
  }
  scores <- score3d[, , ncomp, drop = TRUE] 
  colnames(scores) <- levels(Y)

  Y_pred <- pred_opt$WeightedVote$centroids.dist[, ncomp]
  Y_pred <- factor(Y_pred, levels = levels(Y))

  #
  full_model <- mixOmics::block.splsda(
    X = data_filtered, Y = Y,
    ncomp = ncomp,
    design = "full"   
  )

  w_all <- pooled_abs_loadings(full_model, ncomp)
  ft_score[names(w_all), "all"] <- w_all

  #
  return(list(
    ft_score = ft_score,
    y_pred   = as.character(Y_pred),
    scores   = scores,
    classes  = levels(Y)
  ))
}
"""

def run_diablo(
    data_trn,
    label_trn,
    data_tst,
    label_tst):
    r"""
    """
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects import vectors as rvec
    robjects.r(run_diablo_r)
    pandas2ri.activate()
    robjects.r('library(mixOmics)')
    robjects.r('library(caret)')

    # Encode labels
    label_trn, _ = factorize_label(label_trn.values.flatten())
    label_tst, _ = factorize_label(label_tst.values.flatten())
    #
    ft_score = pd.DataFrame(index=data_trn.columns, columns=['sparse','all'], data=0.0)
    mods = np.array([col.split(SPLITTER)[0] for col in data_trn.columns])
    mods_uni = np.unique(mods)
    data_trn_list = [data_trn.loc[:, mods == mods_uni[i]] for i in range(len(mods_uni))]
    data_tst_list = [data_tst.loc[:, mods == mods_uni[i]] for i in range(len(mods_uni))]

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_data_list = robjects.ListVector({
            f"X{i+1}": pandas2ri.py2rpy(df) for i, df in enumerate(data_trn_list)
        })
        r_data_tst_list = robjects.ListVector({
            f"X{i+1}": pandas2ri.py2rpy(df) for i, df in enumerate(data_tst_list)
        })
    r_y     = robjects.StrVector(label_trn.astype(str))
    r_y_tst = robjects.StrVector(label_tst.astype(str))

    st_time = time.perf_counter()
    run_diablo_func = robjects.globalenv['run_diablo']
    result = run_diablo_func(r_data_list, r_y, r_data_tst_list, r_y_tst)
    print(f"DIABLO (training time + BK identification time) running time: {time.perf_counter() - st_time:.2f} s")

    ########################################### ft score
    with localconverter(default_converter + pandas2ri.converter):
        ft_score_r = robjects.conversion.rpy2py(result.rx2('ft_score'))
    with localconverter(default_converter):
        y_pred = list(robjects.conversion.rpy2py(result.rx2('y_pred')))
    scores_obj = result.rx2('scores')
    if isinstance(scores_obj, np.ndarray):
        scores_r = scores_obj
    else:
        with localconverter(default_converter + numpy2ri.converter):
            scores_r = robjects.conversion.rpy2py(scores_obj)
    classes_obj = result.rx2('classes')
    if isinstance(classes_obj, (list, tuple, np.ndarray)):
        classes_r = list(classes_obj)
    else:
        with localconverter(default_converter):
            classes_r = list(robjects.conversion.rpy2py(classes_obj))

    assert classes_r == ["0","1"]
    for col in ft_score_r.columns:
        ft_score.loc[ft_score_r.index, col] = ft_score_r[col].values

    ### get a single ranking
    # rank non-zero features in sparse column first, then rank non-zero features in all column next (lower ranks than sparse)
    ft_score_rank = pd.DataFrame(index=ft_score.index, columns=['rank'], data=np.nan)
    # rank sparse column
    nonzero_sparse = ft_score['sparse'] != 0
    ft_score_rank.loc[nonzero_sparse, 'rank'] = ft_score.loc[nonzero_sparse, 'sparse'].rank(ascending=False)
    # rank all column
    nonzero_all = (ft_score['all'] != 0) & (~nonzero_sparse)
    ft_score_rank.loc[nonzero_all, 'rank'] = (ft_score.loc[nonzero_all, 'all'].rank(ascending=False) +
                                              ft_score_rank['rank'].max())
    # rank zero features last
    zero_features = ft_score['all'] == 0
    ft_score_rank.loc[zero_features, 'rank'] = ft_score_rank['rank'].max() + 1
    ft_score_rank = ft_score_rank.astype(int)

    ################### perf ###################
    assert len(classes_r) == 2, f"Binary expected, got {len(classes_r)} classes: {classes_r}"
    scores_df = pd.DataFrame(scores_r, columns=classes_r)
    pos_label = classes_r[1]           # change to classes_r[0] if you prefer the other class as positive
    neg_label = classes_r[0]
    # y_true as strings, then binarize wrt pos_label
    y_true = label_tst.flatten().astype(str)
    if pos_label not in scores_df.columns:
        raise ValueError(f"Positive label '{pos_label}' not found in score columns {list(scores_df.columns)}")
    s_pos = scores_df[pos_label].astype(float).values
    auroc = roc_auc_score((y_true == pos_label).astype(int), s_pos)
    aupr  = average_precision_score((y_true == pos_label).astype(int), s_pos)

    ######### perf
    recall = precision = f1 = f1_weighted = f1_macro = None
    recall    = recall_score(y_true, y_pred, pos_label=pos_label)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    f1        = f1_score(y_true, y_pred, pos_label=pos_label)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"AUC-ROC:        {auroc:.4f}")
    print(f"AUCPR:          {aupr:.4f}")
    print(f"F1:  {f1:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"MCC: {mcc:.4f}")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy:       {acc:.4f}")
    perf = {
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'roc_auc': auroc,
        'aucpr': aupr,
        'mcc' : mcc,
        'balanced_acc': balanced_acc,
        'auroc':auroc,
        'aupr':aupr
    }
    print("Performance:", perf)

    return ft_score, ft_score_rank, perf # NOTE

