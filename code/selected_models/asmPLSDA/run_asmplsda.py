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

# asmbPLS-DA. 1% and all.
run_asmplsda_r = """
asmbPLSDA_binary_train_test <- function(
    X1, X2, X3,  # column names for features should have prefix such as mRNA@
    Y, # ensure Y has been encoced into 0 and 1
    X_tst_1, X_tst_2, X_tst_3
) {
  library(asmbPLS)
  data(asmbPLS.example)
  print(asmbPLS.example$quantile.comb.table.cv)
  
  set.seed(42)
  PLS.comp = 3 # default; chosen in both tutorial example and paper
  thres_choices <- c(
    0.0, 0.99
  )
  
  stopifnot(nrow(X1) == nrow(X2), nrow(X2) == nrow(X3), length(Y) == nrow(X1))
  stopifnot(nrow(X_tst_1) == nrow(X_tst_2), nrow(X_tst_2) == nrow(X_tst_3))
  
  Y <- matrix(as.integer(Y), ncol = 1)
  
  # Build train and test design matrices
  X.dim <- c(ncol(X1), ncol(X2), ncol(X3))
  X1      <- as.matrix(X1);      X2      <- as.matrix(X2);      X3      <- as.matrix(X3)
  X_tst_1 <- as.matrix(X_tst_1); X_tst_2 <- as.matrix(X_tst_2); X_tst_3 <- as.matrix(X_tst_3)
  X_train <- cbind(X1, X2, X3)
  X_test  <- cbind(X_tst_1, X_tst_2, X_tst_3)
  storage.mode(X_train) <- "double"
  storage.mode(X_test)  <- "double"
  
  # 
  feature_order <- colnames(X_train)
  P <- length(feature_order)
  C <- length(thres_choices)
  ft_score <- matrix(NA_real_, nrow = P, ncol = C, dimnames = list(feature_order, as.character(thres_choices)))
  
  # fit
  for (thres in thres_choices) {
    quantile.comb <- matrix( # rows are components, and columns are blocks.
      thres,
      nrow = PLS.comp,
      ncol = length(X.dim),
      dimnames = list(NULL, paste0("block", seq_len(length(X.dim))))
    )
    fit <- asmbPLSDA.fit(
      X.matrix   = X_train,
      Y.matrix   = Y,
      PLS.comp   = PLS.comp,
      X.dim      = X.dim,
      quantile.comb = quantile.comb,
      outcome.type  = "binary"
    )
    # Biomarkers
    W_list <- lapply(seq_along(X.dim), function(b) {
      Wb <- fit$X_weight[[b]][, 1:PLS.comp, drop = FALSE]
      start <- if (b == 1) 1 else 1 + sum(X.dim[1:(b-1)])
      idx <- start:(start + X.dim[b] - 1)
      rownames(Wb) <- colnames(X_train)[idx]
      Wb
    })
    W <- do.call(rbind, W_list)
    W <- W[feature_order, , drop = FALSE]
    stopifnot(identical(rownames(W), feature_order))
    stopifnot(ncol(W) == PLS.comp)
    W_abs <- abs(W)
    score_vec <- apply(W_abs, 1, max)  # row-wise max over components
    ft_score[, as.character(thres)] <- score_vec
  }
  ft_score <- as.data.frame(ft_score, check.names = FALSE)
  stopifnot(!any(is.na(ft_score)))
  
  # use the last fitted model to predict
  pred_te <- asmbPLSDA.predict(fit, X.matrix.new = X_test,  PLS.comp = PLS.comp)
  y_pred_te  <- as.integer(pred_te$Y_pred)
  y_score_te <- as.numeric(pred_te$Y_pred_numeric) # higher the score, more likely to be 1 (pos class)
  
  list(
    pred_res  = list(y_pred = y_pred_te, y_score = y_score_te),
    ft_score = ft_score
  )
}
"""
def run_asmplsda(
    data_trn,
    label_trn,
    data_tst,
    label_tst):
    import rpy2.robjects as robjects
    from rpy2.robjects import r as R
    from rpy2.robjects import pandas2ri, numpy2ri, vectors as rvec
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter
    CV_PD = default_converter + pandas2ri.converter
    CV_NP = default_converter + numpy2ri.converter

    robjects.r(run_asmplsda_r)
    R('library(asmbPLS)')

    # encode labels
    label_trn, _ = factorize_label(label_trn.values.flatten())
    label_tst, _ = factorize_label(label_tst.values.flatten())
    y_tst = label_tst

    mods = np.array([col.split(SPLITTER)[0] for col in data_trn.columns])
    mods_uni = np.unique(mods)
    data_trn_list = [data_trn.loc[:, mods == m] for m in mods_uni]
    data_tst_list = [data_tst.loc[:, mods == m] for m in mods_uni]
    X1, X2, X3 = data_trn_list
    X_tst_1, X_tst_2, X_tst_3 = data_tst_list

    with localconverter(default_converter):
        run_func = R['asmbPLSDA_binary_train_test']

    with localconverter(CV_PD):
        X1r  = pandas2ri.py2rpy(X1)
        X2r  = pandas2ri.py2rpy(X2)
        X3r  = pandas2ri.py2rpy(X3)
        Xt1r = pandas2ri.py2rpy(X_tst_1)
        Xt2r = pandas2ri.py2rpy(X_tst_2)
        Xt3r = pandas2ri.py2rpy(X_tst_3)

    Y_vec = rvec.IntVector(label_trn.tolist())

    st_time = time.perf_counter()
    with localconverter(default_converter):
        r_res = run_func(X1r, X2r, X3r, Y_vec, Xt1r, Xt2r, Xt3r)
    with localconverter(default_converter):
        pred_res_r = r_res.rx2('pred_res')
        y_pred_r   = pred_res_r.rx2('y_pred')
        y_score_r  = pred_res_r.rx2('y_score')
        ft_score_r = r_res.rx2('ft_score')
    with localconverter(CV_NP):
        y_pred  = np.asarray(robjects.conversion.rpy2py(y_pred_r)).astype(int).ravel()
        y_score = np.asarray(robjects.conversion.rpy2py(y_score_r)).astype(float).ravel()
    with localconverter(CV_PD):
        ft_score = robjects.conversion.rpy2py(ft_score_r)  # pandas.DataFrame
    print(f"asmPLS-DA (training time + BK identification time) running time: {time.perf_counter() - st_time:.2f} s")

    # NOTE
    ft_score.columns = [1 - float(c) for c in ft_score.columns] # NOTE 1 - thres
    ft_score = ft_score.loc[data_trn.columns]
    ft_score = ft_score.reindex(columns=sorted(ft_score.columns))
    if set(np.unique(y_pred)) == {1, 2}:
        y_pred = y_pred - 1

    def ft_score_unifier(ft_score):
        pct_vals = ft_score.columns.astype(float).values.flatten()
        assert (pct_vals == sorted(pct_vals)).all()
        ranked = {}
        next_rank = 1
        for col in ft_score.columns:
            w = ft_score[col].abs().fillna(0.0)
            candidates = [feat for feat, val in w.items() if (val > 0) and (feat not in ranked)]
            if not candidates: # skip if no features has > 0 weight.
                continue
            candidates.sort(key=lambda k: (-w.loc[k], k)) # |weight| desc, then feature name asc
            for feat in candidates:
                ranked[feat] = next_rank
                next_rank += 1
        never_seen = [f for f in ft_score.index if f not in ranked] # zero weight in any col
        for feat in sorted(never_seen):
            ranked[feat] = next_rank
            next_rank += 1
        ft_score_rank = pd.DataFrame({"score": pd.Series(ranked, dtype=int)})
        ft_score_rank = ft_score_rank.loc[ft_score.index] # align.
        return ft_score_rank

    ft_score_rank = ft_score_unifier(ft_score)

    # metrics
    task_is_biclassif = len(np.unique(y_tst)) == 2
    perf = {
        'acc': float(accuracy_score(y_tst, y_pred)),
        'f1': float(f1_score(y_tst, y_pred, average='binary')),
        'precision': float(precision_score(y_tst, y_pred, zero_division=0)),
        'recall': float(recall_score(y_tst, y_pred)),
        'f1_weighted': float(f1_score(y_tst, y_pred, average='weighted')),
        'f1_macro': float(f1_score(y_tst, y_pred, average='macro')),
        'roc_auc': float(roc_auc_score(y_tst, y_score)) if task_is_biclassif else float('nan'),
        'aucpr': float(average_precision_score(y_tst, y_score)) if task_is_biclassif else float('nan'),
        'mcc': float(matthews_corrcoef(y_tst, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y_tst, y_pred)),
    }
    return ft_score, ft_score_rank, perf
