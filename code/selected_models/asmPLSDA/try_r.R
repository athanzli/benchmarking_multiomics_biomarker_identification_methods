library(asmbPLS)

asmbPLSDA_binary_train_test <- function(
    X1, X2, X3,  # column names for features should have already had prefix such as mRNA@
    Y, # ensure Y has been encoced into integer 0 and 1
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