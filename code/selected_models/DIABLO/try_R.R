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
    loadings_X <- fit$loadings[-fit$indY]
    w_list <- lapply(loadings_X, function(L) {
      L <- as.matrix(L[, seq_len(ncomp), drop = FALSE])       
      v <- if (ncol(L) == 1L) abs(L[, 1]) else apply(abs(L), 1, max)
      stats::setNames(v, rownames(L))                         
    })
    w_vec <- unlist(w_list, use.names = FALSE)
    names(w_vec) <- unlist(lapply(w_list, names), use.names = FALSE)
    w_vec
  }

  all_features <- unique(unlist(lapply(data_filtered, colnames)))
  ft_score <- data.frame(optimal = numeric(length(all_features)),
                         all     = numeric(length(all_features)))
  rownames(ft_score) <- all_features

  grid_vec <- c(5:9, seq(10, 18, 2), seq(20, 30, 5))
  test.keepX <- replicate(length(data_filtered), grid_vec, simplify = FALSE)
  names(test.keepX) <- names(data_filtered)  

  set.seed(1)
  tune.TCGA <- mixOmics::tune.block.splsda(
    X = data_filtered, Y = Y,
    ncomp = ncomp,
    test.keepX = test.keepX,
    design = "full",
    validation = "Mfold", folds = 10, nrepeat = 1,
    dist = "centroids.dist",
    progressBar = FALSE
  )
  list.keepX <- tune.TCGA$choice.keepX  

  opt_model <- mixOmics::block.splsda(
    X = data_filtered, Y = Y,
    ncomp = ncomp,
    design = "full",
    keepX  = list.keepX
  )

  w_opt <- pooled_abs_loadings(opt_model, ncomp)
  ft_score[ names(w_opt), "optimal" ] <- w_opt   

  pred_opt <- predict(opt_model, newdata = data_tst_filtered, dist = "all")
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

  full_model <- mixOmics::block.splsda(
    X = data_filtered, Y = Y,
    ncomp = ncomp,
    design = "full"  
  )

  w_all <- pooled_abs_loadings(full_model, ncomp)
  ft_score[ names(w_all), "all" ] <- w_all

  return(list(
    ft_score = ft_score,
    y_pred   = as.character(Y_pred),
    scores   = scores,
    classes  = levels(Y)
  ))
}