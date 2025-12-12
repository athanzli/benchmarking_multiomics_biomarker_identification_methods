options(scipen = 999)
library(splitstackshape)
library(InterSIM)
library(reticulate)

use_python("/home/athan.li/miniconda3/envs/eval_bk/bin/python3")
setwd("/home/athan.li/eval_bk/code/synthetic_data_generation/InterSIM/")

###############################################################################
# InterSIM
###############################################################################
# NOTE change accordingly
PATH_SAVE <- "/home/athan.li/eval_bk/data/synthetic/InterSIM/"

intersim <- function (
    n.sample = 500,
    cluster.sample.prop = NULL,
    n.sample.in.cluster = NULL,
    delta.methyl = 2,
    delta.expr = 2,
    delta.protein = 2,
    p.DMP = 0.01,
    p.DEG = NULL,
    p.DEG4P=NULL,
    p.DEP = NULL,
    sigma.methyl = NULL, 
    sigma.expr = NULL,
    sigma.protein = NULL,
    cor.methyl.expr = NULL, 
    cor.expr.protein = NULL,
    do.plot = FALSE,
    sample.cluster = TRUE, 
    feature.cluster = TRUE)
{
  print("Length of mean.expr:")
  print(length(mean.expr))
  
  if (p.DMP < 0 | p.DMP > 1) 
    stop("p.DMP must be between 0 to 1")
  if (!is.null(p.DEG) && (p.DEG < 0 | p.DEG > 1)) 
    stop("p.DEG must be between 0 and 1")
  if (!is.null(p.DEP) && (p.DEP < 0 | p.DEP > 1)) 
    stop("p.DEP must be between 0 and 1")
  
  if (!is.null(n.sample.in.cluster)) {
    if (sum(n.sample.in.cluster) != n.sample) 
        stop("The sum of n.sample.in.cluster must be equal to n.sample")
    n.cluster <- length(n.sample.in.cluster)
  }
  else{ # prop must be provided.
    if (sum(cluster.sample.prop) != 1) 
        stop("The proportions must sum up to 1")
    if (!length(cluster.sample.prop) > 1) 
        stop("Number of proportions must be larger than 1")
    n.cluster <- length(cluster.sample.prop)
    # TODO currenty only for two clusters?
    n.sample.in.cluster <- c(round(cluster.sample.prop[-n.cluster] * 
                                    n.sample), n.sample - sum(round(cluster.sample.prop[-n.cluster] * 
                                                                        n.sample)))
  }

  # NOTE original code, which raises error when n.sample is the same in each cluster
  # cluster.id <- do.call(c, sapply(1:n.cluster, function(x) rep(x, 
  #                                                              n.sample.in.cluster[x])))
  # modified code that avoids the default inconsistent conversion
  cluster.id <- rep(1:n.cluster, times = n.sample.in.cluster)
  
  n.CpG <- ncol(cov.M)
  cov.str <- sigma.methyl # NOTE must ensure sigma.methyl is not null
  # NOTE this creates a n.CpG by n.cluster matrix, where each column is i.i.d. from rbinom(n.CpG, 1, p.DMP) sampling.
  # original code
  # DMP <- sapply(1:n.cluster, function(x) rbinom(n.CpG, 1, prob = p.DMP))
  # modified code to ensure at least one cpg is chosen
  print("Sampling DMPs...")
  DMP <- sapply(1:n.cluster, function(x) rbinom(n.CpG, 1, prob = p.DMP))
  while (any(colSums(DMP)==0)) {
    DMP <- sapply(1:n.cluster, function(x) rbinom(n.CpG, 1, prob = p.DMP))
  }
  print("Simulating methylation data...")
  start_time <- Sys.time()
  rownames(DMP) <- names(mean.M)
  d <- lapply(1:n.cluster, function(i) {
    effect <- mean.M + DMP[, i] * delta.methyl
    mvrnorm(n = n.sample.in.cluster[i], mu = effect, Sigma = cov.str)
  })
  end_time <- Sys.time()
  elapsed_time <- difftime(end_time, start_time, units = "secs")
  print(paste("Sim methy time:", round(elapsed_time, 2), "seconds"))

  sim.methyl <- do.call(rbind, d)
  sim.methyl <- rev.logit(sim.methyl) # conversion from M to beta value.
  n.gene <- ncol(cov.expr)
  cov.str <- sigma.expr # NOTE must ensure sigma.expr is not null
  rho.m.e <- cor.methyl.expr # NOTE must ensure cor.methyl.expr is not null
  print("Sampling DEGs...")
  DEG <- sapply(1:n.cluster, function(x) { # NOTE p.DEG must be NULL
      cg.name <- rownames(subset(DMP, DMP[, x] == 1))
  #   gene.name <- as.character(CpG.gene.map.for.DEG[cg.name, ]$tmp.gene) # NOTE
      gene.name <- as.character(CpG.gene.map.for.DEG[CpG.gene.map.for.DEG$tmp.cg %in% cg.name, ]$tmp.gene)
      as.numeric(names(mean.expr) %in% gene.name)
  })
  rownames(DEG) <- names(mean.expr)
  
  #############################################################################
  # add DEGs that can map to protein using the provided mapping (in case n_genes >> n_proteins)
  # this ensures there is at least one DEP.
  #############################################################################
  if (!is.null(p.DEG4P)) {
    pgene.names <- unique(protein.gene.map.for.DEP$gene)
    n.pgene <- length(pgene.names)
    DEG2 <- sapply(1:n.cluster, function(x) rbinom(n.pgene, 1, prob = p.DEG4P))
    while (any(colSums(DEG2)==0)) {
      DEG2 <- sapply(1:n.cluster, function(x) rbinom(n.pgene, 1, prob = p.DEG4P))
    }
    gene.names <- rownames(DEG)
    DEG[gene.names %in% pgene.names, ] <- DEG2  # set the corresponding positions of DEG to DEG2
  }
  #############################################################################
  
  print("Simulating gene expression data...")
  start_time <- Sys.time()
  if (delta.expr == 0) 
    rho.m.e <- 0
  d <- lapply(1:n.cluster, function(i) { # generate gene expression from methylation data.
    effect <- (rho.m.e * methyl.gene.level.mean + sqrt(1 - 
                                                         rho.m.e^2) * mean.expr) + DEG[, i] * delta.expr
    mvrnorm(n = n.sample.in.cluster[i], mu = effect, Sigma = cov.str)
  })
  end_time <- Sys.time()
  elapsed_time <- difftime(end_time, start_time, units = "secs")
  print(paste("Sim gxp time:", round(elapsed_time, 2), "seconds"))
  sim.expr <- do.call(rbind, d)
  n.protein <- ncol(cov.protein)
  cov.str <- sigma.protein # NOTE must ensure sigma.protein is not null!
  rho.e.p <- cor.expr.protein # NOTE must ensure cor.expr.protein is not null!
  DEP <- sapply(1:n.cluster, function(x) { # p.DEP must be NULL
      gene.name <- rownames(subset(DEG, DEG[, x] == 1))
      protein.name <- protein.gene.map.for.DEP[protein.gene.map.for.DEP$gene %in% 
                                                          gene.name, ]$protein
      as.numeric(names(mean.protein) %in% protein.name)
  })
  rownames(DEP) <- names(mean.protein)
  print("Simulating protein expression data...")
  if (delta.protein == 0) 
    rho.e.p <- 0
  d <- lapply(1:n.cluster, function(i) { # generate protein from mRNA expression
    effect <- (rho.e.p * mean.expr.with.mapped.protein + 
                 sqrt(1 - rho.e.p^2) * mean.protein) + DEP[, i] * 
      delta.protein
    mvrnorm(n = n.sample.in.cluster[i], mu = effect, Sigma = cov.str)
  })
  print("Finished simulating protein expression data.")
  sim.protein <- do.call(rbind, d)
  indices <- sample(1:n.sample)
  cluster.id <- cluster.id[indices]
  sim.methyl <- sim.methyl[indices, ]
  sim.expr <- sim.expr[indices, ]
  sim.protein <- sim.protein[indices, ]

  print('POS 4') #
  print(length(cluster.id))
  print(length(rownames(sim.methyl)))
  print(length(rownames(sim.expr)))
  print(length(rownames(sim.protein)))
  rownames(sim.methyl) <- paste("subject", 1:nrow(sim.methyl), 
                                sep = "")
  rownames(sim.expr) <- paste("subject", 1:nrow(sim.expr), 
                              sep = "")
  rownames(sim.protein) <- paste("subject", 1:nrow(sim.protein), 
                                 sep = "")
  d.cluster <- data.frame(rownames(sim.methyl), cluster.id)
  colnames(d.cluster)[1] <- "subjects"
  
  if (do.plot) {
    hmcol <- colorRampPalette(c("blue", "deepskyblue", "white", 
                                "orangered", "red3"))(100)
    if (dev.interactive()) 
      dev.off()
    if (sample.cluster && feature.cluster) {
      dev.new(width = 15, height = 5)
      par(mfrow = c(1, 3))
      aheatmap(t(sim.methyl), color = hmcol, Rowv = FALSE, 
               Colv = FALSE, labRow = NA, labCol = NA, annLegend = T, 
               main = "Methylation", fontsize = 10, breaks = 0.5)
      aheatmap(t(sim.expr), color = hmcol, Rowv = FALSE, 
               Colv = FALSE, labRow = NA, labCol = NA, annLegend = T, 
               main = "Gene expression", fontsize = 10, breaks = 0.5)
      aheatmap(t(sim.protein), color = hmcol, Rowv = FALSE, 
               Colv = FALSE, labRow = NA, labCol = NA, annLegend = T, 
               main = "Protein expression", fontsize = 10, 
               breaks = 0.5)
    }
    else if (sample.cluster) {
      dev.new(width = 15, height = 5)
      par(mfrow = c(1, 3))
      aheatmap(t(sim.methyl), color = hmcol, Rowv = NA, 
               Colv = FALSE, labRow = NA, labCol = NA, annLegend = T, 
               main = "Methylation", fontsize = 8, breaks = 0.5)
      aheatmap(t(sim.expr), color = hmcol, Rowv = NA, 
               Colv = FALSE, labRow = NA, labCol = NA, annLegend = T, 
               main = "Gene expression", fontsize = 8, breaks = 0.5)
      aheatmap(t(sim.protein), color = hmcol, Rowv = NA, 
               Colv = FALSE, labRow = NA, labCol = NA, annLegend = T, 
               main = "Protein expression", fontsize = 8, breaks = 0.5)
    }
    else if (feature.cluster) {
      dev.new(width = 15, height = 5)
      par(mfrow = c(1, 3))
      aheatmap(t(sim.methyl), color = hmcol, Rowv = FALSE, 
               Colv = NA, labRow = NA, labCol = NA, annLegend = T, 
               main = "Methylation", fontsize = 8, breaks = 0.5)
      aheatmap(t(sim.expr), color = hmcol, Rowv = FALSE, 
               Colv = NA, labRow = NA, labCol = NA, annLegend = T, 
               main = "Gene expression", fontsize = 8, breaks = 0.5)
      aheatmap(t(sim.protein), color = hmcol, Rowv = FALSE, 
               Colv = NA, labRow = NA, labCol = NA, annLegend = T, 
               main = "Protein expression", fontsize = 8, breaks = 0.5)
    }
    else {
      dev.new(width = 15, height = 5)
      par(mfrow = c(1, 3))
      aheatmap(t(sim.methyl), color = hmcol, Rowv = NA, 
               Colv = NA, labRow = NA, labCol = NA, annLegend = T, 
               main = "Methylation", fontsize = 8, breaks = 0.5)
      aheatmap(t(sim.expr), color = hmcol, Rowv = NA, 
               Colv = NA, labRow = NA, labCol = NA, annLegend = T, 
               main = "Gene expression", fontsize = 8, breaks = 0.5)
      aheatmap(t(sim.protein), color = hmcol, Rowv = NA, 
               Colv = NA, labRow = NA, labCol = NA, annLegend = T, 
               main = "Protein expression", fontsize = 8, breaks = 0.5)
    }
  }
  return(list(dat.methyl = sim.methyl, dat.expr = sim.expr, 
              dat.protein = sim.protein, clustering.assignment = d.cluster,
              dat.DMP = DMP, dat.DEG = DEG, dat.DEP = DEP))
}

################################################################################
#
################################################################################

run_intersim <- function (
    prop, n.sample.in.cluster, effect, p.DMP, p.DEG4P, n.sample, sigma.methyl, sigma.expr,
    sigma.protein, cor.methyl.expr, cor.expr.protein, task) {
  sim.data <- intersim(n.sample=n.sample, n.sample.in.cluster=n.sample.in.cluster, cluster.sample.prop = prop,
                       delta.methyl=effect, delta.expr=effect, delta.protein=effect,
                       p.DMP=p.DMP, p.DEG4P=p.DEG4P,
                       sigma.methyl=sigma.methyl, sigma.expr=sigma.expr, sigma.protein=sigma.protein,
                       cor.methyl.expr=cor.methyl.expr,
                       cor.expr.protein=cor.expr.protein,
                       do.plot=FALSE, sample.cluster=FALSE, feature.cluster=FALSE)
  # y
  y <- sim.data$clustering.assignment
  y_vector <- y[[2]]
  label <- data.frame(label = y_vector, stringsAsFactors = FALSE, row.names = y[[1]])
  colnames(label) <- "label"
  
  # data
  prefix_DNAm <- "DNAm@"
  prefix_mRNA <- "mRNA@"
  prefix_protein <- "protein@"
  prefixed_methyl <- paste0(prefix_DNAm, colnames(sim.data$dat.methyl))
  prefixed_expr <- paste0(prefix_mRNA, colnames(sim.data$dat.expr))
  prefixed_protein <- paste0(prefix_protein, colnames(sim.data$dat.protein))
  all_prefixed_names <- c(prefixed_methyl, prefixed_expr, prefixed_protein)
  
  sim.methyl <- sim.data$dat.methyl
  sim.expr <- sim.data$dat.expr
  sim.protein <- sim.data$dat.protein
  data <- cbind(sim.methyl, sim.expr, sim.protein)
  
  colnames(data) <- all_prefixed_names
  
  # confirm row names order
  stopifnot(all(rownames(data) == rownames(label)))
  
  # bk
  bk.methyl <- sim.data$dat.DMP
  bk.expr <- sim.data$dat.DEG
  bk.protein <- sim.data$dat.DEP
  
  n_class <- length(n.sample.in.cluster) # NOTE ensure n.sample.in.cluster is not null
  bk_res <- data.frame(matrix(FALSE, 
                              nrow = length(all_prefixed_names),
                              ncol = n_class),
                       stringsAsFactors = FALSE)
  rownames(bk_res) <- all_prefixed_names
  colnames(bk_res) <- as.character(1:n_class)
  
  ##############################################################################
  # each cluster has its own DE molecules. p.DEx is per cluster.
  ##############################################################################
  for (i in 1:n_class) {
    dnams_with_one <- paste0(prefix_DNAm, rownames(bk.methyl)[bk.methyl[, i] == 1])
    stopifnot(length(dnams_with_one) >= 1) # NOTE in case rbinom didn't sample any point
    mrnas_with_one <- paste0(prefix_mRNA, rownames(bk.expr)[bk.expr[, i] == 1])
    proteins_with_one <- paste0(prefix_protein, rownames(bk.protein)[bk.protein[, i] == 1])
    biomarkers_with_one <- c(dnams_with_one, mrnas_with_one, proteins_with_one)
    bk_res[biomarkers_with_one, as.character(i)] <- TRUE
  }
  
  # check
  stopifnot(all(prop.table(table(label))==prop))
  
  # save
  prefix <- paste0("InterSIM_ref=", task, "_")
  param_setting <- paste0("n=", n.sample, "_p.dmp=", p.DMP, "_p.deg4p=", p.DEG4P, "_shift=", effect, "_")
  write.csv(bk_res, paste0(PATH_SAVE, prefix, param_setting, "bk.csv"))
  write.csv(data, paste0(PATH_SAVE, prefix, param_setting, "data.csv"))
  write.csv(label, file = paste0(PATH_SAVE, prefix, param_setting, "label.csv"), row.names = TRUE)
}

################################################################################
# prep
################################################################################

tasks <- c(
  "survival_BRCA"
)

for (task in tasks) {
  print(task)

  pd <- import("pandas")
  data <- py_load_object(sprintf("../../../data/TCGA/%s/%s.pkl", task, paste0(task, '_CNV+DNAm+SNV+mRNA+miRNA+protein')))

  X <- data$X
  name_parts <- sapply(strsplit(colnames(X), "@"), function(x) x[1])
  # keep only columns where the first part is in 'mRNA', 'protein', or 'DNAm'
  X <- X[, name_parts %in% c("mRNA", "protein", "DNAm"), drop = FALSE]

  name_parts <- sapply(strsplit(colnames(X), "@"), function(x) x[1]) # Mods

  dnam <- X[, name_parts == "DNAm", drop = FALSE]
  colnames(dnam) <- sapply(strsplit(colnames(dnam), "@"), function(x) x[2])

  mrna <- X[, name_parts == "mRNA", drop = FALSE]
  colnames(mrna) <- sapply(strsplit(colnames(mrna), "@"), function(x) x[2])

  P2G <- read.csv("../../../data/TCGA/TCGA_protein2gene_mapping.csv", stringsAsFactors = FALSE)
  C2G <- read.csv("../../../data/TCGA/TCGA_cpg2gene_mapping.csv", stringsAsFactors = FALSE)

  genes_from_dnam <- unique(C2G$gene[C2G$`cpg.1` %in% colnames(dnam)])
  common_genes <- intersect(genes_from_dnam, colnames(mrna))
  mrna <- mrna[, common_genes, drop = FALSE]
  cpg_set <- unique(C2G$`cpg.1`[C2G$gene %in% common_genes])
  dnam <- dnam[, colnames(dnam) %in% cpg_set, drop = FALSE]

  ## load external variables
  df <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/cor.expr.protein_", task, ".csv"), header = TRUE, row.names = 1)
  cor.expr.protein <- setNames(as.numeric(df[, 1]), rownames(df))
  df <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/cor.methyl.expr_", task, ".csv"), header = TRUE, row.names = 1)
  cor.methyl.expr <- setNames(as.numeric(df[, 1]), rownames(df))
  df <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/mean.expr_", task, ".csv"), header = TRUE, row.names = 1)
  mean.expr <- setNames(as.numeric(df[, 1]), rownames(df))
  df <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/mean.expr.with.mapped.protein_", task, ".csv"), header = TRUE, row.names = 1)
  mean.expr.with.mapped.protein <- setNames(as.numeric(df[, 1]), rownames(df))
  df <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/mean.M_", task, ".csv"), header = TRUE, row.names = 1)
  mean.M <- setNames(as.numeric(df[, 1]), rownames(df))
  df <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/mean.protein_", task, ".csv"), header = TRUE, row.names = 1)
  mean.protein <- setNames(as.numeric(df[, 1]), rownames(df))
  df <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/methyl.gene.level.mean_", task, ".csv"), header = TRUE, row.names = 1)
  methyl.gene.level.mean <- setNames(as.numeric(df[, 1]), rownames(df))

  CpG.gene.map.for.DEG <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/CpG.gene.map.for.DEG_", task, ".csv"), header = TRUE)
  protein.gene.map.for.DEP <- read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/protein.gene.map.for.DEP_", task, ".csv"), header = TRUE)

  # align features
  dnam <- dnam[, names(mean.M)]
  mrna <- mrna[, names(mean.expr)]

  eps <- 1e-8
  dnam[dnam == 0] <- dnam[dnam == 0] + eps
  dnam[dnam == 1] <- dnam[dnam == 1] - eps
  methylation_M = log2(dnam / (1-dnam))

  ## compute cov
  np <- import("numpy")
  methylation_M_matrix <- as.matrix(methylation_M)
  cov.M <- np$cov(methylation_M_matrix, rowvar = FALSE)
  cov.M <- data.frame(cov.M, row.names = colnames(methylation_M))
  colnames(cov.M) <- colnames(methylation_M)
  mrna_matrix <- as.matrix(mrna)
  cov.expr <- np$cov(mrna_matrix, rowvar = FALSE)
  cov.expr <- data.frame(cov.expr, row.names = colnames(mrna))
  colnames(cov.expr) <- colnames(mrna)

  sigma.methyl <- as.matrix(cov.M)
  sigma.expr <- as.matrix(cov.expr)
  sigma.protein <- as.matrix(read.csv(paste0("/home/athan.li/eval_bk/data/synthetic/InterSIM/ref_values/sigma.protein_", task, ".csv"), header = TRUE, row.names = 1))
  cov.protein <- sigma.protein

  ################################################################################
  # 
  ################################################################################
  if (task == 'BRCA_Subtype_PAM50') {
    n.sample.in.cluster <- c(300, 300, 300, 300, 300)
    n.sample <- 1500
  } else {
    n.sample.in.cluster <- c(50, 50)
    n.sample <- 100
  }
  run_intersim(n.sample = 100, prop = NULL, n.sample.in.cluster = c(50,50), effect = 0.5, p.DMP = 0.01, p.DEG4P=0.1, sigma.methyl = sigma.methyl, sigma.expr = sigma.expr, sigma.protein = sigma.protein, cor.methyl.expr = cor.methyl.expr, cor.expr.protein = cor.expr.protein, task=task)
  run_intersim(n.sample = 100, prop = NULL, n.sample.in.cluster = c(50,50), effect = 1, p.DMP = 0.01, p.DEG4P=0.1, sigma.methyl = sigma.methyl, sigma.expr = sigma.expr, sigma.protein = sigma.protein, cor.methyl.expr = cor.methyl.expr, cor.expr.protein = cor.expr.protein, task=task)
  run_intersim(n.sample = 100, prop = NULL, n.sample.in.cluster = c(50,50), effect = 2, p.DMP = 0.01, p.DEG4P=0.1, sigma.methyl = sigma.methyl, sigma.expr = sigma.expr, sigma.protein = sigma.protein, cor.methyl.expr = cor.methyl.expr, cor.expr.protein = cor.expr.protein, task=task)
  run_intersim(n.sample = 100, prop = NULL, n.sample.in.cluster = c(50,50), effect = 3, p.DMP = 0.01, p.DEG4P=0.1, sigma.methyl = sigma.methyl, sigma.expr = sigma.expr, sigma.protein = sigma.protein, cor.methyl.expr = cor.methyl.expr, cor.expr.protein = cor.expr.protein, task=task)
  run_intersim(n.sample = 100, prop = NULL, n.sample.in.cluster = c(50,50), effect = 4, p.DMP = 0.01, p.DEG4P=0.1, sigma.methyl = sigma.methyl, sigma.expr = sigma.expr, sigma.protein = sigma.protein, cor.methyl.expr = cor.methyl.expr, cor.expr.protein = cor.expr.protein, task=task)
  run_intersim(n.sample = 100, prop = NULL, n.sample.in.cluster = c(50,50), effect = 5, p.DMP = 0.01, p.DEG4P=0.1, sigma.methyl = sigma.methyl, sigma.expr = sigma.expr, sigma.protein = sigma.protein, cor.methyl.expr = cor.methyl.expr, cor.expr.protein = cor.expr.protein, task=task)
}

