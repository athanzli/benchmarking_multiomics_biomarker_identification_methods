#%%
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
P2G = pd.read_csv("../../../data/TCGA/TCGA_protein2gene_mapping.csv", index_col=0)
C2G = pd.read_csv("../../../data/TCGA/TCGA_cpg2gene_mapping.csv", index_col=0)

"""intersim external variables:

cov.M:
    - Used to determine the number of CpG sites (via ncol(cov.M)) and as the default covariance matrix for methylation.
    - save as csv dataframe with row and column names being cpg ids, and values being the covariance.
    - computed on the DNA methylation data after transforming the beta values to M-values
    - #cpgs x #cpgs

mean.M: 
    - Provides the baseline methylation levels; its names are used to label rows of the DMP matrix and it contributes to the methylation “effect.”
    - length of #cpgs

cov.expr:
    - Used to determine the number of genes (via ncol(cov.expr)) and as the default covariance matrix for gene expression.
    - #genes x #genes

cor.methyl.expr: 
    - Used as the default correlation between methylation and expression when cor.methyl.expr is not supplied.
    - a vector of Pearson correlation coefficients computed between the gene-level methylation measures and the corresponding gene expression values.
    - the CpG probes that map to gene were grouped together and the median values of each group of CpGs beta values were computed. Then M-values were computed using logit transformation. The Pearson correlation coefficients between the M-value and mRNA gene expression for each gene were then computed
    - length of #genes

CpG.gene.map.for.DEG:
    - A mapping object that links CpG sites to genes; used to determine which genes are affected (DEG) when p.DEG is not provided.
    - #cpgs x 2

mean.expr:
    - Provides the baseline gene expression levels; its names are used for row labeling and it is used in computing the gene expression“effect.
    - length of #genes

methyl.gene.level.mean:
    - Used in combination with mean.expr to compute the gene expression effect.
    - 
    - length of #genes

cov.protein:
    - Used to determine the number of proteins (via ncol(cov.protein)) and as the default covariance matrix for protein expression.
    - #proteins x #proteins

cor.expr.protein:
    - Used as the default correlation between expression and protein when cor.expr.protein is not supplied.
    - length of #proteins

mean.protein:
    - Provides the baseline protein expression levels; its names are used for row labeling and it factors into the protein effect.
    - length of #proteins

protein.gene.map.for.DEP:
    - A mapping object that links genes to proteins; used to determine which proteins are affected (DEP) when p.DEP is not provided.
    - #proteins x 2

mean.expr.with.mapped.protein:
    - #proteins

"""

tasks = [
    "survival_BRCA",
]

#%%
if __name__ == "__main__":

    for task in tasks:
        ref = task + "_CNV+DNAm+SNV+mRNA+miRNA+protein"

        print("Generating reference data statistics for task:", task)

        with open(f"../../../data/TCGA/{task}/{ref}.pkl", 'rb') as f:
            data = pd.read_pickle(f)

        X = data['X']

        X = X.loc[:, X.columns.str.split("@").str[0].isin(['mRNA','protein','DNAm'])]

        dnam = X.loc[:, X.columns.str.split("@").str[0]=='DNAm']
        dnam.columns = dnam.columns.str.split("@").str[1]
        mrna = X.loc[:, X.columns.str.split("@").str[0]=='mRNA']
        mrna.columns = mrna.columns.str.split("@").str[1]
        prot = X.loc[:, X.columns.str.split("@").str[0]=='protein']
        prot.columns = prot.columns.str.split("@").str[1]

        ## filter by variance
        dnam = dnam.loc[:, dnam.var(axis=0) > 0.025] # 0.01--> 18k features, 27 hours to generate one dataset; 0.02 --> 10k for several tasks, one day

        # sort by columns
        dnam = dnam.reindex(sorted(dnam.columns), axis=1)
        mrna = mrna.reindex(sorted(mrna.columns), axis=1)
        prot = prot.reindex(sorted(prot.columns), axis=1)

        ######
        # feature filtering and mapping
        mrna = mrna.loc[:, np.intersect1d(C2G.loc[dnam.columns, 'gene'].unique(), mrna.columns.values)]
        dnam = dnam.loc[:, dnam.columns.isin(C2G.loc[C2G['gene'].isin(mrna.columns), 'cpg.1'].unique())]
        c2g = C2G.loc[(C2G['gene'].isin(mrna.columns)) & (C2G['cpg.1'].isin(dnam.columns))].drop(columns=['chr']).drop_duplicates().rename(columns={'cpg.1':'tmp.cg', 'gene':'tmp.gene'})
        # validation that each cpg can be mapped to a gene and each gene can be mapped to a cpg through c2g
        assert dnam.columns.isin(c2g.index).all()
        assert mrna.columns.isin(c2g['tmp.gene']).all()
        
        mrna_for_prot = mrna.loc[:, mrna.columns.isin(P2G.loc[prot.columns, 'gene'].unique())]
        prot = prot.loc[:, prot.columns.isin(P2G.loc[P2G['gene'].isin(mrna_for_prot.columns), 'AGID.1'].unique())]
        p2g = P2G.loc[(P2G['gene'].isin(mrna_for_prot.columns)) & (P2G['AGID.1'].isin(prot.columns))].drop(columns=['peptide_target']).drop_duplicates().rename(columns={'AGID.1':'protein'})
        # since multiple proteins can be mapped to 
        # validation that each protein can be mapped to a gene and each gene can be mapped to a protein through p2g
        assert prot.columns.isin(P2G.loc[P2G['gene'].isin(mrna_for_prot.columns), 'AGID.1']).all()
        assert mrna_for_prot.columns.isin(P2G.loc[prot.columns, 'gene']).all()
        c2g.index = np.arange(c2g.shape[0])
        p2g.index = np.arange(p2g.shape[0])
        CpG_gene_map_for_DEG = c2g
        protein_gene_map_for_DEP = p2g
        print("Number of mRNA features:", mrna.shape[1])
        print("Number of protein features:", prot.shape[1])
        print("Number of DNA methylation features:", dnam.shape[1])

        # convert beta-values to M-values
        eps = 1e-8
        dnam = dnam.clip(eps, 1 - eps)
        methylation_M = np.log(dnam / (1 - dnam))

        # covariance computation
        # cov_M = np.cov(methylation_M, rowvar=False)   # CpG sites x CpG sites
        # cov_expr = np.cov(mrna, rowvar=False)      # Genes x Genes
        cov_protein = np.cov(prot, rowvar=False)    # Proteins x Proteins
        # cov_M = pd.DataFrame(cov_M, index=methylation_M.columns, columns=methylation_M.columns)
        # cov_expr = pd.DataFrame(cov_expr, index=mrna.columns, columns=mrna.columns)
        cov_protein = pd.DataFrame(cov_protein, index=prot.columns, columns=prot.columns)

        # mean.M, mean.expr and mean.prot
        mean_M = methylation_M.mean(axis=0).to_frame().sort_index(inplace=False)
        mean_expr = mrna.mean(axis=0).to_frame().sort_index(inplace=False)
        mean_expr_with_mapped_protein = mrna_for_prot.mean(axis=0).to_frame().sort_index(inplace=False)
        mean_protein = prot.mean(axis=0).to_frame().sort_index(inplace=False)

        # methyl.gene.level.mean
        M_grouped_by_gene = CpG_gene_map_for_DEG.groupby('tmp.gene').apply(lambda x: methylation_M.loc[:, x['tmp.cg']].median(axis=1)) # paper: " the CpG probes that map to gene were grouped together and the median values of each group of CpGs were computed. Then M-values were computed using logit transformation. The Pearson correlation coefficients between the M-value and mRNA gene expression for each gene were then computed"
        M_grouped_by_gene = M_grouped_by_gene.T.loc[mrna.index, mrna.columns]
        methyl_gene_level_mean = M_grouped_by_gene.mean(axis=0).to_frame()
        methyl_gene_level_mean.sort_index(inplace=True)

        # cor_methyl_expr
        cor_methyl_expr = pd.Series(index=mrna.columns)
        for gene in mrna.columns:
            cor_methyl_expr[gene] = pearsonr(M_grouped_by_gene[gene], mrna[gene])[0]
        cor_methyl_expr = cor_methyl_expr.to_frame()
        cor_methyl_expr.sort_index(inplace=True)

        # gene_grouped_by_protein
        gene_grouped_by_protein = protein_gene_map_for_DEP.groupby('protein').apply(lambda x: mrna_for_prot.loc[:, x['gene']].mean(axis=1))
        gene_grouped_by_protein = gene_grouped_by_protein.T.loc[prot.index, prot.columns]
        mean_expr_with_mapped_protein = gene_grouped_by_protein.mean(axis=0).to_frame()
        mean_expr_with_mapped_protein.sort_index(inplace=True)

        # cor_expr_protein
        cor_expr_protein = pd.Series(index=prot.columns)
        for protein_mol in prot.columns:
            cor_expr_protein[protein_mol] = pearsonr(prot[protein_mol], gene_grouped_by_protein[protein_mol])[0]
        cor_expr_protein = cor_expr_protein.to_frame()
        cor_expr_protein.sort_index(inplace=True)

        ### check
        #
        assert (np.sort(cov_protein.columns)==cov_protein.columns).all()
        assert (np.sort(cov_protein.index)==cov_protein.index).all()
        #
        assert (np.sort(cor_expr_protein.index)==cor_expr_protein.index).all()
        #
        assert (np.sort(cor_methyl_expr.index)==cor_methyl_expr.index).all()
        #
        assert (np.sort(mean_M.index)==mean_M.index).all()
        #
        assert (np.sort(mean_protein.index==mean_protein.index).all())
        #
        assert (np.sort(mean_expr.index)==mean_expr.index).all()
        #
        assert(np.sort(methyl_gene_level_mean.index==methyl_gene_level_mean.index).all())
        #
        assert (np.sort(mean_expr_with_mapped_protein.index==mean_expr_with_mapped_protein.index).all())
        assert (cov_protein.index==prot.columns).all()
        assert (cov_protein.columns==prot.columns).all()
        assert (cov_protein.columns==mean_protein.index).all()
        # 
        assert (cor_methyl_expr.index==mrna.columns).all()
        #
        assert (cor_expr_protein.index==prot.columns).all()
        #
        assert (methyl_gene_level_mean.index==mean_expr.index).all()
        #
        assert (mean_expr_with_mapped_protein.index==mean_protein.index).all()

        ###########################################################################
        # save
        ###########################################################################
        print("Saving files...")
        # sigma.protein
        cov_protein.to_csv(f"../../../data/synthetic/InterSIM/ref_values/sigma.protein_{task}.csv")
        # cor.methyl.expr
        cor_methyl_expr.to_csv(f"../../../data/synthetic/InterSIM/ref_values/cor.methyl.expr_{task}.csv")
        # cor.expr.protein
        cor_expr_protein.to_csv(f"../../../data/synthetic/InterSIM/ref_values/cor.expr.protein_{task}.csv")
        # mean.M:
        mean_M.to_csv(f"../../../data/synthetic/InterSIM/ref_values/mean.M_{task}.csv")
        # CpG.gene.map.for.DEG:
        CpG_gene_map_for_DEG.to_csv(f"../../../data/synthetic/InterSIM/ref_values/CpG.gene.map.for.DEG_{task}.csv")
        # mean.expr:
        mean_expr.to_csv(f"../../../data/synthetic/InterSIM/ref_values/mean.expr_{task}.csv")
        # methyl.gene.level.mean:
        methyl_gene_level_mean.to_csv(f"../../../data/synthetic/InterSIM/ref_values/methyl.gene.level.mean_{task}.csv")
        # mean.protein:
        mean_protein.to_csv(f"../../../data/synthetic/InterSIM/ref_values/mean.protein_{task}.csv")
        # protein.gene.map.for.DEP:
        protein_gene_map_for_DEP.to_csv(f"../../../data/synthetic/InterSIM/ref_values/protein.gene.map.for.DEP_{task}.csv")
        # mean.expr.with.mapped.protein:
        mean_expr_with_mapped_protein.to_csv(f"../../../data/synthetic/InterSIM/ref_values/mean.expr.with.mapped.protein_{task}.csv")
        # ## too large. compute using cov in R on the fly instead.
        # # sigma.expr
        # cov_expr.to_csv(f"../../data/synthetic/InterSIM/ref_values/sigma.expr_{task}.csv")
        # # sigma.methyl
        # cov_M.to_csv(f"../../data/synthetic/InterSIM/ref_values/sigma.methyl_{task}.csv")
        # ## load pickled ref data in R, and perform the following steps:
        # with open(f"../../data/TCGA/{task}/{ref}.pkl", 'rb') as f:
        #     data = pd.read_pickle(f)
        # X = data['X']
        # X = X.loc[:, X.columns.str.split("@").str[0].isin(['mRNA','protein','DNAm'])]
        # dnam = X.loc[:, X.columns.str.split("@").str[0]=='DNAm']
        # dnam.columns = dnam.columns.str.split("@").str[1]
        # mrna = X.loc[:, X.columns.str.split("@").str[0]=='mRNA']
        # mrna.columns = mrna.columns.str.split("@").str[1]
        # P2G = pd.read_csv("../../data/TCGA/TCGA_protein2gene_mapping.csv", index_col=0)
        # C2G = pd.read_csv("../../data/TCGA/TCGA_cpg2gene_mapping.csv", index_col=0)
        # mrna = mrna.loc[:, np.intersect1d(C2G.loc[dnam.columns, 'gene'].unique(), mrna.columns.values)]
        # dnam = dnam.loc[:, dnam.columns.isin(C2G.loc[C2G['gene'].isin(mrna.columns), 'cpg.1'].unique())]
        # eps = 1e-8
        # dnam = dnam.clip(eps, 1 - eps)
        # methylation_M = np.log(dnam / (1 - dnam))
        # cov_M = np.cov(methylation_M, rowvar=False) # sigma.methyl
        # cov_expr = np.cov(mrna, rowvar=False) #sigma.expr

