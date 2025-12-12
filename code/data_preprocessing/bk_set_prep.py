import pandas as pd
import sys
import numpy as np

sys.path.append("..")
TCGA_DATA_PATH = "/data/zhaohong/TCGA_data/data/"
DATA_PATH = "../../data/bk_set/"

from naming_conversion_mappings import ONCOKB_CANCER_TYPE_MAPPING, CIVIC_MOLECULAR_PROFILE_MAPPING, CIVIC_DISEASE_MAPPING, CGI_TUMOR_NAMES_MAPPING, GENE_NAME_STD_MAP, map_molecular_profile_to_genes, ALL_TCGA_PROJ_IDS

mrna_genes = pd.read_csv(TCGA_DATA_PATH + 'raw/mRNA_gene_list.csv', index_col=0)

###############################################################################
# general preprocessing
###############################################################################
#%% ################### OncoKB prognostic
d = pd.read_csv(DATA_PATH + "raw/oncokb/oncokb_biomarker_prognostic.tsv", sep='\t') # 
d['TCGA_project'] = d['Cancer Types'].map(ONCOKB_CANCER_TYPE_MAPPING)
d = d.loc[(d['TCGA_project']!='nan') & (d['TCGA_project'].notna()) & (d['TCGA_project']!='None')].drop_duplicates()
d['Level'] = d['Level'].map({
    'Px1': 'A',
    'Px2': 'A',
    'Px3': 'B',
})
d['Evidence statement'] = np.nan
d['Evidence source'] = np.nan
d['Database'] = 'OncoKB'
oncokb_prog = d.copy()

# %% ################### OncoKB therapeutic
d = pd.read_csv(DATA_PATH + "raw/oncokb/oncokb_biomarker_therapeutic.tsv", sep='\t')
print(d['Level'].value_counts())
d['Level'] = d['Level'].map({
   'R1' : 'A',
   'R2' : 'D',
   '1': 'A',
   '2': 'A',
   '3': 'B',
   '4': 'D',
})
d['TCGA_project'] = d['Cancer Types'].map(ONCOKB_CANCER_TYPE_MAPPING)
mask = (d['TCGA_project'].notna()) & (d['Gene'].notna()) & (d['Level'].notna())
d = d.loc[mask]
d = d.explode('TCGA_project').reset_index(drop=True)
d = d.drop_duplicates()
d['Evidence statement'] = np.nan
d['Evidence source'] = np.nan
d['Database'] = 'OncoKB'
oncokb_therapeutic = d.copy()

#%% #################### CIVIC
d = pd.read_csv(DATA_PATH + "raw/civic/01-June-2025-ClinicalEvidenceSummaries.tsv", sep='\t') # https://civicdb.org/releases/main Evidence TSV

## evidence level
d = d.loc[(d['evidence_direction']!='Does Not Support')]
d = d.loc[d['evidence_level'].notna()]
# d['evidence_level'].value_counts()
d['Level'] = d['evidence_level'].map({
    'A': 'A',
    'B': 'B',
    'C': 'C',
    'D': 'D',
    'E': 'D',
})

d['TCGA_project'] = d['disease'].map(CIVIC_DISEASE_MAPPING) # confirmed there is no disease entry containing more than one cancer type
d = d.loc[(d['TCGA_project'] != 'nan') & (d['TCGA_project'].notna()) & (d['TCGA_project'] != 'None')]
d = d.explode('TCGA_project').reset_index(drop=True)

d['Gene'] = d['molecular_profile'].map(CIVIC_MOLECULAR_PROFILE_MAPPING)
d = d.loc[(d['Gene'].notna()) & (d['Gene']!='')]
d['Gene'] = d['Gene'].str.split(r';\s*')
d = d.explode('Gene').reset_index(drop=True) # NOTE
# d['evidence_level'].value_counts()
assert d['Gene'].str.contains(';').sum()==0
assert d['Gene'].str.contains(',').sum()==0

## standardize gene names that are in BK set but not in TCGA mrna gene set # found the following: UGT1A -> UGT1A1
print(set(d['Gene'].unique()) - set(mrna_genes['gene_name'].unique()))
print(GENE_NAME_STD_MAP.loc['UGT1A'])
print(mrna_genes.loc[mrna_genes['gene_name']=='UGT1A1'])
d['Gene'].replace('UGT1A', 'UGT1A1', inplace=True)
d = d.drop_duplicates()
d = d.rename(columns={'evidence_statement': 'Evidence statement',
                'evidence_civic_url': 'Evidence source'})
d['Database'] = 'CIViC'
civic = d.copy()

#%% #################### CGI
d = pd.read_csv(DATA_PATH + "raw/cgi/cgi_biomarkers_latest.tsv", sep='\t') # 

## evidence level
d['Level'] = d['Evidence level'].map({
  'FDA guidelines':'A',
  'NCCN guidelines':'A',
  'NCCN/CAP guidelines':'A',
  'CPIC guidelines':'A',
  'European LeukemiaNet guidelines':'A',

  'Early trials':'C',
  'Case report':'C',
  'Early Trials,Case Report':'C',
  'Clinical trials':'C',

  'Late trials':'B',
  'Pre-clinical':'D',
  'Late trials,Pre-clinical':'D'
})

# expand multiple diseases
mask1 = d['Primary Tumor type'].str.contains(';').values
mask2 = d['Primary Tumor type full name'].str.contains(';').values
assert all(mask1 == mask2)
tmp = d.loc[mask1].copy()
for i in range(tmp.shape[0]):
  assert len(tmp['Primary Tumor type'].str.split(';').values[i])==len(tmp['Primary Tumor type full name'].str.split(';').values[i])
d['Primary Tumor type'] = d['Primary Tumor type'].str.split(';')
d['Primary Tumor type full name'] = d['Primary Tumor type full name'].str.split(';')
d = d.explode(['Primary Tumor type', 'Primary Tumor type full name']).reset_index(drop=True).drop_duplicates()
d['TCGA_project'] = d['Primary Tumor type full name'].map(CGI_TUMOR_NAMES_MAPPING)
d = d.explode('TCGA_project').reset_index(drop=True).drop_duplicates()
assert d['Gene'].str.contains(',').sum() == 0
# d.loc[d['Gene'].str.contains(';')]
d['Gene'] = d['Gene'].str.split(r';\s*')
d = d.explode('Gene').reset_index(drop=True) # NOTE
d = d.drop_duplicates()

## standardize gene names that are in BK set but not in TCGA mrna gene set
print(set(d['Gene'].unique()) - set(mrna_genes['gene_name'].unique()))
print(GENE_NAME_STD_MAP.loc[['MLL2', 'MLL', 'C15orf55']])
print(mrna_genes.loc[mrna_genes['gene_name']=='UGT1A1'])
d['Gene'].replace('MLL', 'KMT2A', inplace=True)
d['Gene'].replace('C15orf55', 'NUTM1', inplace=True)
row = d.loc[d.loc[d['Gene']=='MLL2'].index].copy()
row['Gene'] = 'KMT2B'
row.index = pd.Index([d.index.max()+1], dtype='int64')
d = pd.concat([d, row], axis=0)
d['Gene'].replace('MLL2', 'KMT2D', inplace=True)

d['Evidence statement'] = np.nan
d['Evidence source'] = d['Source']
d['Database'] = 'CGI'

# remove "Increased Toxicity" as it is not related to drug efficacy
d = d.loc[~d['Association'].str.contains('Increased Toxicity')]
cgi = d.copy()

#%%
###############################################################################
# prognostic
###############################################################################
#%% #################### CIVIC
d = civic.copy()
d = d[['Gene', 'TCGA_project', 'therapies', 'Level', 'therapy_interaction_type', 'evidence_direction', 'evidence_type', 'evidence_level', 'significance', 'molecular_profile_civic_url', 'source_type', 'Evidence statement', 'Evidence source','Database']].drop_duplicates()
d = d.loc[d['evidence_type']=='Prognostic'].drop_duplicates()
print(d['TCGA_project'].value_counts())
d1 = d[['Gene', 'TCGA_project', 'Level', 'Evidence source', 'Evidence statement', 'Database']].drop_duplicates() 

#%% #################### OncoKB
d = oncokb_prog.copy()
d = d[['Gene', 'TCGA_project', 'Level', 'Evidence source', 'Evidence statement', 'Database']]
d = d.explode('TCGA_project').reset_index(drop=True)
print(set(d['Gene'].unique()) - set(mrna_genes['gene_name'].unique())) # none
d = d.drop_duplicates()
d2 = d.copy()

#%% #################### combine sources
prog_bks = pd.concat([
    d1[['Gene','TCGA_project', 'Level', 'Evidence source', 'Evidence statement', 'Database']],
    d2[['Gene','TCGA_project', 'Level', 'Evidence source', 'Evidence statement', 'Database']]
    ], axis=0).drop_duplicates()
print(prog_bks['Level'].value_counts())
prog_bks = prog_bks.loc[prog_bks['Level']<='D']
prog_bks.to_csv(DATA_PATH + "processed/prog_bks.csv")


tmp = prog_bks[['TCGA_project','Level','Gene']].drop_duplicates()
tmp.groupby(['TCGA_project','Level']).size().unstack()

""" 
Level	A	B	C	D
TCGA_project				
TCGA-ACC	NaN	4.0	NaN	NaN
TCGA-BLCA	NaN	6.0	NaN	NaN
TCGA-BRCA	NaN	14.0	NaN	1.0
TCGA-CESC	NaN	5.0	NaN	NaN
TCGA-CHOL	NaN	4.0	NaN	NaN
TCGA-COAD	NaN	15.0	NaN	1.0
TCGA-DLBC	NaN	5.0	2.0	NaN
TCGA-ESCA	NaN	8.0	NaN	NaN
TCGA-GBM	NaN	9.0	NaN	1.0
TCGA-HNSC	1.0	12.0	NaN	NaN
TCGA-KICH	NaN	6.0	NaN	NaN
TCGA-KIRC	NaN	9.0	NaN	NaN
TCGA-KIRP	NaN	6.0	NaN	NaN
TCGA-LAML	11.0	21.0	NaN	NaN
TCGA-LGG	NaN	6.0	NaN	NaN
TCGA-LIHC	NaN	5.0	NaN	NaN
TCGA-LUAD	NaN	21.0	NaN	1.0
TCGA-LUSC	NaN	16.0	NaN	NaN
TCGA-MESO	NaN	4.0	NaN	NaN
TCGA-OV	NaN	10.0	NaN	NaN
TCGA-PAAD	NaN	9.0	NaN	1.0
TCGA-PCPG	NaN	3.0	NaN	NaN
TCGA-PRAD	NaN	9.0	NaN	1.0
TCGA-READ	NaN	15.0	NaN	1.0
TCGA-SARC	NaN	7.0	NaN	NaN
TCGA-SKCM	NaN	8.0	NaN	NaN
TCGA-STAD	NaN	9.0	NaN	4.0
TCGA-TGCT	NaN	3.0	NaN	NaN
TCGA-THCA	NaN	7.0	NaN	NaN
TCGA-THYM	NaN	3.0	NaN	NaN
TCGA-UCEC	NaN	7.0	NaN	NaN
TCGA-UCS	NaN	3.0	NaN	NaN
TCGA-UVM	NaN	4.0	1.0	NaN
"""

#%%
###############################################################################
# Predictive (drug)
###############################################################################
#%% ###################### CIVIC
d = civic.copy()
mask = (d['TCGA_project'].notna()) & (d['Gene'].notna()) & (d['Level'].notna())
d = d.loc[mask]
d = d.loc[d['evidence_type']=='Predictive']
# remove those with therapy interactions
d = d.loc[d['therapy_interaction_type'].isna(), ['Gene', 'therapies', 'TCGA_project', 'Level', 'Evidence source', 'Evidence statement', 'Database']].drop_duplicates()
d.rename(columns={'therapies': 'Drug'}, inplace=True)
d.drop_duplicates(inplace=True)
print(d['Drug'].value_counts())
d1 = d.copy()
print(d1['Level'].value_counts())

#%% ###################### CGI
d = cgi.copy()
mask = (d['TCGA_project'].notna()) & (d['Gene'].notna()) & (d['Level'].notna())
d = d.loc[mask]
d = d[['Gene', 'Drug', 'Level', 'TCGA_project', 'Evidence source', 'Evidence statement', 'Database']].drop_duplicates()
d = d.loc[(d['Drug']!='[]') & (d['Drug'].notna())]
# remove drug combinations / ignore therapy interactions
d = d.loc[(~d['Drug'].str.contains(';').astype(bool)) & (~d['Drug'].str.contains(',').astype(bool))] 
d['Drug'] = d['Drug'].str.strip('[] ')
d = d.drop_duplicates()
d2 = d.copy()
print(d2['Level'].value_counts()) 

#%% ###################### OncoKB
d = oncokb_therapeutic.copy()
d.rename(columns={'Drugs (for therapeutic implications only)': 'Drug'}, inplace=True)
d = d[['Gene', 'Drug', 'Level', 'TCGA_project', 'Evidence source', 'Evidence statement', 'Database']].drop_duplicates()
d['Drug'] = d['Drug'].str.split(r'[;,/.+]')
d = d.loc[d['Drug'].str.len() == 1]
d['Drug'] = d['Drug'].str[0]
d['Drug'] = d['Drug'].str.strip('[] ')

print(set(d['Gene'].unique()) - set(mrna_genes['gene_name'].unique())) 
d = d.loc[d['Gene']!='Other Biomarkers']

d.drop_duplicates(inplace=True)

d3 = d.copy()
print(d3['Level'].value_counts())

#%% ###################### combine sources
drug_bks = pd.concat([d1, d2, d3], axis=0).drop_duplicates()

### standardize drug name
## ding 2016
ding = pd.read_csv('/home/athan.li/eval_bk/data/TCGA/drug_name_standardization_fromDing2016.csv')
ding.index = ding['Recorded Name from TCGA'].values.flatten()
ding = ding[['Recorded Name from TCGA', 'Standard drug name']].drop_duplicates()
for i in range(len(drug_bks)):
    if (drug_bks.iloc[i, 1] in ding.index) and (ding.loc[drug_bks.iloc[i, 1], 'Standard drug name'] != drug_bks.iloc[i, 1]):
        print(f"Converting {drug_bks.iloc[i, 1]} to {ding.loc[drug_bks.iloc[i, 1], 'Standard drug name']}")
        drug_bks.iloc[i, 1] = ding.loc[drug_bks.iloc[i, 1], 'Standard drug name']
## insepction
variant_to_canonical = {
    "Imatinib": "Imatinib",
    "Imatinib Mesylate": "Imatinib",
    "Akt Inhibitor MK2206": "MK2206",
    "MK2206": "MK2206",
    "PI103": "PI-103",
    "PI-103": "PI-103",
    "AG-120": "Ivosidenib",
    "Ivosidenib": "Ivosidenib",
    "BYL719": "Alpelisib",
    "Alpelisib": "Alpelisib",
    "LEE011": "Ribociclib",
    "Ribociclib": "Ribociclib",
    "PRMT5 Inhibitor AMG 193": "AMG 193",
    "AMG 193": "AMG 193",
    "PI3Kβ Inhibitor AZD8186": "AZD8186",
    "AZD8186": "AZD8186",
    "Ado-Trastuzumab Emtansine": "Trastuzumab Emtansine",
    "Trastuzumab Emtansine": "Trastuzumab Emtansine",
    "Trastuzumab Deruxtecan": "Trastuzumab Deruxtecan",
    "Trastuzumab deruxtecan-nxki": "Trastuzumab Deruxtecan",
    "Retaspimycin Hydrochloride": "Alvespimycin",
    "Alvespimycin": "Alvespimycin",
    "Temsirolimus": "Temsirolimus",
    "Tensirolimus": "Temsirolimus",
    "Fluorouracil": "Fluorouracil",
    "Flourouracil": "Fluorouracil",
    "Mitomycin": "Mitomycin",
    "Mytomycin C": "Mitomycin",
    "Fulvestrant": "Fulvestrant",
    "Fluvestrant": "Fulvestrant",
    "Zenocutuzumab": "Zenocutuzumab",
    "Zenocutuzumab-zbco": "Zenocutuzumab",
}
drug_bks['Drug'] = drug_bks['Drug'].replace(variant_to_canonical)

# drug_bks['Level'].value_counts()
drug_bks = drug_bks.loc[drug_bks['TCGA_project'].notna()]
drug_bks = drug_bks.drop_duplicates()

drug_bks = drug_bks.loc[drug_bks['Level']<='D'] 

drug_bks.to_csv(DATA_PATH + "processed/drug_bks.csv")
